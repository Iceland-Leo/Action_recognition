"""
此程序为每个视频采用inception v3预训练模型，提取最后一个平均池化层的输出作为特征，
每个视频提取共seq_length个特征值(每个特征值为2048维的向量)

seq_length表示每个视频提取的序列长度
    例如，seq_length = 40，表示每个视频有40个时间序列，每个序列为一个帧图片通过cnn网络得到的2048维的特征向量

class_limit表示要从中提取特征的前N个类。
    例如，class_limit = 8，表示仅提取数据集中前8个(按字母顺序排列)的类的特征。 然后在训练模型时设置相同的数字。
"""
import numpy as np
import tensorflow as tf
import os.path
from data import DataSet
from tqdm import tqdm
from PIL import Image
from tensorflow.contrib.slim.nets import inception


def preprocess_input(image):
    """图像预处理，使得像素值在-1-1之间"""
    return 2 * ((image / 255.0) - 0.5)


# 设置默认参数值
seq_length = 40  # 每个视频的时间序列长度
class_limit = None  # 要提取的类的数量，可以是1-101或None表示所有类

# 获取视频信息的列表([train|test], class, filename, nb frames)
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# 进度条功能
pbar = tqdm(total=len(data.data))

# inception_v3网络的图片输入尺寸，检查点文件
img_size = inception.inception_v3.default_image_size
checkpoint_file = os.path.join("inception_v3", "inception_v3.ckpt")

tf.reset_default_graph()
slim = tf.contrib.slim

# 设置输入
input_img = tf.placeholder("float", [None, img_size, img_size, 3])

# 载入inception v3模型
with tf.Session() as sess:
    arg_scope = inception.inception_v3_arg_scope()

    with slim.arg_scope(arg_scope):
        _, end_points = inception.inception_v3(input_img, is_training=False)

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

    for video_data in data.data:

        # 获取此视频时间序列特征的存放路径
        seq_path = os.path.join('data', 'sequences', video_data[2] + '-' + str(seq_length) +
                                '-features')

        # 检查是否已经存在该视频的特征
        if os.path.isfile(seq_path + '.npy'):
            pbar.update(1)
            continue

        # 获取此视频的所有帧图片的路径列表
        frame_images = data.get_frames_for_sample(video_data)

        # 所有帧图像分成seq_length组，要每组的第一张图像即可，即共有seq_length张图片
        frame_images = data.rescale_list(frame_images, seq_length)

        # 循环并提取特征来构建序列
        sequence = []
        for image_path in frame_images:
            # 重新调整图片的大小为299*299*3
            resize_img = Image.open(image_path).resize((img_size, img_size))
            resize_img = np.array(resize_img)

            # 转换成1*299*299*3
            reshaped_img = resize_img.reshape(-1, img_size, img_size, 3)

            # 预处理图片，使得像素值在0-1之间
            reshaped_img_norm = preprocess_input(reshaped_img)

            features = sess.run(end_points['AvgPool_1a'],
                                feed_dict={input_img: reshaped_img_norm})
            features = features.reshape(-1)
            sequence.append(features)

        # 保存序列信息.
        np.save(seq_path, sequence)

        pbar.update(1)

pbar.close()
