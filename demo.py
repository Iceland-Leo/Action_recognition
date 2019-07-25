"""
给定视频路径和保存的模型文件（检查点），对其进行分类。
首先应该提取视频的seq_length个cnn特征，然后放入lstm进行预测
"""
import tensorflow as tf
import numpy as np
import lstm_inference
from data import DataSet


def predict(seq_length, class_limit, feature_length, saved_model, video_name):

    # 获取数据
    data = DataSet(seq_length=seq_length, class_limit=class_limit)
    
    # 利用视频名称提取样本的序列特征值
    sample = data.get_frames_by_filename(video_name)
    sample = np.reshape(sample, [-1, seq_length, feature_length])

    # 确定训练的类数量
    if class_limit is None:
        class_num = 101
    else:
        class_num = class_limit

    # 设置输入
    input_x = tf.placeholder(tf.float32, [None, seq_length, feature_length], name="input-x")

    # 前向传播的计算
    _, output_y = lstm_inference.inference(input_x, class_num, None, train=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model = tf.train.get_checkpoint_state(saved_model)
        if model and model.model_checkpoint_path:
            saver.restore(sess, model.model_checkpoint_path)
            prediction = sess.run(output_y, feed_dict={input_x: sample})
            print("the prediction of the video %s is:" % video_name)
            data.print_class_from_prediction(np.squeeze(prediction, axis=0))  # 删除prediction中第1维(并且该维大小为1)


def main():
    # 保存的检查点文件路径
    saved_model = 'data/checkpoints/'
    # 以下设置必须与训练中使用的长度相匹配
    seq_length = 40
    class_limit = 10
    feature_length = 2048

    video_name = 'v_BasketballDunk_g01_c01'

    predict(seq_length, class_limit, feature_length, saved_model,  video_name)


if __name__ == '__main__':
    main()
