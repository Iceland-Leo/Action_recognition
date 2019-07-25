"""
利用提取到的视频特征训练带窥视孔的LSTM网络
"""
import os.path
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import lstm_inference
from sklearn.utils import shuffle
from data import DataSet

MODEL_SAVE_PATH = "data//test_checkpoints"
MODEL_NAME = "lstm_model.ckpt"
SUMMARY_DIR = "data//logs"


def train(seq_length, feature_length, class_limit=None, batch_size=32, nb_epoch=100):

    # 获取数据
    data = DataSet(seq_length=seq_length, class_limit=class_limit)

    # 获取训练数据(每个视频提取到的40个2048维的特征)
    # x_train为特征数组(数组个数为seq_length)，y_train为类别的One-Hot编码数组(下同)
    x_train, y_train = data.get_all_sequences_in_memory('train')
    x_test, y_test = data.get_all_sequences_in_memory('test')

    # 随机打乱样本的顺序
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # 确定训练的类数量
    if class_limit is None:
        class_num = 101
    else:
        class_num = class_limit

    # 设置模型相关参数
    learning_rate_base = 0.0001
    learning_rate_decay = 0.9
    step_per_epoch = math.ceil(len(x_train) / batch_size)
    regularizer_rate = 0.2

    with tf.name_scope("input"):
        # 设置输入
        input_x = tf.placeholder(tf.float32, [None, seq_length, feature_length], name="input-x")
        input_y = tf.placeholder(tf.float32, [None, class_num], name="input-y")

    # 设置正则化项
    regularizer = tf.contrib.layers.l2_regularizer(regularizer_rate)

    # 训练LSTM模型前向传播的计算
    logits, inference_y = lstm_inference.inference(input_x, class_num, regularizer)

    with tf.name_scope("loss"):
        # 设置平均交叉熵为误差
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))
        tf.summary.scalar("cross_entropy", cross_entropy)
        # 计算总loss损失
        loss = cross_entropy + tf.add_n(tf.get_collection("losses"))

    # 初始化全局迭代次数
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("train"):
        # 设置退化学习率(所有训练样本训练结束一次时更新学习率)
        learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, step_per_epoch, learning_rate_decay)
        tf.summary.scalar("learning-rate", learning_rate)

        # 定义优化器
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 测试时LSTM的前向传播计算
    _, output_y = lstm_inference.inference(input_x, class_num, None, train=False, reuse=True)

    with tf.name_scope("accuracy"):
        # 训练时的准确率
        correct_prediction_train = tf.equal(tf.argmax(inference_y, 1), tf.argmax(input_y, 1))
        acc_train = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))
        tf.summary.scalar("acc_train", acc_train)
        # 测试时准确率
        correct_prediction = tf.equal(tf.argmax(output_y, 1), tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 保存变量
    saver = tf.train.Saver(max_to_keep=1)

    # 整理所有日志生成
    merged = tf.summary.merge_all()

    # 启动Session训练
    with tf.Session() as sess:
        # 初始化写日志的writer，并将当前Tensorflow计算图写入日志
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        # 初始化所有变量
        tf.global_variables_initializer().run()

        # 当前最高准确率，用于确定准确率是否提高
        last_test_acc = 0.0

        # 设置准确率未提高次数的计数器
        acc_not_improved = 0

        # 准确率未提高的最大次数，用于结束训练
        max_acc_not_improved = 30

        test_acc_list = []

        for i in range(nb_epoch):
            print("epoch %d:" % (i + 1))
            for j in range(step_per_epoch):
                start = j * batch_size
                end = min(start + batch_size, len(x_train))
                # 网络优化过程
                _, loss_value, step, summary = sess.run([train_step, cross_entropy, global_step, merged],
                                                        feed_dict={input_x: x_train[start:end],
                                                                   input_y: y_train[start:end]})
                # 将所有日志写入文件，供给TensorBoard使用
                summary_writer.add_summary(summary, step)
                # 打印信息
                print("\tAfter %d training step(s), loss on training batch is %g" %
                      (step, loss_value))

            train_acc = sess.run(accuracy, feed_dict={input_x: x_train,
                                                input_y: y_train})

            # 测试集的准确率
            test_acc = sess.run(accuracy, feed_dict={input_x: x_test, input_y: y_test})
            test_acc_list.append(test_acc)
            print("After %d epoch(es), accuracy on train is %g, accuracy on test is %g" %
                  ((i + 1), train_acc, test_acc))

            # 是否保存现在的模型
            if test_acc > last_test_acc:
                last_test_acc = test_acc
                print("accuracy improved, saved model")
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
                acc_not_improved = 0
            else:
                acc_not_improved += 1
                print("accuracy not improved")

            # 若达到训练结束条件，则停止训练
            if acc_not_improved >= max_acc_not_improved:
                break
        # 作图
        temp = 0
        x = []
        for i in range(len(test_acc_list)):
            temp += 30
            x.append(temp)
        plt.plot(x, test_acc_list, linewidth=3, color='black', marker='o', markerfacecolor='white', markersize=3)
        plt.xlabel('train_step')
        plt.ylabel('test_accuracy')
        plt.legend()
        plt.show()
        print("\n\nEnd of training")


def main():
    """主要训练参数的设置"""
    class_limit = 10  # 整型，可以是1-101或者None
    seq_length = 40  # 每个视频的时间序列长度，与提取CNN特征时的seq_length保持一致
    feature_length = 2048  # 每个特征向量的长度，此处为2048
    batch_size = 32
    nb_epoch = 100  # 迭代轮数

    train(seq_length, feature_length, class_limit=class_limit, batch_size=batch_size, nb_epoch=nb_epoch)


if __name__ == '__main__':
    main()
