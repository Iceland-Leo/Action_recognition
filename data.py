"""
该类用于管理数据
"""
import csv
import numpy as np
import glob
import os.path
import operator


class DataSet:

    def __init__(self, seq_length=40, class_limit=None):
        """
        seq_length = int类型，所考虑的视频帧图片总数
        class_limit = int类型，限制类别数量，None表示无限制.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 300  # 限制能使用的视频的最大帧数为300

        # 获取数据列表([train|test], class, filename, nb frames)
        self.data = self.get_data()

        # 获取类别名列表(若类别数量有限制，则返回限制内的类别名列表)
        self.classes = self.get_classes()

        # 将视频帧数限制在seq_length--max_frames之间，供后续使用
        self.data = self.clean_data()

    @staticmethod
    def get_data():
        """从data_file.csv中获取数据"""
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        # 删除data_file.csv文件中的空白行
        for line in data:
            if len(line) == 0:
                data.remove(line)

        return data

    def clean_data(self):
        """将视频样本的帧数frames限制在seq_length <= frames <= max_frames(仅在需要使用的类别中)。"""
        data_clean = []
        for item in self.data:
            if self.seq_length <= int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """从data中提取class。 如果class有限制，则返回需要的类别名列表"""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # 对类别按字典顺序排序
        classes = sorted(classes)

        # 返回需要的类别名列表
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """给定一个类名，在类列表中返回它的编号。并对它进行one-hot编码用于训练"""
        # 获取类的编号
        class_index = self.classes.index(class_str)

        # one-hot编码.
        class_one_hot = np.eye(len(self.classes), dtype='float')[class_index]

        assert len(class_one_hot) == len(self.classes)

        return class_one_hot

    def split_train_test(self):
        """将数据分成训练组和测试组"""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test):
        """
        试图将所有内容加载到内存中，以便我们可以更快地训练
        """
        # 得到正确的数据集.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        x, y = [], []
        for row in data:
            sequence = self.get_extracted_sequence(row)  # 得到40*2048的特征数据

            if sequence is None:
                print("Can't found sequence. Did you generate them?")
                exit(1)

            x.append(sequence)
            y.append(self.get_class_one_hot(row[1]))  # 某个类别的One-Hot编码

        return np.array(x), np.array(y)

    def get_extracted_sequence(self, sample):
        """获取保存的已经提取到的cnn特征"""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) +
                            '-' + 'features.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename):
        """给定一个样本的文件名，返回该样本的序列特征值"""
        # 寻找样本在data_file.csv中的行
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't found sample: %s" % filename)

        # 获取样本的序列特征值.
        sequence = self.get_extracted_sequence(sample)

        if sequence is None:
            raise ValueError("Can't found sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """从数据文件中给出一个样本行([train|test], class, filename, nb frames)，获取相应的所有帧图片。"""
        video_path = os.path.join('data', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(video_path, filename + '*jpg')))
        return images

    @staticmethod
    def rescale_list(input_list, size):
        """
        给定一个列表input_list和一个大小size，返回一个重新标定的样本列表。
        例如，如果想要一个大小为5的列表并且我们有一个大小为30的列表，
        那么返回一个新的列表，其大小为5，它是原始列表每隔30/5=6个元素提取到的列表
        """
        assert len(input_list) >= size

        # 获取跳过的帧数，上述例子中，skip= 30//5 = 6
        skip = len(input_list) // size

        # 建立新的列表
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # 截断最后，保证输出列表大小为size
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """给出经过softmax处理预测概率向量，返回前nb_to_return类的类名称"""
        # 获得每个类的预测概率
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # 对概率进行降序排序
        # key为定义的函数，指定使用label——prediction的第1维(即字典的值而非键值)进行排序
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # 返回前nb_to_return类的类名称
        for i, class_prediction in enumerate(sorted_lps):
            # 若超出nb_to_return或者类预测概率<0.001，则退出循环
            if i > nb_to_return - 1 or class_prediction[1] <= 0.001:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
