"""
解压UCF101压缩文件后，运行此程序将视频文件移动到相应的训练/测试的文件夹中
"""
import os
import os.path


def get_train_test_lists(version='01'):
    """
    使用一个版本(可以为01,02,03)的训练/测试文件，得到相应的文件名，为文件移动做准备
    """
    # 获取视频文件列表的地址.
    test_file = os.path.join('ucfTrainTestlist', 'testlist' + version + '.txt')
    train_file = os.path.join('ucfTrainTestlist', 'trainlist' + version + '.txt')

    # 建立test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    # 建立train list(除去类标签数字)
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    # 设置字典
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups


def moving_files(file_groups):
    """
    移动文件到正确的位置
    """
    for category, videos in file_groups.items():

        # 对每个video进行移动
        for video in videos:

            # 得到视频的各部分信息
            parts = video.split('/')
            className = parts[0]
            filename = parts[1]

            # 检查这个类是否已经存在，若不存在，则创建文件夹
            if not os.path.exists(os.path.join(category, className)):
                print("Creating a folder for %s/%s" % (category, className))
                os.makedirs(os.path.join(category, className))

            # 检查是否已经移动过此文件(或者说此文件是否存在)
            if not os.path.exists(className + '/' + filename):
                print("Can't found %s to move. Skipping it." % filename)
                continue

            # 移动文件
            temp = os.path.join(category, className, filename)
            print("Moving %s to %s" % (filename, temp))
            os.rename(className + '/' + filename, temp)

    print("Done.")


def main():
    """
    浏览每一个训练/测试样本，并将其移动到正确位置
    """

    # 得到包含train_lists、test_lists的字典，方便移动文件
    file_groups = get_train_test_lists()

    # 移动文件
    moving_files(file_groups)


if __name__ == '__main__':
    main()
