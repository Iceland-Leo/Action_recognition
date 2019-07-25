"""
使用moving_files移动所有文件后，运行此文件从视频中提取图像，并创建一个.csv数据文件，用于训练和测试。
"""
import csv
import glob
import os
import os.path
from subprocess import call


def extracting_files():
    """
    将所有视频分成训练和测试数据后，需要创建一个数据文件，方便在训练时参考该文件，以获取视频信息。

    首先从每个视频中提取帧图片，可有下面的ffmpeg方法实现
    `ffmpeg -i video.mpg image-%04d.jpg`

    再在文件中记录视频的下列数据：
    [train|test], class, filename, nb frames
    """
    data_file = []
    folders = ['train', 'test']

    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))  # 返回train文件夹下所有类的文件夹的目录

        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.avi'))  # 返回某个类文件夹下的所有视频目录

            for video_path in class_files:
                # 获取文件的各部分信息
                video_parts = get_video_parts(video_path)

                train_or_test, classname, filename_no_ext, filename = video_parts

                # 如果没有提取帧数，则提取，若已经提取，则读取信息
                if not check_already_extracted(video_parts):

                    src = os.path.join(train_or_test, classname, filename)
                    img_path = os.path.join(train_or_test, classname,
                                        filename_no_ext + '-%04d.jpg')

                    # ffmpeg -i 00001.M.avi -r -f image2 picture/image%06d.jpg(-f内容可缺省，作用是将视频转换成x个图像)
                    # -i：设定输入流，
                    # 00001.M.avi：视频路径，可以替换为任意视频路径，
                    # -r：设定帧速率，默认为25，
                    # -f:设定输出格式，
                    # image2:表示输出格式为图片，
                    # picture/image%06d.jpg：表示图片输出路径，这里%06d表示图片命名格式
                    call(["ffmpeg", "-i", src, img_path])

                # 获取视频的总帧数
                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files successfully." % (len(data_file)))


def get_nb_frames_for_video(video_parts):
    """给定已提取视频的视频部分信息，返回提取的帧数。"""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(train_or_test, classname,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)


def get_video_parts(video_path):
    """给定视频的完整路径，返回各部分信息。"""
    parts = video_path.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0]  # 去掉扩展名
    classname = parts[1]
    train_or_test = parts[0]

    return train_or_test, classname, filename_no_ext, filename


def check_already_extracted(video_parts):
    """检查是否提取过该视频的帧数。"""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))


def main():
    """
    从视频中提取图像，并构建一个可用作数据输入的新文件(data_file.csv)。
    文件中元素的格式：
    [train|test], class, fileName, nb frames
    """
    extracting_files()


if __name__ == '__main__':
    main()
