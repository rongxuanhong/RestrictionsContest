import os
from datetime import datetime
import logging


def compute_time_consumed(start_time):
    """
    计算训练总耗时
    :param start_time:
    :return:
    """
    time_elapsed = datetime.now() - start_time
    seconds = time_elapsed.seconds
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 3600 % 60
    print("本次训练共耗时 {0} 时 {1} 分 {2} 秒".format(hour, minute, second))


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging
