#
#   Utilities
#   Written by Liang-Han, Lin
#   Created at 2020.1.1
#

from datetime import datetime
from pytz import timezone, utc
import logging
import os
import os.path as osp
import sys

#
#   Logging
#
log = None


#
#   Config
#
debug = False


def get_logger(name, save_path='', tz='Asia/Taipei'):
    global log

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)-.1s %(asctime)s %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    # output logs to file
    if save_path != '':
        file_handler = logging.FileHandler(save_path)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    # set timezone
    def custom_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        converted = utc_dt.astimezone(timezone(tz))
        return converted.timetuple()

    logging.Formatter.converter = custom_time

    return log


#
#   File
#
def is_image(path):
    return osp.splitext(path)[1].lower() in ['.jpg', '.png', '.gif', '.tiff']


def is_annotation(path):
    return osp.splitext(path)[1].lower() in ['.json', '.geojson']


def is_video(path):
    return osp.splitext(path)[1].lower() in ['.avi', '.mp4', '.mov']


def check_dir(*dir_paths):
    for p in dir_paths:
        if not osp.exists(p):
            os.makedirs(p, exist_ok=True)
