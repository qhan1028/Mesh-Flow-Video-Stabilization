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
import numpy as np
import time


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

#
#   Utils - Timer
#   Written by Liang-Han, Lin
#   Created at 2019.2.1
#

class Timer:

    def __init__(self, update=True):
        self.stage, self.start = {}, {}
        self.update = update
        self.start_anonymous = time.time() * 1000

    def tic(self, name=None):
        if name:
            self.start[name] = time.time() * 1000
        else:
            self.start_anonymous = time.time() * 1000

    def toc(self, name=None):
        if name in self.start.keys():
            period = time.time() * 1000 - self.start[name]

            if name not in self.stage.keys():
                self.add_stage(name)
            self.update_min_max(name, period)

        else:
            period = time.time() * 1000 - self.start_anonymous

        return period

    def add_stage(self, name):
        if name not in self.stage.keys():
            self.stage[name] = {'min': np.inf, 'max': 0, 'avg': 0}

    def del_stage(self, name):
        self.stage.pop(name, None)

    def enable_update(self):
        self.update = True

    def disable_update(self):
        self.update = False

    def update_min_max(self, name, t):
        if self.update and name in self.stage.keys():
            if t < self.stage[name]['min']:
                self.stage[name]['min'] = t
            if t > self.stage[name]['max']:
                self.stage[name]['max'] = t

            new_avg = self.stage[name]['avg'] * 0.9 + t * 0.1
            self.stage[name]['avg'] = new_avg

    def summary(self):
        print('\n%15s: %8s %12s %12s' % ('Stage', 'Min', 'Max', 'Avg'))
        for name, t in self.stage.items():
            print('%15s' % name + ': %8.4f ms, %8.4f ms, %8.4f ms' % (t['min'], t['max'], t['avg']))


#
# Timer Program Global Function
#
_timer = Timer()


def tic(name=None):
    _timer.tic(name)


def toc(name=None):
    return _timer.toc(name)
