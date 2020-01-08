from .meshflow import generate_vertex_profiles
from .meshflow import mesh_warp_frame_slow
from .meshflow import mesh_warp_frame_fast
from .meshflow import motion_propagate_L1, motion_propagate_L2
from .meshflow import motion_propagate_fast
from .optimization import offline_optimize_path
from .optimization import real_time_optimize_path
from .optimization import parallel_optimize_path
from .optimization import cvx_optimize_path
from .utils import check_dir, get_logger, is_video, tic, toc
from tqdm import tqdm
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import time
import pickle


log = get_logger('meshflow')

motion_propagate = motion_propagate_fast
mesh_warp_frame = mesh_warp_frame_fast

OPTIMIZER = {
    'offline': offline_optimize_path,
    'real_time': real_time_optimize_path,
    'parallel': parallel_optimize_path,
    'cvx': cvx_optimize_path
}

FOURCC = {
    '.mp4': 'mp4v',
    '.avi': 'xvid'
}

DEBUG = False
SLOW = False

parser = argparse.ArgumentParser('Mesh Flow Stabilization')
parser.add_argument('source_path', type=str, help='input folder or file path')
parser.add_argument('output_dir', type=str, help='output folder')
parser.add_argument('-m', '--method', type=str, choices=list(OPTIMIZER.keys()), default="offline", help='stabilization method')
parser.add_argument('--save-plot', action='store_true', default=False, help='plot paths and motion vectors')
parser.add_argument('--plot-dir', type=str, default='data/plot', help='output graph folder')
parser.add_argument('--save-params', action='store_true', default=False, help='save parameters')
parser.add_argument('--params-dir', type=str, default='data/params', help='parameters folder')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--slow', action='store_true', default=False, help='use original slow method of motion propagation and mesh warping')


class MeshFlowStabilizer:

    def __init__(
            self,
            source_video,
            output_dir,
            plot_dir,
            params_dir,
            method='real_time',
            save_params=True,
            output_ext='.mp4'
    ):
        # block of size in mesh
        self.pixels = 16

        # motion propagation radius
        self.radius = 266

        # method
        self.method = method

        # flags
        self.stabilized = False
        self.frame_warped = False

        # input
        if not osp.exists(source_video):
            raise FileNotFoundError('source video not found')

        name, ext = osp.splitext(osp.basename(source_video))
        self.source_video = source_video

        # output
        if output_ext not in FOURCC:
            raise ValueError('output extension %s not available' % output_ext)
        else:
            self.output_ext = output_ext

        self.combined_path = osp.join(output_dir, name + '-combined' + self.output_ext)
        self.stabilized_path = osp.join(output_dir, name + '-stabilized' + self.output_ext)
        check_dir(output_dir)

        # params
        self.save_params = save_params
        self.params_path = osp.join(params_dir, name + '.pickle')
        self.params_dir = params_dir

        # plot
        self.vertex_profiles_dir = osp.join(plot_dir, 'paths', name)
        self.old_motion_vectors_dir = osp.join(plot_dir, 'old_motion_vectors', name)
        self.new_motion_vectors_dir = osp.join(plot_dir, 'new_motion_vectors', name)

        if self.save_params and osp.exists(self.params_path):
            self._load_params()

        else:
            # read video
            self._read_video()

    def _read_video(self):
        cap = cv2.VideoCapture(self.source_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=1000,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        # Take first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # preserve aspect ratio
        HORIZONTAL_BORDER = int(30)
        VERTICAL_BORDER = int((HORIZONTAL_BORDER * old_gray.shape[1]) / old_gray.shape[0])

        # motion meshes in x-direction and y-direction
        x_motion_meshes = []
        y_motion_meshes = []

        # path parameters
        x_paths = np.zeros((int(old_frame.shape[0] / self.pixels), int(old_frame.shape[1] / self.pixels), 1))
        y_paths = np.zeros((int(old_frame.shape[0] / self.pixels), int(old_frame.shape[1] / self.pixels), 1))

        if DEBUG:
            sum_read_t, sum_features_t, sum_optical_t, sum_motion_t, sum_expand_t, sum_profiles_t = 0, 0, 0, 0, 0, 0
        else:
            bar = tqdm(total=frame_count, ncols=100, desc="%-10s" % "read")

        frame_num = 1
        while frame_num < frame_count:

            # processing frames
            tic('cap')
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            read_t = toc('cap')

            # find corners in it
            tic('features')
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            features_t = toc('features')

            # calculate optical flow
            tic('optical')
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            optical_t = toc('optical')

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # estimate motion mesh for old_frame
            tic('motion')
            x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame)
            motion_t = toc('motion')

            # expand dimensions
            tic('expand')
            try:
                x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
                y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)

            except:
                x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
                y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)
            expand_t = toc('expand')

            # generate vertex profiles
            tic('profiles')
            x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)
            profiles_t = toc('profiles')

            # updates frames
            frame_num += 1
            old_frame = frame.copy()
            old_gray = frame_gray.copy()

            if DEBUG:
                print(
                    '\r%-10s: frame %4d, cap %5.2f, features %5.2f, optical %5.2f, motion %5.2f, expand %5.2f, profiles %5.2f, total %5.2f' %
                    ('read (ms)', frame_num, read_t, features_t, optical_t, motion_t, expand_t, profiles_t,
                     sum([read_t, features_t, optical_t, motion_t, expand_t, profiles_t])), end='    ')
                sum_read_t += read_t
                sum_features_t += features_t
                sum_optical_t += optical_t
                sum_motion_t += motion_t
                sum_expand_t += expand_t
                sum_profiles_t += profiles_t

            else:
                bar.update(1)

        cap.release()

        if DEBUG:
            print()
            log.info(
                'read time (s): cap %5.2f, features %5.2f, optical %5.2f, motion %5.2f, expand %5.2f, profiles %5.2f, total %5.2f' %
                (sum_read_t / 1000, sum_features_t / 1000, sum_optical_t / 1000,
                 sum_motion_t / 1000, sum_expand_t / 1000, sum_profiles_t / 1000,
                 sum([sum_read_t, sum_features_t, sum_optical_t, sum_motion_t, sum_expand_t, sum_profiles_t]) / 1000))
        else:
            bar.close()

        self.horizontal_border = HORIZONTAL_BORDER
        self.vertical_border = VERTICAL_BORDER
        self.x_motion_meshes = x_motion_meshes
        self.y_motion_meshes = y_motion_meshes
        self.x_paths = x_paths
        self.y_paths = y_paths

    def _stabilize(self):
        if not self.stabilized:
            # optimize for smooth vertex profiles
            self.sx_paths = OPTIMIZER[self.method](self.x_paths)
            self.sy_paths = OPTIMIZER[self.method](self.y_paths)

            if self.save_params:
                self._save_params()

            # set flag
            self.stabilized = True

    def _get_frame_warp(self):
        if not self.frame_warped:
            self.x_motion_meshes_2d = np.concatenate(
                (self.x_motion_meshes, np.expand_dims(self.x_motion_meshes[:, :, -1], axis=2)), axis=2)
            self.y_motion_meshes_2d = np.concatenate(
                (self.y_motion_meshes, np.expand_dims(self.y_motion_meshes[:, :, -1], axis=2)), axis=2)
            self.new_x_motion_meshes = self.sx_paths - self.x_paths
            self.new_y_motion_meshes = self.sy_paths - self.y_paths

            # set flag
            self.frame_warped = True

    def _load_params(self):
        # read params
        with open(self.params_path, 'rb') as f:
            params_dict = pickle.load(f)

        self.pixels = params_dict['pixels']
        self.radius = params_dict['radius']
        self.horizontal_border = params_dict['horizontal_border']
        self.vertical_border = params_dict['vertical_border']
        self.x_motion_meshes = params_dict['x_motion_meshes']
        self.y_motion_meshes = params_dict['y_motion_meshes']
        self.x_paths = params_dict['x_paths']
        self.y_paths = params_dict['y_paths']
        self.sx_paths = params_dict['sx_paths']
        self.sy_paths = params_dict['sy_paths']

        # set flag
        self.stabilized = True

    def _save_params(self):
        check_dir(self.params_dir)

        params_dict = {
            'pixels': self.pixels,
            'radius': self.radius,
            'horizontal_border': self.horizontal_border,
            'vertical_border': self.vertical_border,
            'x_motion_meshes': self.x_motion_meshes,
            'y_motion_meshes': self.y_motion_meshes,
            'x_paths': self.x_paths,
            'y_paths': self.y_paths,
            'sx_paths': self.sx_paths,
            'sy_paths': self.sy_paths
        }

        # write params
        with open(self.params_path, 'wb') as f:
            pickle.dump(params_dict, f)

    def generate_stabilized_video(self):
        self._stabilize()
        self._get_frame_warp()

        cap = cv2.VideoCapture(self.source_video)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*FOURCC[self.output_ext])

        combined_shape = (2 * frame_width, frame_height)
        combined_out = cv2.VideoWriter(self.combined_path, fourcc, frame_rate, combined_shape)
        stabilized_shape = (frame_width, frame_height)
        stabilized_out = cv2.VideoWriter(self.stabilized_path, fourcc, frame_rate, stabilized_shape)

        if DEBUG:
            sum_read_t, sum_warp_t, sum_resize_t, sum_write_t = 0, 0, 0, 0
        else:
            bar = tqdm(total=frame_count, ncols=100, desc="%-10s" % "write")

        frame_num = 0
        while frame_num < frame_count:
            try:
                # reconstruct from frames
                tic('cap')
                ret, frame = cap.read()
                new_x_motion_mesh = self.new_x_motion_meshes[:, :, frame_num]
                new_y_motion_mesh = self.new_y_motion_meshes[:, :, frame_num]
                read_t = toc('cap')

                # mesh warping
                tic('warp')
                new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
                warp_t = toc('warp')

                # resize
                tic('resize')
                new_frame = new_frame[
                            self.horizontal_border : -self.horizontal_border,
                            self.vertical_border : -self.vertical_border, :]
                new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
                resize_t = toc('resize')

                # write frame
                tic('write')
                combined_out.write(np.concatenate((frame, new_frame), axis=1))
                stabilized_out.write(new_frame)
                write_t = toc('write')

                # count
                frame_num += 1

                # debug
                if DEBUG:
                    print('\r%-10s: frame %4d, cap %5.2f, warp %5.2f, resize %5.2f, write %5.2f, total %5.2f' %
                          ('write (ms)', frame_num, read_t, warp_t, resize_t, write_t,
                           sum([read_t, warp_t, resize_t, write_t])), end='    ')
                    sum_read_t += read_t
                    sum_warp_t += warp_t
                    sum_resize_t += resize_t
                    sum_write_t += write_t

                else:
                    bar.update(1)

            except IndexError:
                break

        cap.release()
        combined_out.release()
        stabilized_out.release()

        if DEBUG:
            print()
            log.info('write time (s): cap %5.2f, warp %5.2f, resize %5.2f, write %5.2f, total %5.2f' %
                  (sum_read_t / 1000, sum_warp_t / 1000, sum_resize_t / 1000, sum_write_t / 1000,
                   sum([sum_read_t, sum_warp_t, sum_resize_t, sum_write_t]) / 1000))
        else:
            bar.close()

    def plot_vertex_profiles(self):
        self._stabilize()

        # check dir
        check_dir(self.vertex_profiles_dir)

        for i in range(0, self.x_paths.shape[0]):
            for j in range(0, self.x_paths.shape[1], 10):
                plt.plot(self.x_paths[i, j, :])
                plt.plot(self.sx_paths[i, j, :])
                plt.savefig(osp.join(self.vertex_profiles_dir, '%d_%d.png' % (i, j)))
                plt.clf()

    def plot_motion_vectors(self):
        self._stabilize()
        self._get_frame_warp()

        # check dir
        check_dir(self.old_motion_vectors_dir, self.new_motion_vectors_dir)

        # read video
        cap = cv2.VideoCapture(self.source_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = 0
        bar = tqdm(total=frame_count, ncols=100)
        while frame_num < frame_count:
            try:
                # reconstruct from frames
                ret, frame = cap.read()
                x_motion_mesh = self.x_motion_meshes[:, :, frame_num]
                y_motion_mesh = self.y_motion_meshes[:, :, frame_num]
                new_x_motion_mesh = self.new_x_motion_meshes[:, :, frame_num]
                new_y_motion_mesh = self.new_y_motion_meshes[:, :, frame_num]

                # mesh warping
                new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
                new_frame = new_frame[self.horizontal_border:-self.horizontal_border,
                            self.vertical_border:-self.vertical_border, :]
                new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

                # draw old motion vectors
                r = 5
                for i in range(x_motion_mesh.shape[0]):
                    for j in range(x_motion_mesh.shape[1]):
                        theta = np.arctan2(y_motion_mesh[i, j], x_motion_mesh[i, j])
                        cv2.line(frame, (j * self.pixels, i * self.pixels),
                                 (int(j * self.pixels + r * np.cos(theta)), int(i * self.pixels + r * np.sin(theta))),
                                 1)
                cv2.imwrite(osp.join(self.old_motion_vectors_dir, str(frame_num) + '.png'), frame)

                # draw new motion vectors
                for i in range(new_x_motion_mesh.shape[0]):
                    for j in range(new_x_motion_mesh.shape[1]):
                        theta = np.arctan2(new_y_motion_mesh[i, j], new_x_motion_mesh[i, j])
                        cv2.line(new_frame, (j * self.pixels, i * self.pixels),
                                 (int(j * self.pixels + r * np.cos(theta)), int(i * self.pixels + r * np.sin(theta))),
                                 1)
                cv2.imwrite(osp.join(self.new_motion_vectors_dir, str(frame_num) + '.png'), new_frame)

                frame_num += 1
                bar.update(1)

            except:
                break

        bar.close()


def process_file(args):
    tic('all')
    log.info(args.source_path)

    mfs = MeshFlowStabilizer(source_video=args.source_path,
                             output_dir=args.output_dir,
                             plot_dir=args.plot_dir,
                             params_dir=args.params_dir,
                             method=args.method,
                             save_params=args.save_params)
    mfs.generate_stabilized_video()

    if args.save_plot:
        mfs.plot_motion_vectors()
        mfs.plot_vertex_profiles()

    log.info('time elapsed (s): %.2f' % (toc('all') / 1000))


def process_dir(args):
    dir_path = args.source_path
    filenames = os.listdir(dir_path)

    for filename in filenames:
        if is_video(filename):
            args.source_path = osp.join(dir_path, filename)
            process_file(args)


def main(args):
    global motion_propagate, mesh_warp_frame, DEBUG

    DEBUG = args.debug

    if args.slow:
        motion_propagate = motion_propagate_L2
        mesh_warp_frame = mesh_warp_frame_slow
        
    else:
        motion_propagate = motion_propagate_fast
        mesh_warp_frame = mesh_warp_frame_fast

    if osp.exists(args.source_path):
        if osp.isdir(args.source_path):
            process_dir(args)

        else:
            process_file(args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
