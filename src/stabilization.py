from .meshflow import generate_vertex_profiles
from .meshflow import mesh_warp_frame
from .meshflow import motion_propagate
from .optimization import real_time_optimize_path
from .utils import check_dir, get_logger, is_video
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


parser = argparse.ArgumentParser('Mesh Flow Stabilization')
parser.add_argument('source_path', type=str, help='input folder or file path')
parser.add_argument('output_dir', type=str, help='output folder')
parser.add_argument('--plot', action='store_true', default=False, help='plot paths and motion vectors')
parser.add_argument('--plot_dir', type=str, default='data/plot', help='output graph folder')
parser.add_argument('--save-params', action='store_true', default=False, help='save parameters')
parser.add_argument('--params_dir', type=str, default='data/params', help='parameters folder')


class MeshFlowStabilizer:

    def __init__(self, source_video, output_dir, plot_dir, params_dir, save=True):
        # block of size in mesh
        self.pixels = 16

        # motion propagation radius
        self.radius = 300

        if not osp.exists(source_video):
            raise FileNotFoundError('source video not found')

        # setup dir
        name, ext = osp.splitext(osp.basename(source_video))
        self.source_video = source_video

        self.vertex_profiles_dir = osp.join(plot_dir, 'paths', name)
        self.old_motion_vectors_dir = osp.join(plot_dir, 'old_motion_vectors', name)
        self.new_motion_vectors_dir = osp.join(plot_dir, 'new_motion_vectors', name)

        self.combined_path = osp.join(output_dir, name + '-combined' + ext)
        self.stabilized_path = osp.join(output_dir, name + '-stabilized' + ext)
        self.params_path = osp.join(params_dir, name + '.pickle')
        check_dir(output_dir)
        
        self.save = save
        self.stabilized = False
        self.frame_warped = False
        
        if self.save and osp.exists(self.params_path):
            self._load_params()

        else:
            # read video
            self._read_video()

    def _read_video(self):
        cap = cv2.VideoCapture(self.source_video)
        self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

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

        frame_num = 1
        bar = tqdm(total=self.frame_count, ascii=False, desc="read")
        while frame_num < self.frame_count:

            # processing frames
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find corners in it
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # estimate motion mesh for old_frame
            x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame)
            try:
                x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
                y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)

            except:
                x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
                y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)

            # generate vertex profiles
            x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

            # updates frames
            bar.update(1)
            frame_num += 1
            old_frame = frame.copy()
            old_gray = frame_gray.copy()

        cap.release()
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
            self.sx_paths = real_time_optimize_path(self.x_paths)
            self.sy_paths = real_time_optimize_path(self.y_paths)
            self.stabilized = True
            
            if self.save:
                self._save_params()

    def _get_frame_warp(self):
        if not self.frame_warped:
            self.x_motion_meshes_2d = np.concatenate((self.x_motion_meshes, np.expand_dims(self.x_motion_meshes[:, :, -1], axis=2)), axis=2)
            self.y_motion_meshes_2d = np.concatenate((self.y_motion_meshes, np.expand_dims(self.y_motion_meshes[:, :, -1], axis=2)), axis=2)
            self.new_x_motion_meshes = self.sx_paths - self.x_paths
            self.new_y_motion_meshes = self.sy_paths - self.y_paths
            self.frame_warped = True
            
    def _load_params(self):
        with open(self.params_path, 'rb') as f:
            params_dict = pickle.load(f)
            
        self.pixels = params_dict['pixels']
        self.radius = params_dict['radius']
        self.frame_rate = params_dict['frame_rate']
        self.frame_width = params_dict['frame_width']
        self.frame_height = params_dict['frame_height']
        self.frame_count = params_dict['frame_count']
        self.horizontal_border = params_dict['horizontal_border']
        self.vertical_border = params_dict['vertical_border']
        self.x_motion_meshes = params_dict['x_motion_meshes']
        self.y_motion_meshes = params_dict['y_motion_meshes']
        self.x_paths = params_dict['x_paths']
        self.y_paths = params_dict['y_paths']
        self.sx_paths = params_dict['sx_paths']
        self.sy_paths = params_dict['sy_paths']
        self.stabilized = True
        
    def _save_params(self):
        params_dict = {
            'pixels': self.pixels,
            'radius': self.radius,
            'frame_rate': self.frame_rate,
            'frame_height': self.frame_height,
            'frame_width': self.frame_width,
            'frame_count': self.frame_count,
            'horizontal_border': self.horizontal_border,
            'vertical_border': self.vertical_border,
            'x_motion_meshes': self.x_motion_meshes,
            'y_motion_meshes': self.y_motion_meshes,
            'x_paths': self.x_paths,
            'y_paths': self.y_paths,
            'sx_paths': self.sx_paths,
            'sy_paths': self.sy_paths
        }
        with open(self.params_path, 'wb') as f:
            pickle.dump(params_dict, f)

    def generate_stabilized_video(self):
        self._stabilize()
        self._get_frame_warp()

        cap = cv2.VideoCapture(self.source_video)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        combined_shape = (2 * self.frame_width, self.frame_height)
        combined_out = cv2.VideoWriter(self.combined_path, fourcc, self.frame_rate, combined_shape)
        stabilized_shape = (self.frame_width, self.frame_height)
        stabilized_out = cv2.VideoWriter(self.stabilized_path, fourcc, self.frame_rate, stabilized_shape)

        frame_num = 0

        bar = tqdm(total=self.frame_count, ascii=False, desc="output")
        while frame_num < self.frame_count:
            try:
                # reconstruct from frames
                ret, frame = cap.read()
                new_x_motion_mesh = self.new_x_motion_meshes[:, :, frame_num]
                new_y_motion_mesh = self.new_y_motion_meshes[:, :, frame_num]

                # mesh warping
                new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
                new_frame = new_frame[self.horizontal_border:-self.horizontal_border, self.vertical_border:-self.vertical_border, :]
                new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

                # write frame
                combined_out.write(np.concatenate((frame, new_frame), axis=1))
                stabilized_out.write(new_frame)

                # count
                frame_num += 1
                bar.update(1)

            except:
                break

        bar.close()
        cap.release()
        combined_out.release()
        stabilized_out.release()

    def plot_vertex_profiles(self):
        check_dir(self.vertex_profiles_dir)

        if self.stabilized:
            for i in range(0, self.x_paths.shape[0]):
                for j in range(0, self.x_paths.shape[1], 10):
                    plt.plot(self.x_paths[i, j, :])
                    plt.plot(self.sx_paths[i, j, :])
                    plt.savefig(osp.join(self.vertex_profiles_dir, str(i) + '_' + str(j) + '.png'))
                    plt.clf()

    def plot_motion_vectors(self):
        self._stabilize()
        self._get_frame_warp()
        check_dir(self.old_motion_vectors_dir, self.new_motion_vectors_dir)

        frame_num = 0
        cap = cv2.VideoCapture(self.source_video)
        bar = tqdm(total=self.frame_count, ascii=False)
        while frame_num < self.frame_count:
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
                                 (int(j * self.pixels + r * np.cos(theta)), int(i * self.pixels + r * np.sin(theta))), 1)
                cv2.imwrite(osp.join(self.old_motion_vectors_dir, str(frame_num) + '.png'), frame)

                # draw new motion vectors
                for i in range(new_x_motion_mesh.shape[0]):
                    for j in range(new_x_motion_mesh.shape[1]):
                        theta = np.arctan2(new_y_motion_mesh[i, j], new_x_motion_mesh[i, j])
                        cv2.line(new_frame, (j * self.pixels, i * self.pixels),
                                 (int(j * self.pixels + r * np.cos(theta)), int(i * self.pixels + r * np.sin(theta))), 1)
                cv2.imwrite(osp.join(self.new_motion_vectors_dir, str(frame_num) + '.png'), new_frame)

                frame_num += 1
                bar.update(1)
                
            except:
                break

        bar.close()


def process_file(args):
    log.info(args.source_path)

    start_time = time.time()

    mfs = MeshFlowStabilizer(args.source_path, args.output_dir, args.plot_dir, args.params_dir)
    mfs.generate_stabilized_video()

    if args.plot:
        mfs.plot_motion_vectors()
        mfs.plot_vertex_profiles()

    log.info('Time elapsed: %.2f' % (time.time() - start_time))


def process_dir(args):
    dir_path = args.source_path
    filenames = os.listdir(dir_path)

    for filename in filenames:
        if is_video(filename):
            args.source_path = osp.join(dir_path, filename)
            process_file(args)


def main(args):
    if osp.exists(args.source_path):
        if osp.isdir(args.source_path):
            process_dir(args)

        else:
            process_file(args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
