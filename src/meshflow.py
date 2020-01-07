import cv2
import numpy as np
from scipy.signal import medfilt
from .utils import tic, toc

# block of size in mesh
PIXELS = 16

# motion propagation radius
RADIUS = 266


def point_transform(H, pt):
    """
    @param: H is homography matrix of dimension (3x3) 
    @param: pt is the (x, y) point to be transformed
    
    Return:
            returns a transformed point ptrans = H*pt.
    """
    a = H[0,0]*pt[0] + H[0,1]*pt[1] + H[0,2]
    b = H[1,0]*pt[0] + H[1,1]*pt[1] + H[1,2]
    c = H[2,0]*pt[0] + H[2,1]*pt[1] + H[2,2]
    return [a/c, b/c]


def motion_propagate_L2(old_points, new_points, old_frame):
    """
    @param: old_points are points in old_frame that are
            matched feature points with new_frame
    @param: new_points are points in new_frame that are
            matched feature points with old_frame
    @param: old_frame is the frame to which
            motion mesh needs to be obtained
    @param: H is the homography between old and new points

    Return:
            returns a motion mesh in x-direction
            and y-direction for old_frame
    """
    # spreads motion over the mesh for the old_frame
    x_motion = {};
    y_motion = {};
    cols, rows = int(old_frame.shape[1] / PIXELS), int(old_frame.shape[0] / PIXELS)

    # pre-warping with global homography
    H, _ = cv2.findHomography(old_points, new_points, cv2.RANSAC)
    for i in range(rows):
        for j in range(cols):
            pt = [PIXELS * j, PIXELS * i]
            ptrans = point_transform(H, pt)
            x_motion[i, j] = pt[0] - ptrans[0]
            y_motion[i, j] = pt[1] - ptrans[1]

    # distribute feature motion vectors
    temp_x_motion = {};
    temp_y_motion = {}
    for i in range(rows):
        for j in range(cols):
            vertex = [PIXELS * j, PIXELS * i]
            for pt, st in zip(old_points, new_points):

                # velocity = point - feature point match in next frame
                # dst = sqrt((vertex[0]-st[0])**2+(vertex[1]-st[1])**2)

                # velocity = point - feature point in current frame
                dst = np.sqrt((vertex[0] - pt[0]) ** 2 + (vertex[1] - pt[1]) ** 2)
                if dst < RADIUS:
                    ptrans = point_transform(H, pt)
                    try:
                        temp_x_motion[i, j].append(st[0] - ptrans[0])
                    except:
                        temp_x_motion[i, j] = [st[0] - ptrans[0]]
                    try:
                        temp_y_motion[i, j].append(st[1] - ptrans[1])
                    except:
                        temp_y_motion[i, j] = [st[1] - ptrans[1]]

    # apply median filter (f-1) on obtained motion for each vertex
    x_motion_mesh = np.zeros((rows, cols), dtype=float)
    y_motion_mesh = np.zeros((rows, cols), dtype=float)
    for key in x_motion.keys():
        try:
            temp_x_motion[key].sort()
            x_motion_mesh[key] = x_motion[key] + temp_x_motion[key][int(len(temp_x_motion[key]) / 2)]
        except KeyError:
            x_motion_mesh[key] = x_motion[key]
        try:
            temp_y_motion[key].sort()
            y_motion_mesh[key] = y_motion[key] + temp_y_motion[key][int(len(temp_y_motion[key]) / 2)]
        except KeyError:
            y_motion_mesh[key] = y_motion[key]

    # apply second median filter (f-2) over the motion mesh for outliers
    x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])
    y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])

    return x_motion_mesh, y_motion_mesh


def motion_propagate_L1(old_points, new_points, old_frame):
    """
    @param: old_points are points in old_frame that are
            matched feature points with new_frame
    @param: new_points are points in new_frame that are
            matched feature points with old_frame
    @param: old_frame is the frame to which
            motion mesh needs to be obtained
    @param: H is the homography between old and new points

    Return:
            returns a motion mesh in x-direction
            and y-direction for old_frame
    """
    # spreads motion over the mesh for the old_frame
    x_motion = {};
    y_motion = {};
    cols, rows = int(old_frame.shape[1] / PIXELS), int(old_frame.shape[0] / PIXELS)

    # pre-warping with global homography
    H, _ = cv2.findHomography(old_points, new_points, cv2.RANSAC)
    for i in range(rows):
        for j in range(cols):
            pt = [PIXELS * j, PIXELS * i]
            ptrans = point_transform(H, pt)
            x_motion[i, j] = pt[0] - ptrans[0]
            y_motion[i, j] = pt[1] - ptrans[1]

    # distribute feature motion vectors
    temp_x_motion = {};
    temp_y_motion = {}
    for i in range(rows):
        for j in range(cols):
            vertex = [PIXELS * j, PIXELS * i]
            for pt, st in zip(old_points, new_points):

                # velocity = point - feature point match in next frame
                # dst = sqrt((vertex[0]-st[0])**2+(vertex[1]-st[1])**2)

                # velocity = point - feature point in current frame
                # dst = np.sqrt((vertex[0] - pt[0]) ** 2 + (vertex[1] - pt[1]) ** 2)
                if np.abs(vertex[0] - pt[0]) < RADIUS and np.abs(vertex[1] - pt[1])  < RADIUS:
                    ptrans = point_transform(H, pt)
                    try:
                        temp_x_motion[i, j].append(st[0] - ptrans[0])
                    except:
                        temp_x_motion[i, j] = [st[0] - ptrans[0]]
                    try:
                        temp_y_motion[i, j].append(st[1] - ptrans[1])
                    except:
                        temp_y_motion[i, j] = [st[1] - ptrans[1]]

    # apply median filter (f-1) on obtained motion for each vertex
    x_motion_mesh = np.zeros((rows, cols), dtype=float)
    y_motion_mesh = np.zeros((rows, cols), dtype=float)
    for key in x_motion.keys():
        try:
            temp_x_motion[key].sort()
            x_motion_mesh[key] = x_motion[key] + temp_x_motion[key][int(len(temp_x_motion[key]) / 2)]
        except KeyError:
            x_motion_mesh[key] = x_motion[key]
        try:
            temp_y_motion[key].sort()
            y_motion_mesh[key] = y_motion[key] + temp_y_motion[key][int(len(temp_y_motion[key]) / 2)]
        except KeyError:
            y_motion_mesh[key] = y_motion[key]

    # apply second median filter (f-2) over the motion mesh for outliers
    x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])
    y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])

    return x_motion_mesh, y_motion_mesh


def motion_propagate_fast(old_points, new_points, old_frame):
    """
    @param: old_points are points in old_frame that are
            matched feature points with new_frame
    @param: new_points are points in new_frame that are
            matched feature points with old_frame
    @param: old_frame is the frame to which
            motion mesh needs to be obtained
    @param: H is the homography between old and new points

    Return:
            returns a motion mesh in x-direction
            and y-direction for old_frame
    """
    # spreads motion over the mesh for the old_frame
    x_motion, y_motion = {}, {}
    cols, rows = int(old_frame.shape[1] / PIXELS), int(old_frame.shape[0] / PIXELS)

    # pre-warping with global homography
    H, _ = cv2.findHomography(old_points, new_points, cv2.RANSAC)
    for i in range(rows):
        for j in range(cols):
            pt = [PIXELS * j, PIXELS * i]
            ptrans = point_transform(H, pt)
            x_motion[i, j] = pt[0] - ptrans[0]
            y_motion[i, j] = pt[1] - ptrans[1]

    # distribute feature motion vectors
    temp_x_motion, temp_y_motion = {}, {}

    # # transform old points according to homography
    # pt_T = np.concatenate((old_points, np.ones((len(old_points), 1))), axis=1).T
    # pt_x, pt_y = old_points[:, 0], old_points[:, 1]
    # pt_xt, pt_yt, pt_wt = np.dot(H, pt_T)
    # pt_xt /= pt_wt
    # pt_yt /= pt_wt
    #
    # # calculate delta in advance
    # Dx = (new_points[:, 0] - pt_xt.T).flatten()
    # Dy = (new_points[:, 1] - pt_yt.T).flatten()
    #
    # for i, (x, y) in enumerate(zip(pt_x, pt_y)):
    #     # ptrans = point_transform(H, pt)
    #     si = max(np.round((y - RADIUS) / PIXELS).astype(int), 0)
    #     ei = min(np.round((y + RADIUS) / PIXELS).astype(int), rows)
    #     sj = max(np.round((x - RADIUS) / PIXELS).astype(int), 0)
    #     ej = min(np.round((x + RADIUS) / PIXELS).astype(int), cols)
    #
    #     # dx, dy = st[0] - ptrans[0], st[1] - ptrans[1]
    #     for i in range(si, ei):
    #         for j in range(sj, ej):
    #             try:
    #                 temp_x_motion[i, j].append(Dx[i])
    #             except:
    #                 temp_x_motion[i, j] = [Dx[i]]
    #             try:
    #                 temp_y_motion[i, j].append(Dy[i])
    #             except:
    #                 temp_y_motion[i, j] = [Dy[i]]

    for pt, st in zip(old_points, new_points):
        ptrans = point_transform(H, pt)

        si = max(np.floor((pt[1] - RADIUS) / PIXELS).astype(int), 0)
        ei = min(np.ceil((pt[1] + RADIUS) / PIXELS).astype(int), rows)
        sj = max(np.floor((pt[0] - RADIUS) / PIXELS).astype(int), 0)
        ej = min(np.ceil((pt[0] + RADIUS) / PIXELS).astype(int), cols)

        dx, dy = st[0] - ptrans[0], st[1] - ptrans[1]
        for i in range(si, ei):
            for j in range(sj, ej):
                try:
                    temp_x_motion[i, j].append(dx)
                except:
                    temp_x_motion[i, j] = [dx]
                try:
                    temp_y_motion[i, j].append(dy)
                except:
                    temp_y_motion[i, j] = [dy]

    # apply median filter (f-1) on obtained motion for each vertex
    x_motion_mesh = np.zeros((rows, cols), dtype=float)
    y_motion_mesh = np.zeros((rows, cols), dtype=float)
    for key in x_motion.keys():
        try:
            temp_x_motion[key].sort()
            x_motion_mesh[key] = x_motion[key] + temp_x_motion[key][int(len(temp_x_motion[key]) / 2)]
        except KeyError:
            x_motion_mesh[key] = x_motion[key]
        try:
            temp_y_motion[key].sort()
            y_motion_mesh[key] = y_motion[key] + temp_y_motion[key][int(len(temp_y_motion[key]) / 2)]
        except KeyError:
            y_motion_mesh[key] = y_motion[key]

    # apply second median filter (f-2) over the motion mesh for outliers
    x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])
    y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])

    return x_motion_mesh, y_motion_mesh


def generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh):
    """
    @param: x_paths is vertex profiles along x-direction
    @param: y_paths is vertex profiles along y_direction
    @param: x_motion_mesh is obtained motion mesh along 
            x-direction from motion_propagate()
    @param: y_motion_mesh is obtained motion mesh along 
            y-direction from motion_propagate()

    Returns:
            returns updated x_paths, y_paths with new 
            x_motion_mesh, y_motion_mesh added to the 
            last x_paths, y_paths
    """
    new_x_path = x_paths[:, :, -1] + x_motion_mesh
    new_y_path = y_paths[:, :, -1] + y_motion_mesh
    x_paths = np.concatenate((x_paths, np.expand_dims(new_x_path, axis=2)), axis=2)
    y_paths = np.concatenate((y_paths, np.expand_dims(new_y_path, axis=2)), axis=2)
    return x_paths, y_paths


def mesh_warp_frame_slow(frame, x_motion_mesh, y_motion_mesh):
    """
    @param: frame is the current frame
    @param: x_motion_mesh is the motion_mesh to
            be warped on frame along x-direction
    @param: y_motion_mesh is the motion mesh to
            be warped on frame along y-direction

    Returns:
            returns a mesh warped frame according
            to given motion meshes x_motion_mesh,
            y_motion_mesh
    """

    # define handles on mesh in x-direction
    map_x = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    # define handles on mesh in y-direction
    map_y = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    for i in range(x_motion_mesh.shape[0] - 1):
        for j in range(x_motion_mesh.shape[1] - 1):

            src = [[j * PIXELS, i * PIXELS],
                   [j * PIXELS, (i + 1) * PIXELS],
                   [(j + 1) * PIXELS, i * PIXELS],
                   [(j + 1) * PIXELS, (i + 1) * PIXELS]]
            src = np.asarray(src)

            dst = [[j * PIXELS + x_motion_mesh[i, j], i * PIXELS + y_motion_mesh[i, j]],
                   [j * PIXELS + x_motion_mesh[i + 1, j], (i + 1) * PIXELS + y_motion_mesh[i + 1, j]],
                   [(j + 1) * PIXELS + x_motion_mesh[i, j + 1], i * PIXELS + y_motion_mesh[i, j + 1]],
                   [(j + 1) * PIXELS + x_motion_mesh[i + 1, j + 1], (i + 1) * PIXELS + y_motion_mesh[i + 1, j + 1]]]
            dst = np.asarray(dst)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)

            for k in range(PIXELS * i, PIXELS * (i + 1)):
                for l in range(PIXELS * j, PIXELS * (j + 1)):
                    x = H[0, 0] * l + H[0, 1] * k + H[0, 2]
                    y = H[1, 0] * l + H[1, 1] * k + H[1, 2]
                    w = H[2, 0] * l + H[2, 1] * k + H[2, 2]
                    if not w == 0:
                        x = x / (w * 1.0);
                        y = y / (w * 1.0)
                    else:
                        x = l;
                        y = k
                    map_x[k, l] = x
                    map_y[k, l] = y

    # repeat motion vectors for remaining frame in y-direction
    for i in range(PIXELS * x_motion_mesh.shape[0], map_x.shape[0]):
        map_x[i, :] = map_x[PIXELS * x_motion_mesh.shape[0] - 1, :]
        map_y[i, :] = map_y[PIXELS * x_motion_mesh.shape[0] - 1, :]

    # repeat motion vectors for remaining frame in x-direction
    for j in range(PIXELS * x_motion_mesh.shape[1], map_x.shape[1]):
        map_x[:, j] = map_x[:, PIXELS * x_motion_mesh.shape[0] - 1]
        map_y[:, j] = map_y[:, PIXELS * x_motion_mesh.shape[0] - 1]

    # deforms mesh
    new_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return new_frame


def mesh_warp_frame_fast(frame, x_motion_mesh, y_motion_mesh):
    """
    @param: frame is the current frame
    @param: x_motion_mesh is the motion_mesh to
            be warped on frame along x-direction
    @param: y_motion_mesh is the motion mesh to
            be warped on frame along y-direction

    Returns:
            returns a mesh warped frame according
            to given motion meshes x_motion_mesh,
            y_motion_mesh
    """

    # define handles on mesh in x-direction
    map_x = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    # define handles on mesh in y-direction
    map_y = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    # prepare mesh grid (avoid redundant computation)
    K = np.arange(max(x_motion_mesh.shape) * PIXELS)  # rows, y
    L = np.copy(K)  # cols, x
    Kv, Lv = np.meshgrid(K, L, indexing='xy')

    # pre-compute src: mesh grid
    I = np.arange(max(x_motion_mesh.shape)) * PIXELS
    J = np.copy(I)
    Iv, Jv = np.meshgrid(I, J, indexing='xy')
    S = np.concatenate((Iv[:, :, np.newaxis], Jv[:, :, np.newaxis]), axis=2)

    # pre-compute dst: merge x_motion_mesh and y_motion_mesh
    motion_mesh = np.concatenate((x_motion_mesh[:, :, np.newaxis], y_motion_mesh[:, :, np.newaxis]), axis=2)
    D = S[0 : motion_mesh.shape[0], 0 : motion_mesh.shape[1]] + motion_mesh

    # debug
    vertex_t, homo_t, meshgrid_t, transform_t = 0, 0, 0, 0

    for i in range(x_motion_mesh.shape[0] - 1):  # y
        for j in range(x_motion_mesh.shape[1] - 1):  # x

            tic()
            # src = np.array([
            #     [j    , i    ],
            #     [j    , i + 1],
            #     [j + 1, i    ],
            #     [j + 1, i + 1]
            # ]) * PIXELS
            src = np.array([
                S[i    , j    ],
                S[i + 1, j    ],
                S[i    , j + 1],
                S[i + 1, j + 1]
            ])

            # dst = src + np.array([
            #     [x_motion_mesh[i    , j    ], y_motion_mesh[i    , j    ]],
            #     [x_motion_mesh[i + 1, j    ], y_motion_mesh[i + 1, j    ]],
            #     [x_motion_mesh[i    , j + 1], y_motion_mesh[i    , j + 1]],
            #     [x_motion_mesh[i + 1, j + 1], y_motion_mesh[i + 1, j + 1]]
            # ])
            dst = np.array([
                D[i    , j    ],
                D[i + 1, j    ],
                D[i    , j + 1],
                D[i + 1, j + 1]
            ])
            vertex_t += toc()

            tic()
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
            homo_t += toc()

            sk, ek = src[2, 1], src[1, 1]
            sl, el = src[1, 0], src[2, 0]

            tic()
            # K = np.arange(sk, ek)  # y (rows)
            # L = np.arange(sl, el)  # x (cols)
            # kv, lv = np.meshgrid(K, L, indexing='xy')
            kv = Kv[sl : el, sk : ek]
            lv = Lv[sl : el, sk : ek]
            meshgrid_t += toc()

            tic()
            lv = lv.reshape(1, -1)
            kv = kv.reshape(1, -1)
            ones = np.ones(lv.shape)
            coords = np.concatenate([lv, kv, ones], axis=0)

            x, y, w = np.dot(H, coords)
            x /= w
            y /= w

            x[w == 0] = lv.reshape(-1)[w == 0]
            y[w == 0] = kv.reshape(-1)[w == 0]

            map_x[sk : ek, sl : el] = x.reshape((PIXELS, PIXELS)).T
            map_y[sk : ek, sl : el] = y.reshape((PIXELS, PIXELS)).T
            transform_t += toc()

    # print('\n\tvertex %5.2f, homo %5.2f, meshgrid %5.2f, transform %5.2f (ms)' % (vertex_t, homo_t, meshgrid_t, transform_t))

    # repeat motion vectors for remaining frame in y-direction
    for i in range(PIXELS * x_motion_mesh.shape[0], map_x.shape[0]):
        map_x[i, :] = map_x[PIXELS * x_motion_mesh.shape[0] - 1, :]
        map_y[i, :] = map_y[PIXELS * x_motion_mesh.shape[0] - 1, :]

    # repeat motion vectors for remaining frame in x-direction
    for j in range(PIXELS * x_motion_mesh.shape[1], map_x.shape[1]):
        map_x[:, j] = map_x[:, PIXELS * x_motion_mesh.shape[0] - 1]
        map_y[:, j] = map_y[:, PIXELS * x_motion_mesh.shape[0] - 1]

    # deforms mesh
    new_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return new_frame