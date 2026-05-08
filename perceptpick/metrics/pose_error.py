# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Implementation of the pose error functions described in:
Hodan, Michel et al., "BOP: Benchmark for 6D Object Pose Estimation", ECCV'18
Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW'16
"""

import math
import numpy as np
from scipy import spatial



# Minimal misc functions needed for mssd and mspd
def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def project_pts(pts, K, R, t):
    """Projects 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx2 ndarray with 2D image coordinates of the projections.
    """
    assert pts.shape[1] == 3
    P = K.dot(np.hstack((R, t.reshape(-1, 1))))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T


def iou(bb_a, bb_b):
    """Calculates the Intersection over Union (IoU) of two 2D bounding boxes.

    :param bb_a: 2D bounding box (x1, y1, w1, h1) -- see calc_2d_bbox.
    :param bb_b: 2D bounding box (x2, y2, w2, h2) -- see calc_2d_bbox.
    :return: The IoU value.
    """
    # [x1, y1, width, height] --> [x1, y1, x2, y2]
    tl_a, br_a = (bb_a[0], bb_a[1]), (bb_a[0] + bb_a[2], bb_a[1] + bb_a[3])
    tl_b, br_b = (bb_b[0], bb_b[1]), (bb_b[0] + bb_b[2], bb_b[1] + bb_b[3])

    # Intersection rectangle.
    tl_inter = max(tl_a[0], tl_b[0]), max(tl_a[1], tl_b[1])
    br_inter = min(br_a[0], br_b[0]), min(br_a[1], br_b[1])

    # Width and height of the intersection rectangle.
    w_inter = br_inter[0] - tl_inter[0]
    h_inter = br_inter[1] - tl_inter[1]

    if w_inter > 0 and h_inter > 0:
        area_inter = w_inter * h_inter
        area_a = bb_a[2] * bb_a[3]
        area_b = bb_b[2] * bb_b[3]
        iou = area_inter / float(area_a + area_b - area_inter)
    else:
        iou = 0.0

    return iou


def calc_2d_bbox(xs, ys, im_size=None, clip=False):
    """Calculates 2D bounding box of the given set of 2D points.

    :param xs: 1D ndarray with x-coordinates of 2D points.
    :param ys: 1D ndarray with y-coordinates of 2D points.
    :param im_size: Image size (width, height) (used for optional clipping).
    :param clip: Whether to clip the bounding box (default == False).
    :return: 2D bounding box (x, y, w, h), where (x, y) is the top-left corner
      and (w, h) is width and height of the bounding box.
    """
    bb_min = [xs.min(), ys.min()]
    bb_max = [xs.max(), ys.max()]
    if clip:
        assert im_size is not None
        bb_min = clip_pt_to_im(bb_min, im_size)
        bb_max = clip_pt_to_im(bb_max, im_size)
    return [bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]]


def clip_pt_to_im(pt, im_size):
    """Clips a 2D point to the image frame.

    :param pt: 2D point (x, y).
    :param im_size: Image size (width, height).
    :return: Clipped 2D point (x, y).
    """
    return [min(max(pt[0], 0), im_size[0] - 1), min(max(pt[1], 0), im_size[1] - 1)]


# Standard Library
from typing import Optional
from dataclasses import dataclass


@dataclass
class POSE_ERROR_VSD_ARGS:
    # all args in pose_error.vsd
    R_e: Optional[np.ndarray] = None
    t_e: Optional[np.ndarray] = None
    R_g: Optional[np.ndarray] = None
    t_g: Optional[np.ndarray] = None
    depth_im: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    vsd_deltas: Optional[float] = None
    vsd_taus: Optional[list] = None
    vsd_normalized_by_diameter: Optional[bool] = None
    diameter: Optional[float] = None
    obj_id: Optional[int] = None
    step: Optional[str] = None

    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)
        return self

    def to_file(self, path):
        for key, value in self.__dict__.items():
            if value is None:
                raise ValueError("Field {} is None".format(key))
            value = np.array(value)
        np.savez(path, self.__dict__)

    def from_file(path):
        args = POSE_ERROR_VSD_ARGS()
        data = np.load(path, allow_pickle=True)
        data = data["arr_0"].item()
        for key, value in data.items():
            if key == "vsd_taus":
                setattr(args, key, list(value))
            if key == "vsd_normalized_by_diameter":
                setattr(args, key, bool(value))
            if key == "step":
                setattr(args, key, str(value))
            if key == "obj_id":
                setattr(args, key, int(value))
            else:
                setattr(args, key, value)
        return args



def mssd(R_est, t_est, R_gt, t_gt, pts, syms):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_sym = transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(pts_est - pts_gt_sym, axis=1).max())
    return min(es)


def mspd(R_est, t_est, R_gt, t_gt, K, pts, syms):
    """Maximum Symmetry-Aware Projection Distance (MSPD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    proj_est = project_pts(pts, K, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = project_pts(pts, K, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(proj_est - proj_gt_sym, axis=1).max())
    return min(es)


def add(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable
    views - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


def te(t_est, t_gt):
    """Translational Error.

    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    assert t_est.size == t_gt.size == 3
    error = np.linalg.norm(t_gt - t_est)
    return error


def proj(R_est, t_est, R_gt, t_gt, K, pts):
    """Average distance of projections of object model vertices [px]
    - by Brachmann et al. (CVPR'16).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    proj_est = project_pts(pts, K, R_est, t_est)
    proj_gt = project_pts(pts, K, R_gt, t_gt)
    e = np.linalg.norm(proj_est - proj_gt, axis=1).mean()
    return e


def cou_mask(mask_est, mask_gt):
    """Complement over Union of 2D binary masks.

    :param mask_est: hxw ndarray with the estimated mask.
    :param mask_gt: hxw ndarray with the ground-truth mask.
    :return: The calculated error.
    """
    mask_est_bool = mask_est.astype(np.bool)
    mask_gt_bool = mask_gt.astype(np.bool)

    inter = np.logical_and(mask_gt_bool, mask_est_bool)
    union = np.logical_or(mask_gt_bool, mask_est_bool)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


def cus(R_est, t_est, R_gt, t_gt, K, renderer, obj_id):
    """Complement over Union of projected 2D masks.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :return: The calculated error.
    """
    # Render depth images of the model at the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
    depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]

    # Masks of the rendered model and their intersection and union.
    mask_est = depth_est > 0
    mask_gt = depth_gt > 0
    inter = np.logical_and(mask_gt, mask_est)
    union = np.logical_or(mask_gt, mask_est)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


def cou_bb(bb_est, bb_gt):
    """Complement over Union of 2D bounding boxes.

    :param bb_est: The estimated bounding box (x1, y1, w1, h1).
    :param bb_gt: The ground-truth bounding box (x2, y2, w2, h2).
    :return: The calculated error.
    """
    e = 1.0 - iou(bb_est, bb_gt)
    return e


def cou_bb_proj(R_est, t_est, R_gt, t_gt, K, renderer, obj_id):
    """Complement over Union of projected 2D bounding boxes.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :return: The calculated error.
    """
    # Render depth images of the model at the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
    depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]

    # Masks of the rendered model and their intersection and union
    mask_est = depth_est > 0
    mask_gt = depth_gt > 0

    ys_est, xs_est = mask_est.nonzero()
    bb_est = calc_2d_bbox(xs_est, ys_est, im_size=None, clip=False)

    ys_gt, xs_gt = mask_gt.nonzero()
    bb_gt = calc_2d_bbox(xs_gt, ys_gt, im_size=None, clip=False)

    e = 1.0 - iou(bb_est, bb_gt)
    return e
