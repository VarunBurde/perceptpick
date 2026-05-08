import copy
import logging

import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from ..core import _types as core
from ..core import grasp
from ..utils import transforms as util
from ..utils import mesh_helpers as mesh_processing


_log = logging.getLogger(__name__)


def rays_within_cone(axis, angle, n=10, uniform_on_plane=False):
    """
    Samples `n` rays originating from a cone apex.
    Cone is specified by it's `axis` vector from apex towards ground area and by the `angle` between axis and sides.
    We sample uniformly on the angle, causing the rays to be distributed more towards the center of the ground plane
    of the cone. If this is not desired, use `uniform_on_plane` parameter.

    :param axis: (3,) vector pointing from cone apex in the direction the rays will be cast
    :param angle: this is the max angle between axis and cast rays in [rad]
    :param n: number of rays to cast (defaults to 10)
    :param uniform_on_plane: if set to True, the rays will show uniform distribution on the ground plane of the cone.

    :return: (n, 3) numpy array with ray direction vectors (normalised)
    """
    # sample spherical coordinates: inclination in [0, angle], azimuth in [0, 2*pi]
    azimuth = np.random.uniform(0, 2*np.pi, n)
    if uniform_on_plane:
        inclination = np.arctan(np.random.uniform(0, np.tan(angle), n))
    else:
        inclination = np.random.uniform(0, angle, n)

    # convert from spherical to cartesian coordinates (radius = 1, i.e. vectors are normalized)
    cartesian = np.empty((n, 3, 1))
    cartesian[:, 0, 0] = np.sin(inclination)*np.cos(azimuth)
    cartesian[:, 1, 0] = np.sin(inclination)*np.sin(azimuth)
    cartesian[:, 2, 0] = np.cos(inclination)

    # transform so that z-axis aligns cone axis
    rot_mat = util.rotation_to_align_vectors([0, 0, 1], axis)
    ray_directions = np.matmul(rot_mat, cartesian)

    return ray_directions[:, :, 0]


class AntipodalGraspSampler:
    """
    A sampler for  antipodal grasps. Sampler looks for two contact points that satisfy the antipodal constraints
    for a given friction coefficient mu.

    # todo: currently broken, it is still using gripper model which should not be the case
    """

    def __init__(self, mu=0.25, n_orientations=12, n_rays=100, max_targets_per_ref_point=1, only_grasp_from_above=True,
                 no_contact_below_z=0.02, min_grasp_width=0.001, verbose=False):
        self.mu = mu
        self.n_orientations = n_orientations
        self.n_rays = n_rays
        self.min_grasp_width = min_grasp_width
        self.max_targets_per_ref_point = max_targets_per_ref_point
        self.only_grasp_from_above = only_grasp_from_above
        self.no_contact_below_z = no_contact_below_z
        self.verbose = verbose

    @staticmethod
    def construct_halfspace_grasp_set(reference_point, target_points, n_orientations):
        """
        For all pairs of reference point and target point grasps will be constructed at the center point.
        A grasp can be seen as a frame, the x-axis will point towards the target point and the z-axis will point
        in the direction from which the gripper will be approaching.
        This method only samples grasps from the halfspace above, the object will not be grasped from below.
        (Similar to GPNet data set.)

        :param reference_point: (3,) np array
        :param target_points: (n, 3) np array
        :param n_orientations: int, number of different orientations to use

        :return: grasp.GraspSet
        """
        reference_point = np.reshape(reference_point, (1, 3))
        target_points = np.reshape(target_points, (-1, 3))

        # center points in the middle between ref and target, x-axis pointing towards target point
        d = target_points - reference_point
        center_points = reference_point + 1/2 * d
        distances = np.linalg.norm(d, axis=-1)
        x_axes = d / distances[:, np.newaxis]

        # get unique x_axis representation = must only point upwards in z
        mask = x_axes[:, 2] < 0
        x_axes[mask] *= -1

        # y_tangent is constructed orthogonal to world z and gripper x
        y_tangent = -np.cross(x_axes, [0, 0, 1])
        y_tangent = y_tangent / np.linalg.norm(y_tangent, axis=-1)[:, np.newaxis]

        # the z-axes of the grasps are this y_tangent vector, but rotated around grasp's x-axis by theta
        # let's get the axis-angle representation to construct the rotations
        theta = np.linspace(0, np.pi, num=n_orientations)
        # multiply each of the x_axes with each of the thetas
        # [n, 3] * [m] --> [n, m, 3] --> [n*m, 3]
        # (first we have 1st x-axis with all thetas, then 2nd x-axis with all thetas, etc..)
        axis_angles = np.einsum('ik,j->ijk', x_axes, theta).reshape(-1, 3)
        rotations = R.from_rotvec(axis_angles)
        poses = np.empty(shape=(len(rotations), 4, 4))

        for i in range(len(x_axes)):
            for j in range(n_orientations):
                # apply the rotation to the y_tangent to get grasp z
                index = i*n_orientations + j
                rot_mat = rotations[index].as_matrix()
                z_axis = rot_mat @ y_tangent[i]

                # finally get y
                y_axis = np.cross(z_axis, x_axes[i])
                y_axis = y_axis / np.linalg.norm(y_axis)

                poses[index] = util.tf_from_xyz_pos(x_axes[i], y_axis, z_axis, center_points[i])

        gs = grasp.GraspSet.from_poses(poses)
        gs.widths = np.tile(distances, n_orientations).reshape(n_orientations, len(distances)).T.reshape(-1)

        return gs

    @staticmethod
    def construct_grasp_set(reference_point, target_points, n_orientations):
        """
        For all pairs of reference point and target point grasps will be constructed at the center point.
        A grasp can be seen as a frame, the x-axis will point towards the target point and the z-axis will point
        in the direction from which the gripper will be approaching.

        :param reference_point: (3,) np array
        :param target_points: (n, 3) np array
        :param n_orientations: int, number of different orientations to use

        :return: grasp.GraspSet
        """
        reference_point = np.reshape(reference_point, (1, 3))
        target_points = np.reshape(target_points, (-1, 3))

        # center points in the middle between ref and target, x-axis pointing towards target point
        d = target_points - reference_point
        center_points = reference_point + 1/2 * d
        distances = np.linalg.norm(d, axis=-1)
        x_axis = d / distances[:, np.newaxis]

        # construct y-axis and z-axis orthogonal to x-axis
        y_axis = np.zeros(x_axis.shape)
        while (np.linalg.norm(y_axis, axis=-1) == 0).any():
            tmp_vec = util.generate_random_unit_vector()
            y_axis = np.cross(x_axis, tmp_vec)
            # todo: using the same random unit vec to construct all frames will lead to very similar orientations
            #       we might want to randomize this even more by using individual, random unit vectors
        y_axis = y_axis / np.linalg.norm(y_axis, axis=-1)[:, np.newaxis]
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis, axis=-1)[:, np.newaxis]

        # with all axes and the position, we can construct base frames
        tf_basis = util.tf_from_xyz_pos(x_axis, y_axis, z_axis, center_points).reshape(-1, 4, 4)

        # generate transforms for the n_orientations (rotation around x axis)
        theta = np.arange(0, 2*np.pi, 2*np.pi / n_orientations)
        tf_rot = np.tile(np.eye(4), (n_orientations, 1, 1))
        tf_rot[:, 1, 1] = np.cos(theta)
        tf_rot[:, 1, 2] = -np.sin(theta)
        tf_rot[:, 2, 1] = np.sin(theta)
        tf_rot[:, 2, 2] = np.cos(theta)

        # apply transforms
        tfs = np.matmul(tf_basis[np.newaxis, :, :, :], tf_rot[:, np.newaxis, :, :]).reshape(-1, 4, 4)
        gs = grasp.GraspSet.from_poses(tfs)

        # add distances as gripper widths (repeat n_orientation times)
        gs.widths = np.tile(distances, n_orientations).reshape(n_orientations, len(distances)).T.reshape(-1)

        return gs

    def sample(self, object_instance, n=10, max_gripper_width=0.08):
        """
        Samples n grasps for the given object_instance.

        :param object_instance: core.ObjectInstance
        :param n: integer, number of grasps to sample
        :param max_gripper_width: float, maximum opening width of gripper, ie grasps with contact points that are
                                  farther apart than this will not be considered

        :return: grasp.GraspSet
        """
        _log.debug('preparing to sample grasps...')

        # we make use of both open3d and trimesh, therefore keep both representations in memory...
        mesh = object_instance.get_mesh()
        _trimesh = mesh_processing.as_trimesh(mesh)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(_trimesh)

        # we first sample reference points (first contact point) from the mesh surface
        # therefore, we first sample many points at once and then just use some of these at random
        n_sample = np.max([n, 1000, len(mesh.triangles)])
        ref_points = util.o3d_pc_to_numpy(mesh_processing.poisson_disk_sampling(mesh, n_points=n_sample))
        np.random.shuffle(ref_points)

        if self.no_contact_below_z is not None:
            keep = ref_points[:, 2] > self.no_contact_below_z
            ref_points = ref_points[keep]

        _log.debug(f'sampled {len(ref_points)} first contact point candidates, beginning to find grasps.')

        # determine some parameters for casting rays in a friction cone
        angle = np.arctan(self.mu)
        _log.debug('mu is', self.mu, 'hence angle of friction cone is', np.rad2deg(angle), '°')

        gs = grasp.GraspSet()
        gs_contacts = np.empty((0, 2, 3))
        i_ref_point = 0

        with tqdm(total=n, disable=not self.verbose) as progress_bar:
            while len(gs) < n:
                if i_ref_point >= len(ref_points):
                    _log.info('ran out of ref_points, sampling more grasps would likely yield only more of the same')
                    break

                p_r = ref_points[i_ref_point, 0:3]
                n_r = ref_points[i_ref_point, 3:6]
                _log.debug(f'sampling ref point no {i_ref_point}: point {p_r}, normal {n_r}')
                i_ref_point = (i_ref_point + 1) % len(ref_points)

                # cast random rays from p_r within the friction cone to identify potential contact points
                ray_directions = rays_within_cone(-n_r, angle, self.n_rays)
                ray_origins = np.tile(p_r, (self.n_rays, 1))
                locations, _, index_tri = intersector.intersects_location(
                    ray_origins, ray_directions, multiple_hits=True)
                _log.debug(f'* casting {self.n_rays} rays, leading to {len(locations)} intersection locations')
                if len(locations) == 0:
                    continue

                # eliminate intersections with origin
                mask_is_not_origin = ~np.isclose(locations, p_r, atol=1e-11).all(axis=-1)
                locations = locations[mask_is_not_origin]
                index_tri = index_tri[mask_is_not_origin]
                _log.debug(f'* ... of which {len(locations)} are not with ref point')
                if len(locations) == 0:
                    continue

                # eliminate contact points too far or too close
                distances = np.linalg.norm(locations - p_r, axis=-1)
                mask_is_within_distance = \
                    (distances <= max_gripper_width)\
                    | (distances >= self.min_grasp_width)
                locations = locations[mask_is_within_distance]
                index_tri = index_tri[mask_is_within_distance]
                _log.debug(f'* ... of which {len(locations)} are within gripper width constraints')
                if len(locations) == 0:
                    continue

                normals = mesh_processing.compute_interpolated_vertex_normals(_trimesh, locations, index_tri)

                # compute angles to check antipodal constraints
                d = (locations - p_r).reshape(-1, 3)
                signs = np.zeros(len(d))
                angles = util.angle(d, normals, sign_array=signs, as_degree=False)

                # exclude target points which do not have opposing surface orientations
                # positive sign means vectors are facing into a similar direction as connecting vector, as expected
                mask_faces_correct_direction = signs > 0
                locations = locations[mask_faces_correct_direction]
                normals = normals[mask_faces_correct_direction]
                angles = angles[mask_faces_correct_direction]
                _log.debug(f'* ... of which {len(locations)} are generally facing in opposing directions')
                if len(locations) == 0:
                    continue

                # check friction cone constraint
                mask_friction_cone = angles <= angle
                locations = locations[mask_friction_cone]
                normals = normals[mask_friction_cone]
                angles = angles[mask_friction_cone]
                _log.debug(f'* ... of which {len(locations)} are satisfying the friction constraint')
                if len(locations) == 0:
                    continue

                # check below z contact
                if self.no_contact_below_z is not None:
                    mask_below_z = locations[:, 2] > self.no_contact_below_z
                    locations = locations[mask_below_z]
                    normals = normals[mask_below_z]
                    angles = angles[mask_below_z]
                    _log.debug(f'* ... of which {len(locations)} are above the specified z value')
                    if len(locations) == 0:
                        continue

                # make sure we do not use more than max_targets_per_ref_point
                if len(locations) > self.max_targets_per_ref_point:
                    indices = farthest_point_sampling(locations, self.max_targets_per_ref_point)
                    locations = locations[indices]
                    _log.debug(f'* ... of which we sample {len(locations)} to construct grasps')

                # now proceed to construct all the grasps (with n_orientations)
                if self.only_grasp_from_above:
                    grasps = self.construct_halfspace_grasp_set(p_r, locations, self.n_orientations)
                else:
                    grasps = self.construct_grasp_set(p_r, locations, self.n_orientations)

                # also compute the contact points
                contacts = np.empty((len(locations), 2, 3))
                contacts[:, 0] = p_r
                contacts[:, 1] = locations
                contacts = np.repeat(contacts, self.n_orientations, axis=0)

                gs_contacts = np.concatenate([gs_contacts, contacts], axis=0)
                gs.add(grasps)
                _log.debug(f'* added {len(grasps)} grasps (with {self.n_orientations} orientations for each point pair)')
                progress_bar.update(len(grasps))

        return gs, gs_contacts

    def check_collisions(self, graspset, scene, gripper_mesh, with_plane=False):
        """
        This will check collisions for the given graspset with a given gripper mesh.

        :param graspset: The n-elem grasp.GraspSet to check collisions for
        :param scene: core.Scene containing the objects in the scene
        :param gripper_mesh: open3d.geometry.TriangleMesh to check collisions against (gripper)
        :param with_plane: bool, whether to also check collisions with the ground plane

        :return: ndarray (n,) of dtype=bool
        """
        # we need collision operations which are not available in o3d, hence use trimesh
        manager = trimesh.collision.CollisionManager()

        # add scene objects
        meshes = scene.get_mesh_list(with_bg_objects=True, with_plane=with_plane)  # todo: get directly as trimesh
        for i, obj in enumerate(meshes):
            manager.add_object(f'add_obj_{i}', mesh_processing.as_trimesh(obj))

        # prepare gripper mesh
        gripper_mesh = copy.deepcopy(gripper_mesh)
        gripper_mesh = mesh_processing.as_trimesh(gripper_mesh)

        collision_array = np.empty(len(graspset), dtype=bool)

        _log.debug('checking collisions...')
        for i, g in tqdm(enumerate(graspset), disable=not self.verbose):
            collision_array[i] = manager.in_collision_single(gripper_mesh, transform=g.pose)

        return collision_array


def farthest_point_sampling(point_cloud, k):
    """Approximate farthest-point sampling: pick ``k`` points from the input
    that are mutually as far apart as possible. Used by ``AntipodalGraspSampler``
    to spread target contact points across the mesh surface.
    """
    if len(point_cloud) < k:
        raise ValueError(f"point cloud has {len(point_cloud)} elements, cannot sample {k}")

    point_cloud = point_cloud[:, :3]
    indices = np.zeros(k, dtype=int)
    distances = np.full(len(point_cloud), fill_value=np.inf)

    for i in range(1, k):
        d = ((point_cloud[indices[i - 1]] - point_cloud) ** 2).sum(axis=1)
        distances = np.minimum(distances, d)
        indices[i] = np.argmax(distances)
    return indices
