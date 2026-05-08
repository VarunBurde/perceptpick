"""Mesh utilities used by the antipodal sampler and the URDF generator.

Slimmed from burg-toolkit's ``mesh_processing.py`` to only what the active
pipeline needs: ``compute_mesh_inertia`` (for URDF), ``as_trimesh``,
``poisson_disk_sampling``, ``compute_interpolated_vertex_normals`` (all for
the antipodal sampler).
"""
import logging

import numpy as np
import open3d as o3d
import trimesh

_log = logging.getLogger(__name__)


def as_trimesh(mesh):
    """Coerce an Open3D TriangleMesh or a trimesh.Trimesh into trimesh."""
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        vertex_normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
        triangle_normals = np.asarray(mesh.triangle_normals) if mesh.has_triangle_normals() else None
        return trimesh.Trimesh(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles),
            vertex_normals=vertex_normals, triangle_normals=triangle_normals,
        )
    raise TypeError(f"Given mesh must be trimesh or o3d mesh. Got {type(mesh)}.")


def poisson_disk_sampling(mesh, radius=0.003, n_points=None, with_normals=True, init_factor=5):
    """Approximate Poisson-disk sampling on a TriangleMesh surface."""
    if n_points is None:
        s = np.sqrt(mesh.get_surface_area())
        n_s = (s + 2 * radius) / (2 * radius)
        n_points = int(n_s ** 2)
    return mesh.sample_points_poisson_disk(
        number_of_points=n_points, init_factor=init_factor, use_triangle_normal=with_normals
    )


def compute_interpolated_vertex_normals(mesh, points, triangle_indices=None):
    """For each point on the mesh surface, return a barycentric-weighted
    interpolated vertex normal. Used by the antipodal sampler to evaluate
    contact-cone alignment."""
    mesh = as_trimesh(mesh)
    if triangle_indices is None:
        closest_points, _, triangle_indices = trimesh.proximity.closest_point(mesh, points)
    else:
        closest_points = points

    faces = mesh.faces[triangle_indices]
    vertices = mesh.vertices[faces]
    vertex_normals = mesh.vertex_normals[faces]

    dist = np.linalg.norm(vertices - closest_points[:, None, :], axis=-1)
    weights = 1 / (dist ** (1 / 2))
    # When a query point lies exactly on a vertex, distance is 0 → use that
    # vertex's normal directly to avoid division-by-zero.
    nan_indices = np.nonzero(dist == 0)
    for i in range(len(nan_indices[0])):
        point_idx, dim_idx = nan_indices[0][i], nan_indices[1][i]
        w = np.zeros(3)
        w[dim_idx] = 1.0
        weights[point_idx] = w

    normals = np.average(weights[:, :, None] * vertex_normals, axis=1)
    normals = normals / np.linalg.norm(normals, axis=-1)[:, None]
    assert not np.isnan(normals).any(), "interpolated normal computation produced NaNs"
    return normals


def compute_mesh_inertia(mesh, mass):
    """Inertia tensor + center of mass for a (closed) mesh of given mass.

    Used by ``ObjectType.generate_urdf``. Warns if the mesh isn't watertight
    — the result will still be returned but may be less accurate.
    """
    mesh = as_trimesh(mesh)
    if not mesh.is_watertight:
        _log.warning("Computing inertia and COM despite mesh not being watertight.")
    mesh.density = mass / mesh.volume
    return mesh.moment_inertia, mesh.center_mass
