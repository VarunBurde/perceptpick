"""I/O helpers — only the pieces the active pipeline needs.

Slimmed from burg-toolkit's ``io.py`` to two functions: ``load_mesh`` (used
by ``ObjectType.mesh`` for lazy loading) and ``save_urdf`` (used by
``ObjectType.generate_urdf``). Everything else (YAMLObject, depth-image
writers, third-party dataset readers) is gone.
"""
import os

import numpy as np
import open3d as o3d


def load_mesh(mesh_fn, texture_fn=None):
    """Load an Open3D TriangleMesh and pre-compute normals."""
    mesh = o3d.io.read_triangle_mesh(mesh_fn, enable_post_processing=True)
    if texture_fn is not None:
        mesh.textures = [o3d.io.read_image(texture_fn)]
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def save_urdf(
    fn,
    mesh_fn,
    name,
    origin=None,
    inertia=None,
    com=None,
    mass=0,
    friction=0.24,
    scale=1.0,
    overwrite_existing=True,
):
    """Write a single-link URDF referencing ``mesh_fn`` as both visual and
    collision geometry, with the given mass / inertia / friction / center of
    mass. Used by ``ObjectType.generate_urdf``.
    """
    if not overwrite_existing and os.path.exists(fn):
        return

    if origin is None:
        origin = [0, 0, 0]
    if inertia is None:
        inertia = np.eye(3) * 0.001
    if com is None:
        com = [0, 0, 0]

    origin_str = " ".join(map(str, origin))
    com_str = " ".join(map(str, com))

    with open(fn, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<robot name="{name}">\n')
        f.write('\t<link name="base">\n')
        # collision
        f.write('\t\t<collision>\n')
        f.write('\t\t\t<geometry>\n')
        f.write(f'\t\t\t\t<mesh filename="{mesh_fn}" scale="{scale} {scale} {scale}"/>\n')
        f.write('\t\t\t</geometry>\n')
        f.write(f'\t\t\t<origin xyz="{origin_str}"/>\n')
        f.write(f'\t\t\t<contact_coefficients mu="{friction}" />\n')
        f.write('\t\t</collision>\n')
        # visual
        f.write('\t\t<visual>\n')
        f.write('\t\t\t<geometry>\n')
        f.write(f'\t\t\t\t<mesh filename="{mesh_fn}" scale="{scale} {scale} {scale}"/>\n')
        f.write('\t\t\t</geometry>\n')
        f.write(f'\t\t\t<origin xyz="{origin_str}"/>\n')
        f.write('\t\t</visual>\n')
        # inertial
        f.write('\t\t<inertial>\n')
        f.write(f'\t\t\t<mass value="{mass}"/>\n')
        f.write(
            f'\t\t\t<inertia ixx="{inertia[0, 0]}" ixy="{inertia[0, 1]}" ixz="{inertia[0, 2]}"'
            f' iyy="{inertia[1, 1]}" iyz="{inertia[1, 2]}" izz="{inertia[2, 2]}" />\n'
        )
        f.write(f'\t\t\t<origin xyz="{com_str}"/>\n')
        f.write('\t\t</inertial>\n')

        f.write('\t</link>\n')
        f.write('</robot>')
