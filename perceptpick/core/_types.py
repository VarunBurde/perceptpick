"""Core data structures: ObjectType, ObjectInstance, Scene.

Slimmed from the burg-toolkit ``core.py`` to only the surface the BOG
benchmark actually uses. ObjectLibrary, StablePoses, YAML serialization,
thumbnail rendering, ground-area / out-of-bounds checks, and the
auto-generate-VHACD path are all gone — none of them were called by any
of the four pipeline scripts.
"""
import logging
import os

import numpy as np

from ..utils import io
from ..utils import mesh_helpers as mesh_processing


_log = logging.getLogger(__name__)


class ObjectType:
    """A graspable object: identifier + paths to mesh / URDF / VHACD + physics.

    Mesh is loaded lazily on first read of ``self.mesh``. ``__getstate__``
    drops the cached mesh before pickling so multiprocessing workers
    re-load it (Open3D's TriangleMesh is not pickle-safe — see
    https://github.com/isl-org/Open3D/issues/218).
    """

    def __init__(
        self,
        identifier,
        name=None,
        mesh=None,
        mesh_fn=None,
        vhacd_fn=None,
        urdf_fn=None,
        mass=None,
        friction_coeff=None,
    ):
        if mesh is not None and mesh_fn is not None:
            raise ValueError("Cannot create ObjectType with both mesh and mesh_fn")
        if mesh is None and mesh_fn is None:
            raise ValueError("Cannot create ObjectType with no mesh — provide mesh or mesh_fn")
        self.identifier = identifier
        self.name = name or identifier
        self._mesh = mesh
        self.mesh_fn = mesh_fn
        self.vhacd_fn = vhacd_fn
        self.urdf_fn = urdf_fn
        self.mass = mass or 0
        self.friction_coeff = friction_coeff or 0.24

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = io.load_mesh(mesh_fn=self.mesh_fn)
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_mesh"] = None
        return state

    def generate_urdf(self, urdf_fn, use_vhacd=True):
        """Write a URDF for this object. Requires ``mesh`` to compute inertia
        and (if ``use_vhacd``) ``vhacd_fn`` to point at a pre-generated VHACD.
        """
        inertia, com = mesh_processing.compute_mesh_inertia(self.mesh, self.mass)
        if use_vhacd:
            if self.vhacd_fn is None:
                raise ValueError(
                    f"ObjectType {self.identifier} has no vhacd_fn — "
                    f"generate VHACD first (see perceptpick.assets.mesh_prep)."
                )
            rel_mesh_fn = os.path.relpath(self.vhacd_fn, os.path.dirname(urdf_fn))
        else:
            if self.mesh_fn is None:
                raise ValueError("ObjectType has no mesh_fn for non-VHACD URDF")
            rel_mesh_fn = os.path.relpath(self.mesh_fn, os.path.dirname(urdf_fn))

        io.save_urdf(
            urdf_fn,
            mesh_fn=rel_mesh_fn,
            name=self.identifier,
            origin=[0, 0, 0],
            inertia=inertia,
            com=com,
            mass=self.mass,
            friction=self.friction_coeff,
            scale=1.0,
            overwrite_existing=True,
        )

    def __str__(self):
        return (
            f"ObjectType({self.identifier}, mass={self.mass}, "
            f"mesh_fn={self.mesh_fn}, vhacd_fn={self.vhacd_fn}, urdf_fn={self.urdf_fn})"
        )


class ObjectInstance:
    """An ObjectType placed in the world at a 4×4 pose."""

    def __init__(self, object_type, pose=None):
        self.object_type = object_type
        self.pose = np.eye(4) if pose is None else np.asarray(pose)

    def get_mesh(self):
        """Return a copy of the underlying mesh transformed into world space.
        Used by the antipodal sampler when sampling reference points on the
        object surface."""
        import copy
        mesh = copy.deepcopy(self.object_type.mesh)
        mesh.transform(self.pose)
        return mesh

    def __str__(self):
        return f"ObjectInstance({self.object_type.identifier})"


class Scene:
    """A scene is a list of ObjectInstances (foreground) plus optional
    background instances (visual reference, no physics)."""

    def __init__(self, objects=None, bg_objects=None):
        self.objects = objects if objects is not None else []
        self.bg_objects = bg_objects if bg_objects is not None else []

    def __str__(self):
        return (
            f"Scene(objects={[str(o) for o in self.objects]}, "
            f"bg_objects={[str(o) for o in self.bg_objects]})"
        )
