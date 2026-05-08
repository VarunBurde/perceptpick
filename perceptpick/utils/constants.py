"""Package-level constants used by the simulator and core data structures."""
import os

# Default ground-area extent for ``Scene`` (x, y) in meters. ISO A3 sheet —
# inherited from burg-toolkit's convention.
SIZE_A3 = (0.420, 0.297)

# Folder holding bundled URDF assets next to this file (currently just the
# invisible ``dummy_xyz_robot.urdf`` mount used by ``sim/robots.py``).
ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets/")
