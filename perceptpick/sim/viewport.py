"""Free helpers for cleaning up the PyBullet GUI viewport.

Used by both ``GraspSimulatorWithGravityControl`` (Stage B) and the
``evaluate_pick`` simulator path (Stage C) so visualisation looks consistent.
"""
from __future__ import annotations


_FLAGS = (
    "COV_ENABLE_GUI",
    "COV_ENABLE_RGB_BUFFER_PREVIEW",
    "COV_ENABLE_DEPTH_BUFFER_PREVIEW",
    "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
)


def configure_clean_viewport(p_client) -> None:
    """Hide PyBullet's side panels and preview thumbnails so only the 3D
    viewport is visible. Mouse pan/zoom still work."""
    for flag_name in _FLAGS:
        flag = getattr(p_client, flag_name, None)
        if flag is not None:
            p_client.configureDebugVisualizer(flag, 0)


def draw_world_frame(p_client, axis_len: float = 0.1, line_width: float = 2.0) -> None:
    """Draw red/green/blue X/Y/Z axes at the world origin via debug lines.
    Debug items don't survive ``resetSimulation``, so callers should redraw
    after each scene reset."""
    p_client.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0], lineWidth=line_width, lifeTime=0)
    p_client.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0], lineWidth=line_width, lifeTime=0)
    p_client.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1], lineWidth=line_width, lifeTime=0)
