"""Each registered gripper class instantiates without crashing.

We don't actually load them in PyBullet here (would need a connected client);
we just check the classes are importable and have the expected interface.
"""
from perceptpick.grippers import GRIPPER_REGISTRY, GripperBase


def test_each_gripper_subclasses_base():
    for name, cls in GRIPPER_REGISTRY.items():
        assert issubclass(cls, GripperBase), f"{name} must subclass GripperBase"


def test_registry_is_complete():
    expected = {
        "franka", "robotiq_2f_85", "robotiq_2f_140", "wsg_50", "wsg_32",
        "ezgripper", "sawyer", "robotiq_3f", "kinova_3f",
        "barrett", "barrett_2f", "rg2",
    }
    assert expected.issubset(set(GRIPPER_REGISTRY.keys()))
