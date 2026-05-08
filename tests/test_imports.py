"""Lightweight smoke test — every public module imports cleanly."""


def test_top_level():
    import perceptpick
    assert perceptpick.__version__


def test_core():
    from perceptpick.core import (
        Grasp, GraspSet, ObjectInstance, ObjectType, Scene,
    )
    import numpy as np
    ot = ObjectType(identifier="t", mesh_fn="/tmp/x.obj")
    inst = ObjectInstance(ot, pose=np.eye(4))
    s = Scene()
    s.objects.append(inst)
    assert len(s.objects) == 1


def test_grippers():
    from perceptpick.grippers import (
        all_grippers, get_gripper_class, GRIPPER_REGISTRY,
        Franka, Robotiq2F140, WSG50,
    )
    assert len(all_grippers) >= 9
    assert get_gripper_class("franka") is Franka
    assert "wsg_50" in GRIPPER_REGISTRY


def test_sampling_sim():
    from perceptpick.sampling import AntipodalGraspSampler
    from perceptpick.sim import (
        GraspScores, GraspSimulator, GraspSimulatorWithGravityControl,
    )
    assert GraspScores.SUCCESS is not None


def test_eval_metrics():
    from perceptpick.eval.pick import PickEvalConfig, evaluate_pick, output_path_for
    from perceptpick.eval.ranker import rank_grippers, best_gripper_for
    from perceptpick.metrics.pose_error import add, adi, mssd, mspd, re, te


def test_assets_viz():
    from perceptpick.assets.mesh_prep import prepare_one
    from perceptpick.viz.grasps_viewer import visualize_grasps


def test_configs():
    from perceptpick.configs import (
        GRIPPER_MAX_WIDTHS, MESH_SOURCES, SamplingConfig, SimulationConfig, YCB_OBJECTS,
    )
    assert len(YCB_OBJECTS) == 21
    assert "franka" in GRIPPER_MAX_WIDTHS
    assert SamplingConfig().n_grasps == 5000
