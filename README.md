# PerceptPick

**B**OP-style Benchmark for **O**bject **G**rasping — a physics-simulation framework
for evaluating how 6D pose estimation and 3D reconstruction errors propagate to
downstream robotic grasping success.

This is the reference implementation for:

> **Benchmarking the Effects of Object Pose Estimation and Reconstruction on
> Robotic Grasping Success**
> Varun Burde, Pavel Burget, Torsten Sattler — Czech Technical University in Prague.
> *Accepted at IEEE International Conference on Robotics and Automation (ICRA) 2026.*

## Why this benchmark exists

Modern 6D pose estimators (FoundationPose, MegaPose, …) and 3D reconstruction
methods (BakedSDF, MonoSDF, NeRFacto, Neuralangelo, Instant-NGP, UniSurf,
VolSDF, RealCapture, …) are typically evaluated in *isolation* using
task-agnostic geometric metrics — ADD, MSSD for poses; Chamfer distance for
reconstructions. Those numbers don't tell you whether a robot can actually
*pick the object up*.

The paper closes that gap. We measure grasping as the downstream task and
characterize how mesh fidelity and pose accuracy compound to determine
binary pick success, across millions of physics simulations:

- **9 grippers** × **21 YCB-V objects** × **8 reconstruction methods** ×
  **2 pose estimators**
- **5,000 antipodal grasp candidates** sampled per `(object, gripper)` pair
- **18.8M+ simulated grasps** at 240 Hz in PyBullet

## Headline findings

1. **Pose accuracy dominates grasp success**, not mesh quality. FoundationPose
   reaches ~89.9% average estimated success; MegaPose ~59.4%. Even with a
   reconstructed mesh used for both grasp generation *and* pose estimation,
   high-quality pose estimation can compensate for moderate mesh flaws.
2. **3D spatial metrics (MSSD / ADD / ADI / translation error) strongly
   correlate with grasp success.** 2D projection error (MSPD) and pure
   rotation error are *poor* predictors — many high-MSPD grasps still
   succeed. This validates that 2D metrics miss what matters for physical
   contact.
3. **Mesh artifacts reduce grasp candidate count** (more `Collision`
   failures during sampling, especially on Instant-NGP), but given an
   accurate pose the *executed* grasps still succeed. UniSurf, with smoother
   surfaces, occasionally beats the oracle CAD mesh because it avoids
   edge-case collisions.
4. **Symmetry matters.** Rotation error around an object's symmetry axis has
   negligible effect on grasp success; rotation error orthogonal to the axis
   degrades performance sharply.
5. **No single best gripper for all objects.** EZGripper wins on ~76% of
   objects in the YCB-V suite under ideal conditions; WSG-32 and
   Robotiq 2F-140 cover the rest.

## Pipeline

The paper formalises the benchmark as a transformation chain through four
frames — World (W), Camera (C), Object (O), Gripper (G):

```
T_w2g^gt  = T_w2c · T_c2o^gt  · T_o2g           (ideal: GT pose)
T_w2g^est = T_w2c · T_c2o^est · T_o2g           (realistic: estimated pose)
```

`T_o2g` is a precomputed canonical grasp library; `T_c2o^est` comes from a
6D pose estimator. The benchmark always executes the grasp on the
**ground-truth object pose** but plans it from `T_w2g^est` — this isolates
the perception error from real-world clutter and dynamics.

Three experimental conditions are evaluated, parameterised by which mesh
generates the grasp library and which mesh the pose estimator is given:

| Condition | Grasp Mesh | Pose Mesh | What it measures |
|---|---|---|---|
| **Oracle / Oracle**          | GT  | GT  | Ideal-baseline grasp performance |
| **Oracle / Reconstructed**   | GT  | rec | Pure pose-estimation error impact |
| **Reconstructed / Reconstructed** | rec | rec | End-to-end realistic perception |

Three stages, run as five numbered scripts:

```
1) scripts/01_prepare_assets.py    Stage A — VHACD + URDF for object meshes
2) scripts/02_grasp_sweep.py       Stage B — sample antipodal grasps, simulate
                                              each (object, gripper), and rank
                                              grippers per object
3) scripts/03_report_grippers.py   Stage B reporting — paper-style 4-panel
                                              summary + markdown table
4) scripts/04_evaluate.py          Stage C — benchmark with GT/reconstructed
                                              mesh + GT/estimated pose
5) scripts/05_visualize.py         Inspect grasps in Open3D (debug viewer)
```

Stage B's sweep runs headless and parallel by default (over the whole
21 × 9 sweep). Pass `--gui` for live PyBullet viewing of a single combo
or a small subset; pass `--headless` to Stage C for batch runs.

## Metrics

- **Grasp Generation Success Rate (S_gen)** — percentage of viable grasps
  produced by sampling on a given mesh; proxy for "grasp coverage" at the
  candidate-generation stage.
- **Estimated Success Rate (S_est)** — percentage of grasps that succeed when
  executed using a pose-estimator's output, conditioned on the grasp being
  successful under the ground-truth pose. This is the primary
  end-to-end metric.

Failure modes are categorised by physics-based outcome breakdown:

- **Successful** — gripper closes, holds, lifts the object against gravity
- **Slipped** — initial contact made, hold lost during lift
- **No Contact** — fingers close without touching the object (typically
  caused by translation error in the estimated pose)
- **Collision** — gripper body collides with the object during approach,
  preventing a valid grasp

## Install

```bash
pixi install
pixi shell
```

Installs `perceptpick` as an editable package plus the scientific stack
(numpy, pybullet, trimesh, open3d, pandas, …).

## Get the YCB-Video dataset

```bash
cd dataset
wget https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_base.zip
wget https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_models.zip
wget https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip
unzip ycbv_base.zip
unzip ycbv_models.zip -d ycbv
unzip ycbv_test_bop19.zip -d ycbv
```

You should end up with `dataset/ycbv/{models,models_eval,models_fine,test/000048..000059,...}`.

## Adding a new reconstructed-mesh source

To benchmark a 3D-reconstruction method that's not already in the paper:

1. **Drop your meshes** at `dataset/reconstructed_mesh/<YourMethod>/obj_NNNNNN.obj`
   — one file per YCB object (1-21), BOP filename convention. Meshes
   should already be in meters; the asset prep defaults `mesh_scale=1.0`
   for non-GT sources (only `GT` triggers the BOP mm→m conversion).
2. **Run asset prep** to generate URDF + VHACD:
   ```bash
   pixi run python scripts/01_prepare_assets.py --dataset ycbv --mesh-source <YourMethod>
   ```
   Outputs land at `assets/ycbv/<YourMethod>/{meshes,vhacd,urdf}/`.
3. **Run Stage B** to sample antipodal grasps on the new meshes and
   produce per-object gripper rankings:
   ```bash
   pixi run python scripts/02_grasp_sweep.py --dataset ycbv --mesh-source <YourMethod> --n-grasps 5000
   ```
4. (Optional) **Run Stage C** with the new mesh as either `--gt-mesh` or
   `--est-mesh` to measure pose-impact effects against an existing pose CSV.

## Adding a new pose estimator

To benchmark a 6D pose estimator that's not already in the paper:

1. **Run your pose estimator** on the YCB-V test set's RGB-D images.
   For each `(scene, image, object)` tuple in
   `test_targets_bop19.json`, output the estimated camera-frame pose
   `(R, t)` plus a confidence score and per-frame inference time.
2. **Format the output as a BOP-style CSV** with header
   `scene_id,im_id,obj_id,score,R,t,time` and one row per detection:
   - `R` — 9 space-separated floats (row-major 3×3 rotation)
   - `t` — 3 space-separated floats (translation in **millimeters**)
   - `score` — float confidence
   - `time` — float seconds (per-frame inference time)
3. **Drop the CSV** at
   `assets/ycbv/<MeshSource>/pose_estimates/<YourEstimator>.csv`. Pick
   the mesh-source folder corresponding to the 3D model you gave your
   pose estimator at inference time — typically `GT/` for an oracle
   baseline; `BakedSDF/` if you fed the BakedSDF reconstruction; etc.
4. **Run Stage C** referencing the CSV:
   ```bash
   pixi run python scripts/04_evaluate.py --dataset ycbv \
       --gt-mesh GT --est-mesh <MeshSource> \
       --pose-csv <YourEstimator>.csv \
       --gripper auto --workers 4 --resume --headless
   ```

## Pre-prepared assets (download from Hugging Face)

If you'd rather reproduce the paper's numbers without running pose
estimation or 3D reconstruction yourself, we ship the assets directly on
[`varunburde/perceptpick`](https://huggingface.co/datasets/varunburde/perceptpick) in
the canonical layout — **no zip / no rearrange step**. The HF repo's
`ycbv/` folder maps 1-to-1 onto this repo's `assets/ycbv/`:

- **9 mesh sources** — oracle CAD (`GT`) plus 8 reconstruction methods:
  `BakedSDF, MonoSDF, Nerfacto, Neuralangelo, NGP, RealCAP, UniSurf,
  VolSDF`. Each ships pre-built `meshes/`, `vhacd/`, and `urdf/`.
- **18 pose-estimator CSVs** — FoundationPose and MegaPose run on each
  of the 9 mesh sources, co-located inside each method's
  `pose_estimates/` subfolder.

```bash
# 1. Download the whole assets tree directly into ./assets (~7 GB)
hf download varunburde/perceptpick --repo-type=dataset --local-dir assets

# 2. Skip Stage A entirely; jump to Stage B / C
pixi run python scripts/02_grasp_sweep.py --dataset ycbv --mesh-source GT --n-grasps 5000
pixi run python scripts/04_evaluate.py --dataset ycbv \
    --gt-mesh GT --est-mesh GT \
    --pose-csv FoundationPose.csv --gripper auto --workers 4 --resume --headless
```

The HF repo's per-method `pose_estimates/` matches the layout the new
pose estimator from the previous section drops into — drop your own CSVs
alongside `FoundationPose.csv` / `MegaPose.csv` to compare with the
paper's baselines.

## Dataset layout

The benchmark expects a **BOP-style** directory tree under `<dataset_root>`
(default `<repo>/dataset/`) plus a parallel `<assets_root>` (default
`<repo>/assets/`) holding pre-prepared URDFs, VHACDs, and pose CSVs.

```
dataset/                                     # <dataset_root>
└── ycbv/                                    # one folder per BOP dataset (--dataset)
    ├── camera_*.json
    ├── dataset_info.md
    ├── test_targets_bop19.json
    ├── models/                              # GT CAD meshes (BOP)
    │   ├── obj_000001.ply ... obj_000021.ply
    │   └── models_info.json                 # symmetry annotations
    ├── models_eval/                         # ADD/MSSD eval meshes
    ├── models_fine/                         # high-resolution GT (optional)
    └── test/                                # BOP test scenes
        └── 000048/, 000049/, ...
            ├── scene_camera.json, scene_gt.json, scene_gt_info.json
            ├── rgb/, depth/

assets/                                      # <assets_root>
└── ycbv/
    ├── GT/                                  # oracle CAD assets
    │   ├── meshes/obj_NNNNNN.obj
    │   ├── vhacd/obj_NNNNNN_vhacd.obj
    │   ├── urdf/obj_NNNNNN.urdf
    │   └── pose_estimates/                  # pose-estimator CSVs
    │       ├── FoundationPose.csv           # FoundationPose on GT meshes
    │       └── MegaPose.csv                 # MegaPose on GT meshes
    ├── BakedSDF/, MonoSDF/, Nerfacto/       # reconstructed-mesh sources
    │   ├── meshes/, vhacd/, urdf/
    │   └── pose_estimates/
    │       ├── FoundationPose.csv           # FoundationPose on this method
    │       └── MegaPose.csv
    ├── Neuralangelo/, NGP/, RealCAP/, UniSurf/, VolSDF/

dataset/reconstructed_mesh/                  # OPTIONAL — only if you regenerate
└── <method>/                                # assets locally via 01_prepare_assets.py.
    └── obj_NNNNNN.obj                       # Lowercase folder names match the
                                             # legacy 01_prepare_assets discovery.
```

- **GT meshes** in `<dataset>/models/` come from BOP's `ycbv_models.zip`.
- **Pre-prepared assets** (per-method `meshes/, vhacd/, urdf/, pose_estimates/`)
  ship as a single archive on the project's HF dataset; download and unpack
  into `<assets_root>/ycbv/`. Skip Stage A if you go this route.
- **Reconstructed source meshes** are only needed if you regenerate assets
  locally with `scripts/01_prepare_assets.py --all-mesh-sources`. Drop
  them at `dataset/reconstructed_mesh/<method>/obj_NNNNNN.obj`.
- **Pose CSVs** are co-located inside each `<method>/pose_estimates/`
  folder. Filename = canonical estimator name (`FoundationPose.csv`,
  `MegaPose.csv`). The CSV uses the BOP submission format:
  `scene_id, im_id, obj_id, score, R, t, time`.
- **Mesh-source names** are paper-cased and case-sensitive:
  `GT, BakedSDF, MonoSDF, Nerfacto, Neuralangelo, NGP, RealCAP, UniSurf, VolSDF`.

## Quickstart — run the whole benchmark

The benchmark unit of work is the whole dataset. Each command below
processes all 21 YCB objects (and the 9-gripper paper sweep for Stage B).
Everything is resumable, so you can Ctrl-C and restart at any time.

```bash
# Stage A — assets for the GT CAD meshes (and any reconstruction methods you have)
pixi run python scripts/01_prepare_assets.py --dataset ycbv --all-mesh-sources

# Stage B — sample antipodal grasps for every (object, gripper),
#          simulate each in PyBullet, rank grippers per object by success rate.
#          Single fire-and-forget command; writes output/grasp_poses/GT/gripper_rankings.json at the end.
pixi run python scripts/02_grasp_sweep.py --dataset ycbv --mesh-source GT --n-grasps 5000

# Stage C — three conditions corresponding to the paper's table

# Condition 1: Oracle / Oracle (ideal baseline)
pixi run python scripts/04_evaluate.py --dataset ycbv \
    --gt-mesh GT --est-mesh GT \
    --pose-csv FoundationPose.csv --gripper auto --workers 4 --resume --headless

# Condition 2: Oracle for grasps, Reconstructed for pose (isolating pose error)
pixi run python scripts/04_evaluate.py --dataset ycbv \
    --gt-mesh GT --est-mesh BakedSDF \
    --pose-csv FoundationPose.csv --gripper auto --workers 4 --resume --headless

# Condition 3: Reconstructed for both (end-to-end realistic)
pixi run python scripts/04_evaluate.py --dataset ycbv \
    --gt-mesh BakedSDF --est-mesh BakedSDF \
    --pose-csv FoundationPose.csv --gripper auto --workers 4 --resume --headless
```

## Stage B reporting

After Stage B finishes, generate a paper-style 4-panel summary plus a
markdown table from `output/grasp_poses/<mesh-source>/gripper_rankings.json`:

```bash
# Outputs: output/reports/gripper_analysis_<mesh-source>.{png,md}
pixi run python scripts/03_report_grippers.py --mesh-source GT
```

The PNG mirrors the paper's Fig 2 (per-object success heatmap, best-gripper
share pie, per-gripper average bar, failure-mode stacked bar). Pass
`--no-plot` to skip matplotlib and write only the markdown table.

## Watch one grasp simulate (debug)

When you want to see what the simulator is doing, the grasp-sweep script
also accepts a single `(object, gripper)` combo. In that mode the PyBullet
GUI opens and the rankings file is left untouched:

```bash
pixi run python scripts/02_grasp_sweep.py \
    --dataset ycbv --object 7 --gripper robotiq_2f_140 --n-grasps 50

# Browse the resulting grasps in Open3D
pixi run python scripts/05_visualize.py --object 7 --gripper robotiq_2f_140
```

## Output layout

```
output/
├── grasp_poses/<mesh-source>/<object_name>/<gripper>.json    # B-1 outputs
├── grasp_poses/<mesh-source>/gripper_rankings.json           # B-2 output (one per source)
├── reports/gripper_analysis_<mesh-source>.{png,md}           # B reporting
├── picking_success/<gt-mesh>_vs_<est-mesh>/<pose-method>/    # C outputs
│       <scene>_<image>_<obj>.json
└── visualization/                                            # plots
```

## Simulation details

- Friction coefficient on the GT object: 0.5 (representative of plastic on
  rubber); deliberately suppresses pure dynamic-slip failures so that
  collisions and no-contact reflect planning- and perception-driven errors.
- Simulation rate: 240 Hz, 100 settle iterations.
- No ground plane. Gravity is disabled until after gripper closure to prevent
  premature object motion, then enabled at -9.81 m/s² to test stable lifts.

## Supported reconstruction methods

`GT` (oracle CAD) plus `BakedSDF, MonoSDF, Nerfacto, Neuralangelo,
NGP, RealCAP, UniSurf, VolSDF`. Names match the paper's tables and are
case-sensitive — they're used verbatim as folder names under
`assets/<dataset>/<mesh-source>/`.

## Supported grippers

Two-finger: `franka, robotiq_2f_85, robotiq_2f_140, wsg_50, wsg_32, ezgripper, sawyer`
Three-finger: `robotiq_3f, kinova_3f, barrett, barrett_2f`

## Supported pose estimators (CSV inputs)

The benchmark consumes BOP-style pose result CSVs
(`scene_id, im_id, obj_id, score, R, t, time`). Drop them under each
mesh source's `pose_estimates/` folder, named after the estimator:

```
assets/ycbv/GT/pose_estimates/FoundationPose.csv     # FoundationPose on GT
assets/ycbv/GT/pose_estimates/MegaPose.csv           # MegaPose on GT
assets/ycbv/BakedSDF/pose_estimates/FoundationPose.csv  # FoundationPose on BakedSDF
...
```

The paper reports results for FoundationPose and MegaPose. For
backwards-compat the legacy flat layout
(`dataset/<dataset>/methods_poses/<name>.csv`) still works as a fallback.

## Acknowledgments

This package is **heavily derived from
[burg-toolkit](https://github.com/mrudorfer/burg-toolkit)** (MIT License,
Copyright © 2022 Martin Rudorfer). The gripper URDFs and meshes, the
antipodal grasp sampler, the PyBullet simulation harness, the core data
structures, and most utility modules in `perceptpick/{core,sampling,sim,viz,utils}/`
trace back to that codebase. Please cite burg-toolkit if you use this
benchmark in academic work.

The pose error metrics in `perceptpick/metrics/pose_error.py` are taken
verbatim from the [BOP toolkit](https://github.com/thodan/bop_toolkit) by
Tomas Hodan (CTU Prague).

## Citation

Accepted at **IEEE International Conference on Robotics and Automation (ICRA) 2026**.

```bibtex
@inproceedings{burde2026perceptpick,
  title     = {Benchmarking the Effects of Object Pose Estimation and
               Reconstruction on Robotic Grasping Success},
  author    = {Burde, Varun and Burget, Pavel and Sattler, Torsten},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```
