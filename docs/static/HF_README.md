---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - robotics
  - grasping
  - pose-estimation
  - 3d-reconstruction
  - simulation
  - ycb
  - icra
pretty_name: PerceptPick Benchmark
size_categories:
  - 1G<n<10G
---

# PerceptPick — Benchmarking the Effects of Object Pose Estimation and Reconstruction on Robotic Grasping Success

This dataset accompanies the paper accepted at **IEEE ICRA 2026**:

> **Benchmarking the Effects of Object Pose Estimation and Reconstruction on Robotic Grasping Success**
> Varun Burde, Pavel Burget, Torsten Sattler — Czech Technical University in Prague.
> [arXiv:2602.17101](https://arxiv.org/abs/2602.17101) · [Project page](https://varunburde.github.io/perceptpick) · [Code](https://github.com/varunburde/perceptpick)

## Abstract

3D reconstruction serves as the foundational layer for numerous robotic perception tasks, including 6D object pose estimation and grasp pose generation. Modern 3D reconstruction methods for objects can produce visually and geometrically impressive meshes from multi-view images, yet standard geometric evaluations do not reflect how reconstruction quality influences downstream tasks such as robotic manipulation performance. This paper addresses this gap by introducing a large-scale, physics-based benchmark that evaluates 6D pose estimators and 3D mesh models based on their functional efficacy in grasping. We analyze the impact of model fidelity by generating grasps on various reconstructed 3D meshes and executing them on the ground-truth model, simulating how grasp poses generated with an imperfect model affect interaction with the real object. This assesses the combined impact of pose error, grasp robustness, and geometric inaccuracies from 3D reconstruction. Our results show that reconstruction artifacts significantly decrease the number of grasp pose candidates but have a negligible effect on grasping performance given an accurately estimated pose. Our results also reveal that the relationship between grasp success and pose error is dominated by spatial error, and even a simple translation error provides insight into the success of the grasping pose of symmetric objects.

---

## Layout

The dataset ships in the canonical tree layout used by the PerceptPick code — one folder per **mesh source** (oracle CAD `GT` plus eight reconstruction methods), each containing pre-built meshes, V-HACD convex decompositions, URDFs, and BOP-style pose-estimator CSVs.

```
ycbv/
├── GT/                                # Oracle CAD baseline
│   ├── meshes/obj_NNNNNN.obj
│   ├── vhacd/obj_NNNNNN_vhacd.obj
│   ├── urdf/obj_NNNNNN.urdf
│   └── pose_estimates/
│       ├── FoundationPose.csv         # FoundationPose run on GT meshes
│       └── MegaPose.csv               # MegaPose run on GT meshes
├── BakedSDF/, MonoSDF/, Nerfacto/     # Reconstructed-mesh sources
│   ├── meshes/, vhacd/, urdf/
│   └── pose_estimates/
│       ├── FoundationPose.csv         # FoundationPose run on this method's meshes
│       └── MegaPose.csv
├── Neuralangelo/, NGP/, RealCAP/, UniSurf/, VolSDF/
```

The tree maps 1-to-1 onto the `assets/ycbv/` folder of the [code repo](https://github.com/varunburde/perceptpick).

## Reconstruction methods

| Method | Type | Description |
|---|---|---|
| **GT** | Oracle | Ground-truth CAD model (upper bound) |
| **BakedSDF** | Neural SDF | Baked signed distance field reconstruction |
| **MonoSDF** | Neural SDF | Monocular depth-guided SDF |
| **Nerfacto** | NeRF | Nerfstudio's factored NeRF |
| **Neuralangelo** | Neural SDF | High-fidelity surface reconstruction from video |
| **NGP** | NeRF | Instant Neural Graphics Primitives |
| **RealCAP** | Classical | Real-world capture baseline |
| **UniSurf** | Neural SDF | Unified surface reconstruction |
| **VolSDF** | Neural SDF | Volume rendering with SDF |

## Pose estimators

- **FoundationPose** — transformer-based 6-DoF pose estimator
- **MegaPose** — large-scale generalizable pose estimator

Each `pose_estimates/<Estimator>.csv` is a BOP-style submission CSV with columns
`scene_id, im_id, obj_id, score, R, t, time`. The estimator was fed the mesh source
of its parent folder at inference time (e.g.
`ycbv/BakedSDF/pose_estimates/FoundationPose.csv` is FoundationPose run on the
BakedSDF reconstructed meshes).

## Objects

21 YCB objects from the YCB-Video object set (object IDs 1–21).

## Download

```bash
# Pull the entire tree directly into ./assets (~5.6 GB)
hf download varunburde/perceptpick --repo-type=dataset --local-dir assets
```

Or use the legacy CLI:

```bash
huggingface-cli download varunburde/perceptpick --repo-type=dataset --local-dir assets
```

To consume the data from Python:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

csv_path = hf_hub_download(
    "varunburde/perceptpick",
    "ycbv/BakedSDF/pose_estimates/FoundationPose.csv",
    repo_type="dataset",
)
df = pd.read_csv(csv_path)
print(df.head())
```

## Citation

```bibtex
@inproceedings{burde2026perceptpick,
  title     = {Benchmarking the Effects of Object Pose Estimation and
               Reconstruction on Robotic Grasping Success},
  author    = {Burde, Varun and Burget, Pavel and Sattler, Torsten},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  eprint    = {2602.17101},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi       = {10.48550/arXiv.2602.17101}
}
```

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The YCB-Video subset follows the original [YCB dataset license](http://www.ycbbenchmarks.com/).
