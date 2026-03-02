# V-JEPA Robotics Dataset Pipeline

A fully runnable prototype of a production-grade data curation pipeline for
training **V-JEPA–style action-conditioned latent world models**.

---

## What This Does

Takes simulated raw robot logs → cleans & aligns them → generates structured
training windows → emits 9 quality flags → computes `example_quality_score`
→ outputs a training-ready dataset with a manifest index.

**Target model:** V-JEPA-style latent-space world model (predict future video
representations conditioned on robot actions, no pixel reconstruction).

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python pipeline/main.py

# 4. Inspect results
ls data_processed/examples/    # .npz training examples
cat data_processed/manifest.json | python3 -m json.tool | head -60
ls data_processed/plots/       # visualizations
```

---

## Folder Structure

```
vjepa_pipeline/
│
├── config.yaml                # All thresholds and parameters
├── requirements.txt
├── README.md
│
├── pipeline/
│   ├── __init__.py
│   ├── log_generator.py       # Stage 0: Synthetic raw logs
│   ├── validator.py           # Stage 1: Ingestion & validation
│   ├── aligner.py             # Stages 2-3: Drift correction + alignment
│   ├── sampler.py             # Stage 4: Window sampling
│   ├── quality.py             # Stage 5: Quality flag detection
│   ├── scorer.py              # Stage 6: EQS computation
│   ├── exporter.py            # Stage 7: Dataset serialization
│   └── main.py                # Orchestrator
│
├── data_raw/                  # Generated after running
│   └── ep_XXXX/
│       ├── frames/            # PNG video frames
│       ├── camera_timestamps.csv
│       ├── actions.csv
│       ├── states.csv
│       └── meta.json
│
└── data_processed/            # Generated after running
    ├── examples/              # Accepted .npz training examples
    ├── rejected/              # Below-threshold examples (kept for ablation)
    ├── manifest.csv           # Filterable index of all examples
    ├── manifest.json          # JSON version
    └── plots/
        ├── drift_correction.png
        ├── quality_histogram.png
        └── sample_windows.png
```

---

## Training Example Schema

Each `.npz` file contains one training window anchored at time `t`:

| Array         | Shape           | Description                                 |
| ------------- | --------------- | ------------------------------------------- |
| `ctx_video`   | `(60, H, W, 3)` | 2s × 30FPS RGB context frames               |
| `tgt_video`   | `(30, H, W, 3)` | 1s × 30FPS RGB prediction target            |
| `ctx_actions` | `(T, 6)`        | Action window t-2s → t+1s (6-DOF delta EEF) |
| `ctx_states`  | `(60, 17)`      | 2s of joint angles + EEF pose + quaternion  |

Metadata lives in `manifest.csv` columns:
`episode_id`, `anchor_time`, `example_quality_score`, `quality_bitmask`,
`quality_flags`, `action_alignment_method`, `event_tag`, `accepted`

---

## Pipeline Stages

### Stage 0 — LogGenerator

Fabricates synthetic robot episodes with configurable noise:

- Dropped frames, timestamp jitter, clock drift
- Robot stalls, exposure shifts, Gaussian blur
- Action execution lag, packet loss in state stream

### Stage 1 — EpisodeValidator

Hard-drops episodes missing required streams or below minimum duration.
In production: also verifies MD5 checksums and protobuf schema conformance.

### Stages 2-3 — DriftCorrector + TimestampAligner

- **DriftCorrector**: Linear regression on heartbeat timestamps estimates
  `T_ctrl = α·T_cam + Δt` and removes cumulative clock drift.
- **TimestampAligner**: Aligns 50Hz control signals to 30Hz video grid using:
  - **CubicSpline** for joint angles / EEF xyz (preserves C² continuity)
  - **SLERP** for quaternion orientations (correct geodesic interpolation in SO(3))
  - **Nearest-neighbor** for actions (avoids synthesizing commands never issued)

### Stage 4 — WindowSampler

Slides a (2s context + 1s target) window at configurable stride.
Enforces a **2-frame blind gap** at anchor time `t` to prevent action leakage
(model can't cheat by peeking at future frames).

### Stage 5 — QualityFlagger (9 flags)

| Flag        | Detection                                   | Type     |
| ----------- | ------------------------------------------- | -------- |
| `MISSING_V` | Black/missing frames > threshold            | **Hard** |
| `JITTER_A`  | High variance in action magnitude changes   | Soft     |
| `BLUR`      | Laplacian variance < threshold              | Soft     |
| `STALL`     | Cumulative joint delta < threshold          | Soft     |
| `EXPO_S`    | Max per-frame intensity jump > threshold    | Soft     |
| `SYNC_ERR`  | Action energy / state change ratio too high | Soft     |
| `DROP_P`    | NaN fraction in states > threshold          | **Hard** |
| `COMP_A`    | 8×8 DCT block boundary discontinuity        | Soft     |
| `JITTER_V`  | Video timestamp interval variance           | Soft     |

### Stage 6 — ExampleScorer

```
EQS = prod(1 - F_i for Hard flags) × exp(-Σ w_j × P_j for Soft flags)
```

- Hard flags collapse EQS to 0 immediately
- Soft flag weights configurable in `config.yaml`
- In production: weights tuned on 2% manually reviewed subset

### Stage 7 — DatasetExporter

Writes `.npz` files and `manifest.csv`. Examples below `min_quality_score`
go to `rejected/` (not discarded — threshold changes trigger selective backfill).

---

## Quality Score Formula

```python
# Hard penalty: any hard flag → 0
hard_product = 0.0 if any hard flag else 1.0

# Soft penalty: weighted exponential decay
eqs = hard_product * exp(-sum(weight[flag] for active soft flags))
```

---

## Inspecting Results

```python
import pandas as pd, numpy as np

# Load manifest
df = pd.read_csv("data_processed/manifest.csv")
print(df[["episode_id", "anchor_time", "example_quality_score", "quality_flags"]].head(10))

# Load one training example
data = np.load("data_processed/examples/ep_0000__t3_4381.npz")
print("Context video:", data["ctx_video"].shape)   # (60, 64, 64, 3)
print("Target video:",  data["tgt_video"].shape)   # (30, 64, 64, 3)
print("Actions:",       data["ctx_actions"].shape) # (T, 6)
print("States:",        data["ctx_states"].shape)  # (60, 17)
```

---

## Mapping to Production

| Prototype Component    | Production Equivalent                           |
| ---------------------- | ----------------------------------------------- |
| LogGenerator           | ROS bag parser / binary log extractor           |
| CSV storage            | Apache Parquet + WebDataset `.tar` shards       |
| Sequential processing  | Ray/Spark parallelism across episodes           |
| Laplacian blur check   | GPU-accelerated CUDA kernel via NV DALI         |
| manifest.csv           | Parquet feature store with lineage tracking     |
| config.yaml thresholds | ML feature store with A/B versioning            |
| 2% QA sampling         | Human review UI with uncertainty-based sampling |

---

## Configuration

All thresholds in `config.yaml` under the `quality:` key. Key parameters:

- `blur_laplacian_threshold`: Lower = more sensitive blur detection
- `stall_joint_delta_threshold`: Minimum cumulative joint motion to not flag
- `min_quality_score`: EQS threshold for accepted vs rejected examples
- `weights.*`: Per-flag soft penalty weights (tune on manually reviewed subset)
