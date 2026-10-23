# QuadTrack visualization and person tracking

The scripts assume this layout:

```text
evidence_aware_tracking/
├── QuadTrack/train/<sequence>/{img1,gt/gt.txt}
└── code/
```

## Environment

Dependencies are installed in `code/.venv`:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code
source .venv/bin/activate
```

To reinstall them:

```bash
python -m pip install -r requirements.txt
```

The default device is automatic: CUDA GPU when PyTorch can see one, otherwise CPU.

## 1. Visualize ground truth

Create videos for the first three sequences (`0015`, `0016`, `0017`):

```bash
python visualize_ground_truth.py
```

Select two sequences explicitly:

```bash
python visualize_ground_truth.py --sequences 0015 0021 --fps 30
```

Quick 100-frame test:

```bash
python visualize_ground_truth.py --sequences 0015 --max-frames 100
```

Outputs are written to `outputs/ground_truth/`.

## 2. YOLO + ByteTrack and SORT

Run both trackers on sequence `0015`:

```bash
python track_people.py --sequences 0015 --tracker both
```

Run selected sequences with only ByteTrack on GPU 0:

```bash
python track_people.py --sequences 0015 0016 0017 --tracker bytetrack --device 0
```

Run SORT on CPU:

```bash
python track_people.py --sequences 0015 --tracker sort --device cpu
```

Quick smoke test:

```bash
python track_people.py --sequences 0015 --tracker both --max-frames 30 --imgsz 640
```

Useful options:

- `--weights yolo11n.pt`: detector weights; a local `.pt` path is also accepted.
- `--conf 0.25`: YOLO confidence threshold.
- `--imgsz 640`: inference resolution; increase for small people if GPU memory permits.
- `--max-frames N`: process only the first `N` frames.
- `--output-dir PATH`: change output location.
- `--run-name NAME`: append a controlled-experiment label such as `imgsz1280` to output names.

Each tracker creates an annotated MP4 and a MOT-format TXT file in `outputs/tracking/`.
The model name is included in each output name, so runs with different weights do not overwrite each other.
Both scripts use `tqdm`, which shows completed frames, elapsed time, estimated remaining time,
and processing speed.

## 3. Compare tracking results with ground truth

For example, compare YOLO11 small and large results at IoU 0.5:

```bash
python evaluate_mot.py \
  outputs/model_comparison/0015_yolo11s_bytetrack.txt \
  outputs/model_comparison/0015_yolo11l_bytetrack.txt \
  --gt ../QuadTrack/train/0015/gt/gt.txt \
  --iou 0.5 \
  --json outputs/model_comparison/metrics.json
```

The report includes detection precision, recall, F1 and mean matched IoU, plus tracking MOTA,
IDF1 and identity-switch count. These are baseline MOT calculations without ignored-region handling.
