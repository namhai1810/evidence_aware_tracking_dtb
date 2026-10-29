# HIVE v2 + HR ReID để sửa ID Switch trên QuadTrack

## 1. Mục tiêu thực nghiệm

Thực nghiệm này kiểm tra xem historical frame do HIVE v2 chọn có thực sự giúp
sửa ID Switch hay không. Đây không còn là bài toán chỉ đánh giá HIVE chọn đúng
frame, mà là một pipeline hoàn chỉnh:

1. Nhận một ID Switch event từ manifest hiện có.
2. Chọn một historical frame bằng Nearest-frame hoặc HIVE v2.
3. Lấy đúng một HR crop từ panorama gốc `2048×480`.
4. Dùng ReID để quyết định merge hai ID hay giữ nguyên.
5. Replay lại ID trong kết quả MOT.
6. Đánh giá lại IDSW, IDF1, HOTA và MOTA trên tracking sau replay.

Ba phương pháp được so sánh:

| Method | Historical frame selector | HR budget | Sửa ID |
|---|---|---:|---|
| LR-only | Không dùng | 0 | Không |
| Nearest-frame + HR ReID | Historical candidate gần nhất | 1 crop/event | ReID policy |
| HIVE v2 + HR ReID | Frozen HIVE v2 | 1 crop/event | Cùng ReID policy |

Nearest-frame và HIVE dùng cùng model ReID, cùng policy và cùng ngân sách HR.
Do đó khác biệt cuối cùng chủ yếu đến từ chất lượng historical frame selector.

## 2. Những phần đã triển khai

### 2.1 Giữ nguyên HIVE v2

Checkpoint được dùng trực tiếp, không train hoặc fine-tune lại:

```text
hr_finding/runs/hive_history_v2_seed42/checkpoints/best.pt
```

SHA-256:

```text
954b23039cc590f0abdb37ab4edd9c77b3f0d37f3a9c573933065ff540b6d0dd
```

HIVE chỉ nhận các input LR đã dùng khi train:

- Target crop.
- Old-track history.
- Historical candidates.
- Competing-track history.
- Temporal metadata.

Các trường `gt_id`, candidate `label` và `matched_gt_id` không được truyền vào
`model.forward()`.

### 2.2 Truy xuất HR crop

Sau khi selector chọn một candidate, code mở frame panorama gốc `2048×480` và
crop theo bounding box của candidate. Mỗi method HR chỉ request đúng một crop
cho mỗi event.

HIVE không được nhìn ảnh HR trước khi chọn frame. Ảnh HR chỉ được đọc sau khi
HIVE hoặc Nearest-frame đã đóng băng lựa chọn.

### 2.3 HR ReID decision policy

Model ReID:

```text
yolo26n-reid.onnx
```

Model sinh embedding 512 chiều. Backend inference dùng OpenCV DNN nên không cần
cài thêm `onnxruntime`.

Với mỗi event, code tạo các embedding sau:

- `old`: prototype từ tối đa 5 LR crops của old identity.
- `target`: LR crop tại ID Switch frame.
- `hr`: HR crop từ historical frame đã chọn.
- `competitors`: prototype của tối đa 12 identity cạnh tranh trong 30 frame
  trước target.

Các score quan trọng:

```text
evidence_bridge = min(cos(hr, old), cos(hr, target))
evidence_margin = evidence_bridge - max(cos(hr, competitor_i))
target_margin   = cos(old, target) - max(cos(target, competitor_i))
```

Policy chỉ merge khi tất cả điều kiện đã tune trên validation đều đạt:

| Threshold | Giá trị đã chọn |
|---|---:|
| `bridge_min` | 0.50 |
| `old_target_min` | 0.70 |
| `evidence_margin_min` | 0.10 |
| `target_margin_min` | 0.10 |

Cùng một bộ threshold được dùng cho Nearest-frame và HIVE. Policy không đọc GT.

### 2.4 Replay tracking

Khi policy chấp nhận merge, code đổi ID cho continuous future segment của
`lr_new_pred_id` thành output identity của `lr_old_pred_id`:

- Không thay bounding box hoặc confidence.
- Không chạy lại detector và không thay thứ tự frame.
- Segment kết thúc nếu track bị ngắt quá 30 frame.
- Merge bị từ chối nếu tạo ra hai detection cùng output ID trong một frame.

Nhờ vậy, kết quả replay vẫn là MOT-format tracking hợp lệ và có thể đánh giá lại
trực tiếp.

### 2.5 Đánh giá sau replay

Pipeline tính lại IDSW, IDF1, HOTA, MOTA, detection metrics và kết quả theo từng
sequence.

Một event chỉ được tính là **thực sự sửa được** khi cùng GT identity có cùng
output ID tại historical matched frame và target frame sau replay.

`wrong correction` được tính khi merge đã được áp dụng nhưng:

- Không khôi phục được continuity của event; hoặc
- Segment bị sửa có dominant GT identity khác identity của event.

Vì vậy HIVE chọn đúng HR identity không tự động được tính là sửa ID thành công.

## 3. Chống GT leakage và cách chia dữ liệu

Validation dùng để tune threshold:

```text
0019, 0028, 0030
```

Held-out test chỉ dùng để đánh giá:

```text
0016, 0017, 0025, 0031
```

GT chỉ được dùng tại hai giai đoạn:

1. Tính objective để chọn threshold trên validation.
2. Tính tracking metrics và correction outcome sau replay trên held-out test.

GT không được dùng để chọn historical frame hoặc quyết định merge trên test.

Một giới hạn cần ghi rõ: timestamp và cặp old/new ID của event hiện lấy từ
evaluator-created ID-switch manifest. Vì vậy đây là thí nghiệm **ID correction
với oracle switch trigger**, chưa phải hệ thống tự phát hiện ID Switch online.

## 4. Kết quả kiểm tra full split

Code đã được chạy end-to-end trên toàn bộ validation và held-out test để kiểm tra
pipeline. Lần kiểm tra này được lưu tạm dưới `/tmp/hive_hr_reid_fullcheck`; khi
chạy command chính, kết quả lâu dài sẽ được ghi vào thư mục `outputs/`.

### Tracking metrics sau replay

| Method | HR requests | IDSW ↓ | IDF1 ↑ | HOTA ↑ | MOTA ↑ |
|---|---:|---:|---:|---:|---:|
| LR-only | 0 | 92 | 0.4476 | 0.3618 | 0.3538 |
| Nearest-frame + HR ReID | 92 | 92 | 0.4476 | 0.3618 | 0.3538 |
| HIVE v2 + HR ReID | 92 | **88** | **0.4503** | **0.3630** | **0.3541** |

So với LR-only, HIVE v2 + HR ReID:

- Giảm IDSW từ `92` xuống `88`, tương đương `4.3%`.
- Tăng IDF1 từ `0.4476` lên `0.4503` (`+0.0027`).
- Tăng HOTA từ `0.3618` lên `0.3630` (`+0.0011`).
- Không tạo thêm switch point mới trong lần kiểm tra này.

### Tách riêng retrieval và correction outcome

| Method | Chọn đúng HR identity | Policy yêu cầu merge | Merge thực thi | IDSW thực sự sửa | Sửa sai | Switch mới |
|---|---:|---:|---:|---:|---:|---:|
| Nearest-frame + HR ReID | 25/92 | 2 | 0 | 0 | 0 | 0 |
| HIVE v2 + HR ReID | 58/92 | 5 | 4 | **4** | **0** | **0** |

Hai merge của Nearest-frame bị replay safety check từ chối do conflict. HIVE
tạo 5 merge request, 4 merge được áp dụng và cả 4 event đều thực sự được sửa.

### Kết quả theo sequence

| Sequence | Method | IDSW | IDF1 | MOTA |
|---|---|---:|---:|---:|
| 0016 | LR-only | 13 | 0.3016 | 0.4579 |
| 0016 | HIVE + HR ReID | 13 | 0.3016 | 0.4579 |
| 0017 | LR-only | 17 | 0.3379 | 0.2912 |
| 0017 | HIVE + HR ReID | 15 | 0.3379 | 0.2918 |
| 0025 | LR-only | 61 | 0.4099 | 0.2748 |
| 0025 | HIVE + HR ReID | 60 | 0.4136 | 0.2749 |
| 0031 | LR-only | 1 | 0.9528 | 0.9170 |
| 0031 | HIVE + HR ReID | 0 | 0.9573 | 0.9177 |

HIVE chọn đúng HR identity `58/92` event, nhưng chỉ 4 event qua được ReID
threshold, replay safety check và tiêu chí actual correction. Điều này minh họa
đúng yêu cầu: **HIVE chọn đúng không đồng nghĩa ID Switch đã được sửa**.

## 5. Cách chạy

Chạy foreground:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code
bash run_hive_v2_hr_reid_experiment.sh
```

Chạy background để có thể đóng terminal:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code && nohup bash ./run_hive_v2_hr_reid_experiment.sh > ./outputs/hive_v2_hr_reid_replay.log 2>&1 < /dev/null & echo $!
```

Log:

```text
/home/dtb/workspace/evidence_aware_tracking/code/outputs/hive_v2_hr_reid_replay.log
```

Report cuối cùng:

```text
/home/dtb/workspace/evidence_aware_tracking/code/outputs/hive_v2_hr_reid_replay/full_run/REPORT.md
```

Smoke test với hai event mỗi split:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code
bash run_hive_v2_hr_reid_experiment.sh --max-events-per-split 2 --device cpu
```

Không dùng kết quả smoke test làm số liệu báo cáo vì nó không chứa đầy đủ split.

## 6. Output được tạo

```text
outputs/hive_v2_hr_reid_replay/full_run/
├── REPORT.md
├── summary.json
├── events/
│   ├── validation_evidence.jsonl
│   ├── test_evidence.jsonl
│   ├── validation_threshold_search.json
│   ├── nearest_hr_reid_decisions.jsonl
│   ├── hive_v2_hr_reid_decisions.jsonl
│   ├── *_metrics.json
│   └── *_failure_frames.json
├── predictions/
│   ├── lr_only/
│   ├── nearest_hr_reid/
│   └── hive_v2_hr_reid/
└── visuals/
    ├── tracking_metrics.png
    ├── correction_outcomes.png
    ├── validation_threshold_search.png
    ├── nearest_hr_reid_examples.jpg
    └── hive_v2_hr_reid_examples.jpg
```

Ý nghĩa chính:

- `REPORT.md`: báo cáo Markdown có bảng và hình minh họa.
- `summary.json`: config, split, thresholds và aggregate metrics.
- `*_decisions.jsonl`: ReID score, quyết định, lý do reject/merge và actual
  outcome của từng event.
- `predictions/`: MOT files sau replay để kiểm tra hoặc đánh giá lại.
- `visuals/`: biểu đồ tracking và contact sheet gồm old LR, target LR và HR
  evidence đã chọn.

## 7. Cấu trúc source code

```text
hive_hr_reid/
├── run_experiment.py   # Inference, tuning, replay, report và visualization
├── replay.py           # Identity merge, collision guard và outcome evaluation
├── hota.py             # HOTA evaluator và micro aggregation theo sequence
├── test_replay.py      # Regression tests không cần load model weights
└── README.md
```

Wrapper command:

```text
run_hive_v2_hr_reid_experiment.sh
```

Các regression test đã pass, gồm replay đúng tracklet, từ chối merge gây
same-frame collision, perfect tracking có HOTA bằng 1.0 và policy không đọc các
evaluation/GT fields.

