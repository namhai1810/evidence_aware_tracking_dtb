# HIVE v2 — HR Historical Frame Ranking Model

> HIVE v2 không đọc ảnh HR. Model sử dụng bằng chứng LR để xếp hạng frame lịch sử nào nên được truy xuất lại ở độ phân giải `2048×480`. HSV similarity không phải input của model.

## 1. Bài toán

Tại một ID-switch event ở frame `T`, model nhận old-track history, target hiện tại, tối đa 20 historical candidates và lịch sử của competing ID. Output là một utility score cho từng candidate.

```mermaid
flowchart LR
    A["ID switch tại frame T"] --> B["Tìm lịch sử T−1 … T−150"]
    B --> C["Tối đa 20 LR candidates"]
    C --> D["HIVE v2 ranking"]
    D --> E["Top-1 historical frame"]
    E -. "chưa triển khai" .-> F["Retrieve frame HR 2048×480"]
    F -.-> G["HR ReID + merge identity"]
    G -.-> H["Tracking replay: IDSW / IDF1 / MOTA"]
```

Phần đã triển khai kết thúc ở `Top-1 historical frame`. HR retrieval, identity merge và tracking replay là bước tiếp theo.

## 2. Input tensors

Với batch size mặc định `B = 8`:

| Tensor | Shape | Nội dung |
|---|---:|---|
| `target_crop` | `[B, 3, 128, 64]` | Target tại switch frame `T` |
| `old_crops` | `[B, 5, 3, 128, 64]` | Tối đa 5 crop của old track |
| `old_mask` | `[B, 5]` | Old-history slot hợp lệ |
| `candidate_crops` | `[B, 20, 3, 128, 64]` | Tối đa 20 historical candidates |
| `candidate_mask` | `[B, 20]` | Candidate slot hợp lệ |
| `competitor_crops` | `[B, 5, 3, 128, 64]` | History của predicted ID mới |
| `competitor_mask` | `[B, 5]` | Competitor slot hợp lệ |
| `metadata` | `[B, 20, 3]` | `age`, `age²`, `log_age` |
| `labels` | `[B, 20]` | Positive/negative khi training |

```mermaid
flowchart TB
    subgraph I["Một event trong batch"]
        O["Old history<br/>≤5 × 3×128×64"]
        T["Target<br/>3×128×64"]
        C["Candidates<br/>≤20 × 3×128×64"]
        U["Competitor history<br/>≤5 × 3×128×64"]
        M["Temporal metadata<br/>20 × 3"]
    end

    O --> ENC["Shared Tiny CNN"]
    T --> ENC
    C --> ENC
    U --> ENC

    ENC --> OE["Old embeddings<br/>B×5×64"]
    ENC --> TE["Target embedding<br/>B×64"]
    ENC --> CE["Candidate embeddings<br/>B×20×64"]
    ENC --> UE["Competitor embeddings<br/>B×5×64"]
```

Các slot thiếu được padding bằng zero. Mask đảm bảo padded crops không được encode và không tham gia ranking.

## 3. LR crop preprocessing

```mermaid
flowchart LR
    A["Original panorama<br/>2048×480"] --> B["Resize toàn panorama<br/>512×120"]
    B --> C["Scale bounding box"]
    C --> D["Crop người trên LR frame"]
    D --> E["Letterbox 128×64<br/>fill = 114"]
    E --> F["BGR → RGB"]
    F --> G["÷ 255"]
    G --> H["ImageNet normalization"]
    H --> I["Tensor 3×128×64"]
```

Không crop người ở HR rồi resize crop xuống LR. Toàn panorama được resize trước để mô phỏng đúng dữ liệu edge.

## 4. Shared Tiny Appearance Encoder

Mọi old, target, candidate và competitor crop dùng chung một CNN.

```mermaid
flowchart LR
    X["3×128×64"] --> C0["Conv 3×3 s2<br/>3→16"]
    C0 --> S0["16×64×32"]
    S0 --> D1["DS Block s2<br/>16→24"]
    D1 --> S1["24×32×16"]
    S1 --> D2["DS Block s2<br/>24→32"]
    D2 --> S2["32×16×8"]
    S2 --> D3["DS Block s2<br/>32→48"]
    D3 --> S3["48×8×4"]
    S3 --> D4["DS Block s2<br/>48→64"]
    D4 --> S4["64×4×2"]
    S4 --> P["AdaptiveAvgPool"]
    P --> V["64×1×1"]
    V --> F["Flatten + Dropout 0.2"]
    F --> L["Linear 64→64"]
    L --> N["L2 normalize"]
    N --> Z["Embedding ∈ R⁶⁴"]
```

Mỗi Depthwise-Separable Block:

```mermaid
flowchart LR
    A["Input"] --> B["Depthwise Conv 3×3"]
    B --> C["BatchNorm + SiLU"]
    C --> D["Pointwise Conv 1×1"]
    D --> E["BatchNorm + SiLU"]
```

| CNN layer | Output của một crop |
|---|---:|
| Input | `[3,128,64]` |
| Conv `3×3`, stride 2 | `[16,64,32]` |
| DS `16→24`, stride 2 | `[24,32,16]` |
| DS `24→32`, stride 2 | `[32,16,8]` |
| DS `32→48`, stride 2 | `[48,8,4]` |
| DS `48→64`, stride 2 | `[64,4,2]` |
| Adaptive average pooling | `[64,1,1]` |
| Linear + L2 normalization | `[64]` |

## 5. Target-conditioned old-history attention

Với old embedding `hᵢ ∈ R⁶⁴` và target embedding `t ∈ R⁶⁴`:

```text
xᵢ = concat(hᵢ, t, |hᵢ − t|) ∈ R¹⁹²
aᵢ = W₂ tanh(W₁xᵢ + b₁) + b₂
αᵢ = masked_softmax(aᵢ)
old = Σᵢ αᵢhᵢ ∈ R⁶⁴
```

Ma trận attention:

```text
W₁ ∈ R⁶⁴ˣ¹⁹²
W₂ ∈ R¹ˣ⁶⁴
```

```mermaid
flowchart LR
    H["Old embeddings<br/>B×5×64"] --> CAT["concat(hᵢ, t, |hᵢ−t|)<br/>B×5×192"]
    T["Target<br/>B×64"] --> CAT
    CAT --> A1["Linear 192→64"]
    A1 --> TANH["Tanh"]
    TANH --> A2["Linear 64→1"]
    A2 --> SM["Masked Softmax<br/>B×5"]
    SM --> SUM["Weighted Sum"]
    H --> SUM
    SUM --> O["Old prototype<br/>B×64"]
```

Competitor history dùng một attention aggregator khác nhưng có cùng cấu trúc, tạo competitor prototype `[B,64]`.

## 6. Candidate Relation Encoder

Với mỗi candidate `c`, old prototype `o` và target `t`, model tạo vector:

```text
r₀ = concat(
    o, t, c,
    |o−t|, |o−c|, |t−c|,
    o⊙t, o⊙c, t⊙c,
    cos(o,t), cos(o,c), cos(t,c)
)
```

Trong đó `⊙` là element-wise product.

```text
9 × 64 + 3 cosine values = 579 dimensions
r₀ shape = [B,20,579]
```

```mermaid
flowchart LR
    O["Old prototype<br/>B×64"] --> R["Relation construction"]
    T["Target<br/>B×64"] --> R
    C["Candidates<br/>B×20×64"] --> R
    R --> R0["B×20×579"]
    R0 --> L1["Linear 579→96"]
    L1 --> N1["LayerNorm + SiLU<br/>Dropout 0.2"]
    N1 --> L2["Linear 96→96"]
    L2 --> S["SiLU"]
    S --> OUT["Relation features<br/>B×20×96"]
```

Các ma trận:

```text
Wrelation₁ ∈ R⁹⁶ˣ⁵⁷⁹
Wrelation₂ ∈ R⁹⁶ˣ⁹⁶
```

## 7. Temporal Metadata Encoder

Với candidate cách switch `s` giây:

```text
age     = clip(s / 5, 0, 1)
age²    = age × age
log_age = log(1+s) / log(1+5)
```

```mermaid
flowchart LR
    M["[age, age², log_age]<br/>B×20×3"] --> L["Linear 3→16"]
    L --> N["LayerNorm 16"]
    N --> S["SiLU"]
    S --> O["Temporal features<br/>B×20×16"]
```

```text
Wmetadata ∈ R¹⁶ˣ³
```

## 8. Competing Hypothesis Encoder

Với competitor prototype `u`:

```text
q₀ = concat(
    u,
    |c−u|,
    c⊙u,
    cos(c,u),
    cos(t,u),
    competitor_available
)
```

```text
3 × 64 + 3 = 195 dimensions
q₀ shape = [B,20,195]
```

```mermaid
flowchart LR
    U["Competitor<br/>B×64"] --> Q["Hypothesis construction"]
    C["Candidate<br/>B×20×64"] --> Q
    T["Target<br/>B×64"] --> Q
    A["Available mask"] --> Q
    Q --> Q0["B×20×195"]
    Q0 --> L["Linear 195→48"]
    L --> N["LayerNorm + SiLU<br/>Dropout 0.2"]
    N --> MASK["× available mask"]
    MASK --> OUT["Competitor features<br/>B×20×48"]
```

```text
Wcompetitor ∈ R⁴⁸ˣ¹⁹⁵
```

Nếu competitor history không tồn tại, toàn bộ nhánh này trả về zero.

## 9. Feature fusion và ranking head

```mermaid
flowchart LR
    R["Candidate relation<br/>B×20×96"] --> CAT["Concatenate"]
    M["Temporal features<br/>B×20×16"] --> CAT
    C["Competitor features<br/>B×20×48"] --> CAT
    CAT --> Z["B×20×160"]
    Z --> L1["Linear 160→96"]
    L1 --> N["LayerNorm + SiLU<br/>Dropout 0.2"]
    N --> L2["Linear 96→1"]
    L2 --> LOGIT["Utility logits<br/>B×20"]
    LOGIT --> MASK["Invalid candidates → −10000"]
    MASK --> SORT["Sort descending"]
    SORT --> TOP["Top-1 / Top-3 / Top-5 frames"]
```

Feature fusion:

```text
96 relation + 16 temporal + 48 competitor = 160 dimensions
```

Scorer matrices:

```text
Wscore₁ ∈ R⁹⁶ˣ¹⁶⁰
Wscore₂ ∈ R¹ˣ⁹⁶
```

Raw logits được dùng để ranking. Probability chỉ dùng cho classification/request threshold:

```text
probability = sigmoid(logit)
```

## 10. End-to-end tensor flow

```mermaid
flowchart TB
    T0["Target crop<br/>B×3×128×64"] --> ENC["Shared CNN"]
    O0["Old crops<br/>B×5×3×128×64"] --> ENC
    C0["Candidate crops<br/>B×20×3×128×64"] --> ENC
    U0["Competitor crops<br/>B×5×3×128×64"] --> ENC

    ENC --> T["Target B×64"]
    ENC --> O1["Old B×5×64"]
    ENC --> C["Candidates B×20×64"]
    ENC --> U1["Competitor B×5×64"]

    O1 --> OA["Old attention"]
    T --> OA
    OA --> O["Old prototype B×64"]

    U1 --> UA["Competitor attention"]
    T --> UA
    UA --> U["Competitor prototype B×64"]

    O --> REL["Candidate relation<br/>B×20×96"]
    T --> REL
    C --> REL

    U --> COMP["Competitor relation<br/>B×20×48"]
    T --> COMP
    C --> COMP

    META["Metadata B×20×3"] --> ME["Metadata encoder<br/>B×20×16"]

    REL --> FUSE["Concat B×20×160"]
    COMP --> FUSE
    ME --> FUSE
    FUSE --> SCORE["Scorer B×20×1"]
    SCORE --> OUT["Ranking logits B×20"]
```

## 11. Training loss

Label của mỗi candidate:

```text
1 = candidate thuộc đúng GT identity
0 = candidate thuộc người khác hoặc unmatched
```

### Weighted BCE

```text
Lcls = BCEWithLogits(score, label)
```

Positive weight được tính tự động từ train set, hiện khoảng `2.253`.

### Event-normalized RankNet

Với positive score `s⁺` và negative score `s⁻` trong cùng event:

```text
Lpair = softplus(−(s⁺−s⁻))
      = log(1 + exp(s⁻−s⁺))
```

Loss buộc `s⁺ > s⁻`. Các pairs được average trong từng event rồi mới average giữa events.

```mermaid
flowchart LR
    P["Positive scores"] --> PAIR["All positive-negative pairs"]
    N["Negative scores"] --> PAIR
    PAIR --> DIFF["s⁺ − s⁻"]
    DIFF --> SP["softplus(−difference)"]
    SP --> EV["Mean per event"]
    EV --> B["Mean across batch"]
```

Loss tổng:

```text
L = 1.0 × Lcls + 0.5 × Lrank
```

Teacher/margin loss hiện bị tắt: `lambda_margin = 0`.

## 12. Parameter count

| Module | Registered parameters |
|---|---:|
| Shared appearance CNN | 12,040 |
| Old + competitor attention | 24,962 |
| Simple appearance branch | 18,720 |
| Candidate relation encoder | 65,184 |
| Competitor relation encoder | 9,504 |
| Metadata encoder | 96 |
| Final scorer | 15,745 |
| **Total registered** | **146,251** |

`hive_full` dùng `CandidateRelationEncoder`, nhưng code vẫn khởi tạo `SimpleAppearanceRelation`. Vì vậy 18,720 parameters của simple branch được đăng ký nhưng không đi qua forward path của `hive_full`. Forward path thực tế dùng khoảng `127,531` parameters.

## 13. Kết quả hiện tại

Held-out test gồm 92 ID-switch events:

```mermaid
flowchart LR
    E["92 test events"] --> O["80 events có positive<br/>Oracle 86.96%"]
    E --> NP["12 events không có positive"]
    O --> H["HIVE top-1 đúng<br/>58 events"]
    O --> W["Có positive nhưng top-1 sai<br/>22 events"]
```

| Phương pháp | Correct top-1 | Success@1 |
|---|---:|---:|
| Nearest historical frame | 25/92 | 27.17% |
| LR proposal order | 48/92 | 52.17% |
| **HIVE v2** | **58/92** | **63.04%** |
| Candidate-pool oracle | 80/92 | 86.96% |

| HIVE metric | Kết quả |
|---|---:|
| Success@1 | 63.04% |
| Success@3 | 78.26% |
| Success@5 | 83.70% |
| MRR | 72.19% |
| NDCG@5 | 68.97% |
| Model latency | 0.815 ms/event |

So với nearest-frame, HIVE chọn đúng thêm 33 events. So với LR proposal order, HIVE chọn đúng thêm 10 events.

Kết quả này đo khả năng **chọn đúng historical evidence**. Nó chưa phải số ID switches đã được sửa, vì HR ReID, identity merge và tracking replay chưa được triển khai.

## 14. Source files

- `models/appearance_encoder.py`: Tiny CNN và attention aggregator.
- `models/hypothesis_encoder.py`: candidate relation và competing hypothesis.
- `models/hive.py`: feature fusion và ranking head.
- `datasets/dataset.py`: event tensors và masks.
- `datasets/preprocessing.py`: LR reconstruction, letterbox và temporal metadata.
- `losses/ranking_losses.py`: BCE và RankNet loss.
- `configs/hive_history_v2.yaml`: cấu hình HIVE v2 hiện tại.
