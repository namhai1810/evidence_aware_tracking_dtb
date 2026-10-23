# QuadTrack retrieval report index

This index distinguishes the historical GT/Hungarian oracle analyses from the cross-resolution embedding experiment.

| Report | Retrieval matching signal | Use of GT | Intended interpretation |
|---|---|---|---|
| [ByteTrack GT/Hungarian oracle](highres_retrieval_hungarian_gt_oracle_bytetrack_report.md) | GT identity assigned through Hungarian IoU matching | Selects and labels helpful candidates | Theoretical availability upper bound; not ReID retrieval |
| [BoT-SORT+ReID GT/Hungarian oracle](highres_retrieval_hungarian_gt_oracle_botsort_reid_report.md) | GT identity assigned through Hungarian IoU matching | Selects and labels helpful candidates | Oracle analysis of ReID-tracker outputs; not cross-resolution embedding retrieval |
| [Cross-resolution ReID embedding retrieval](highres_retrieval_reid_embedding_report.md) | Cosine similarity between low-resolution old-track/target embeddings and historical full-resolution detection embeddings | Identifies evaluation switch events and scores retrieved identities only; it does not rank candidates | Actual embedding-ranking experiment; track merging is not yet applied |

The embedding report is the relevant result for evaluating whether appearance features can retrieve useful original-resolution history. The two files containing `hungarian_gt_oracle` should be used only as oracle references.
