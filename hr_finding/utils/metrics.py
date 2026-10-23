"""Candidate-level classification and event-level ranking metrics."""

from __future__ import annotations

import math

import torch


class MetricAccumulator:
    def __init__(
        self,
        classification_threshold: float = 0.5,
        request_threshold: float | None = None,
    ) -> None:
        self.threshold = classification_threshold
        self.request_threshold = (
            classification_threshold if request_threshold is None else request_threshold
        )
        self._scores: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._masks: list[torch.Tensor] = []
        self._event_ids: list[str] = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        event_ids: list[str],
    ) -> None:
        self._scores.append(logits.detach().float().cpu())
        self._labels.append(labels.detach().float().cpu())
        self._masks.append(mask.detach().bool().cpu())
        self._event_ids.extend(event_ids)

    def compute(self) -> dict[str, float | int]:
        if not self._scores:
            return {}
        scores = torch.cat(self._scores, dim=0)
        labels = torch.cat(self._labels, dim=0)
        masks = torch.cat(self._masks, dim=0)
        valid_scores = scores[masks]
        valid_labels = labels[masks]
        probabilities = torch.sigmoid(valid_scores)
        predictions = probabilities >= self.threshold
        positives = valid_labels > 0.5
        tp = int((predictions & positives).sum())
        fp = int((predictions & ~positives).sum())
        fn = int((~predictions & positives).sum())
        tn = int((~predictions & ~positives).sum())
        total_candidates = max(1, tp + fp + fn + tn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        total_events = scores.shape[0]
        events_with_candidates = 0
        oracle_events = 0
        requested_events = 0
        correct_requested_events = 0
        successes = {1: 0, 3: 0, 5: 0}
        reciprocal_rank_sum = 0.0
        ndcg_sum = 0.0
        eligible_reciprocal_rank = 0.0
        eligible_ndcg = 0.0
        for event_scores, event_labels, event_mask in zip(scores, labels, masks):
            event_scores = event_scores[event_mask]
            event_labels = event_labels[event_mask]
            if not event_scores.numel():
                continue
            events_with_candidates += 1
            order = torch.argsort(event_scores, descending=True)
            ranked_labels = event_labels[order]
            top_probability = float(torch.sigmoid(event_scores[order[0]]))
            if top_probability >= self.request_threshold:
                requested_events += 1
                correct_requested_events += int(bool(ranked_labels[0] > 0.5))
            positive_positions = torch.nonzero(ranked_labels > 0.5, as_tuple=False).flatten()
            has_positive = bool(positive_positions.numel())
            if has_positive:
                oracle_events += 1
                first_rank = int(positive_positions[0]) + 1
                rr = 1.0 / first_rank
                reciprocal_rank_sum += rr
                eligible_reciprocal_rank += rr
                for k in successes:
                    successes[k] += int(bool((ranked_labels[:k] > 0.5).any()))

                gains = ranked_labels[:5].clamp_min(0)
                discounts = torch.tensor(
                    [1.0 / math.log2(index + 2) for index in range(len(gains))]
                )
                dcg = float((gains * discounts).sum())
                ideal = torch.sort(event_labels, descending=True).values[:5].clamp_min(0)
                idcg = float((ideal * discounts[: len(ideal)]).sum())
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_sum += ndcg
                eligible_ndcg += ndcg

        denominator = max(1, total_events)
        eligible_denominator = max(1, oracle_events)
        oracle_coverage = oracle_events / denominator
        success_at_1 = successes[1] / denominator
        return {
            "accuracy": (tp + tn) / total_candidates,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "candidate_count": int(masks.sum()),
            "event_count": total_events,
            "events_with_candidates": events_with_candidates,
            "events_with_positive": oracle_events,
            "events_without_positive": total_events - oracle_events,
            "success_at_1": success_at_1,
            "success_at_3": successes[3] / denominator,
            "success_at_5": successes[5] / denominator,
            "success_at_1_when_retrievable": successes[1] / eligible_denominator,
            "success_at_3_when_retrievable": successes[3] / eligible_denominator,
            "success_at_5_when_retrievable": successes[5] / eligible_denominator,
            "mrr": reciprocal_rank_sum / denominator,
            "mrr_when_retrievable": eligible_reciprocal_rank / eligible_denominator,
            "ndcg_at_5": ndcg_sum / denominator,
            "ndcg_at_5_when_retrievable": eligible_ndcg / eligible_denominator,
            "oracle_coverage": oracle_coverage,
            "oracle_gap_at_1": oracle_coverage - success_at_1,
            "request_rate": requested_events / denominator,
            "request_precision": (
                correct_requested_events / requested_events if requested_events else 0.0
            ),
        }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
