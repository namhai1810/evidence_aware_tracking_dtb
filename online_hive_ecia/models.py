"""Frozen HIVE, OSNet-AIN and Recovery-Aware ECIA online adapters."""

from __future__ import annotations

import json
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from build_hive_history_ranking_v2 import appearance_descriptor, select_candidate_pool
from hive_ecia.inference import recovery_decision, tensorize_event
from hive_ecia.recovery_aware import RecoveryAwareBaseState, build_from_checkpoint
from hive_ecia.reid_encoders import build_reid_encoder
from hive_hr_reid.run_experiment import clipped_crop
from hr_finding.datasets.preprocessing import image_to_tensor, temporal_metadata
from hr_finding.models.hive import HIVE

from .types import Observation, TriggerProposal


CODE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PreparedBaseEvent:
    """Cached LR evidence and frozen-Base state for one online event."""

    cache_key: tuple
    tensors: dict[str, torch.Tensor]
    base_state: RecoveryAwareBaseState
    competitor_ids: list[str | int]
    diagnostics: dict


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (CODE_ROOT / path).resolve()


def normalized_mean(values: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [np.asarray(value, dtype=np.float32) for value in values if value is not None]
    if not valid:
        return None
    result = np.mean(valid, axis=0)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-12 else None


class OnlineModelBundle:
    """Frozen online HIVE->one HR crop->ECIA inference bundle."""

    def __init__(self, config: dict, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.lowres_width, self.lowres_height = map(int, config["input"]["physical_lr_size"])
        self.crop_hw = tuple(map(int, config["hive"]["crop_size"]))

        hive_path = resolve_path(config["hive"]["checkpoint"])
        hive_checkpoint = torch.load(hive_path, map_location="cpu", weights_only=False)
        hive_config = hive_checkpoint["config"]
        self.hive = HIVE(hive_config["model"], metadata_dim=3).to(device)
        self.hive.load_state_dict(hive_checkpoint["model"], strict=True)
        self.hive.eval()
        for parameter in self.hive.parameters():
            parameter.requires_grad_(False)

        ecia_path = resolve_path(config["ecia"]["checkpoint"])
        ecia_checkpoint = torch.load(ecia_path, map_location="cpu", weights_only=False)
        self.ecia = build_from_checkpoint(ecia_checkpoint, device)
        reid_path = resolve_path(config["ecia"]["reid_checkpoint"])
        self.reid = build_reid_encoder(config["ecia"]["reid_encoder"], reid_path, device)

        thresholds_path = resolve_path(config["ecia"]["thresholds"])
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        selected = thresholds.get("selected_thresholds", thresholds)
        self.min_old_probability = float(
            selected.get("min_old_probability", config["ecia"]["min_old_probability"])
        )
        self.min_old_margin = float(
            selected.get("min_old_margin", config["ecia"]["min_old_margin"])
        )
        self.checkpoint_audit = {
            "hive": str(hive_path),
            "ecia": str(ecia_path),
            "reid": str(reid_path),
            "thresholds": str(thresholds_path),
            "min_old_probability": self.min_old_probability,
            "min_old_margin": self.min_old_margin,
        }
        # Per-run bounded caches.  One model instance owns both caches, so the
        # Base-guided path never reloads ECIA and never encodes one LR crop
        # twice while it remains in causal history.
        self.embedding_cache_limit = int(config.get("base_guided_trigger", {}).get("embedding_cache_size", 4096))
        self._lr_embedding_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._base_event_cache: OrderedDict[tuple, PreparedBaseEvent] = OrderedDict()
        self.cache_counters = defaultdict(int)

    @staticmethod
    def _observation_key(observation: Observation) -> tuple[str, int]:
        return observation.token.key(), int(observation.frame_id)

    def _cache_put(self, key: tuple[str, int], value: np.ndarray) -> None:
        self._lr_embedding_cache[key] = np.asarray(value, dtype=np.float32)
        self._lr_embedding_cache.move_to_end(key)
        while len(self._lr_embedding_cache) > self.embedding_cache_limit:
            self._lr_embedding_cache.popitem(last=False)

    def _observation_embeddings(self, observations: list[Observation]) -> list[np.ndarray | None]:
        """Return normalized frozen ReID embeddings with batched cache misses."""

        missing: list[Observation] = []
        seen_missing: set[tuple[str, int]] = set()
        for observation in observations:
            key = self._observation_key(observation)
            if key in self._lr_embedding_cache:
                self.cache_counters["lr_hits"] += 1
                self._lr_embedding_cache.move_to_end(key)
            elif key not in seen_missing:
                seen_missing.add(key)
                missing.append(observation)
        if missing:
            encoded = self.reid.encode([row.lr_crop for row in missing])
            self.cache_counters["lr_misses"] += len(missing)
            for observation, embedding in zip(missing, encoded):
                if embedding is None:
                    continue
                value = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(value))
                if norm > 1e-12:
                    value = value / norm
                self._cache_put(self._observation_key(observation), value)
        return [self._lr_embedding_cache.get(self._observation_key(row)) for row in observations]

    def prepare_base_event(
        self,
        *,
        proposal: TriggerProposal,
        target_observation: Observation,
        old_history: list[Observation],
        competitor_histories: list[tuple[str | int, list[Observation]]],
    ) -> PreparedBaseEvent | None:
        """Encode LR evidence and execute the frozen Base branch once."""

        old_rows = old_history[-int(self.config["history"]["old_samples"]) :]
        competitor_rows = [row for _, rows in competitor_histories for row in rows]
        all_rows = [*old_rows, target_observation, *competitor_rows]
        embeddings = self._observation_embeddings(all_rows)
        old_values = embeddings[: len(old_rows)]
        target_value = embeddings[len(old_rows)] if len(embeddings) > len(old_rows) else None
        cursor = len(old_rows) + 1
        competitor_ids: list[str | int] = []
        competitor_values: list[np.ndarray] = []
        for identity, rows in competitor_histories:
            values = embeddings[cursor : cursor + len(rows)]
            cursor += len(rows)
            value = normalized_mean(values)
            if value is not None:
                competitor_ids.append(identity)
                competitor_values.append(value)
        old_value = normalized_mean(old_values)
        if old_value is None or target_value is None:
            return None

        key = (
            int(proposal.frame_id),
            proposal.old.token.key(),
            proposal.new_token.key(),
            tuple(competitor_ids),
        )
        cached = self._base_event_cache.get(key)
        if cached is not None:
            self.cache_counters["base_hits"] += 1
            return cached

        dimension = int(old_value.size)
        event = {
            "old": old_value,
            "target": np.asarray(target_value, dtype=np.float32),
            "hr": np.zeros(dimension, dtype=np.float32),
            "hr_available": False,
            "competitors": (
                np.stack(competitor_values).astype(np.float32)
                if competitor_values
                else np.zeros((0, dimension), dtype=np.float32)
            ),
        }
        tensors = tensorize_event(event, self.device)
        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            base_state = self.ecia.forward_base(
                old=tensors["old"],
                target=tensors["target"],
                competitors=tensors["competitors"],
                competitor_mask=tensors["competitor_mask"],
            )
        self._synchronize()
        probabilities = torch.softmax(base_state.base_decision_logits.detach().float(), dim=-1)[0]
        identity_probabilities = self.ecia._masked_softmax(
            base_state.base_scores.detach().float(), base_state.hypothesis_mask
        )[0]
        decision_predicted_index = int(probabilities.argmax().item())
        identity_predicted_index = int(identity_probabilities.argmax().item())
        valid_identity_count = int(base_state.hypothesis_mask[0].sum().item())
        best_competitor_probability = (
            float(identity_probabilities[1:valid_identity_count].max().item())
            if valid_identity_count > 1
            else 0.0
        )
        prepared = PreparedBaseEvent(
            cache_key=key,
            tensors=tensors,
            base_state=base_state,
            competitor_ids=competitor_ids,
            diagnostics={
                "predicted_index": decision_predicted_index,
                "identity_predicted_index": identity_predicted_index,
                "old_probability": float(probabilities[0].item()),
                "old_margin": float(
                    probabilities[0].item() - probabilities[1:].max().item()
                ),
                "old_competitor_margin": float(
                    identity_probabilities[0].item() - best_competitor_probability
                ),
                "top_probability": float(probabilities.max().item()),
                "abstain_probability": float(probabilities[-1].item()),
                "normalized_entropy": float(base_state.normalized_entropy[0].item()),
                "base_identity_probabilities": identity_probabilities.cpu().tolist(),
                "decision_probabilities": probabilities.cpu().tolist(),
                "competitor_ids": competitor_ids,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            },
        )
        self._base_event_cache[key] = prepared
        self.cache_counters["base_misses"] += 1
        # Base states retain GPU tensors; only same-frame reuse is required.
        while len(self._base_event_cache) > int(self.config.get("base_guided_trigger", {}).get("base_event_cache_size", 32)):
            self._base_event_cache.popitem(last=False)
        return prepared

    def decide_from_prepared_base(self, prepared: PreparedBaseEvent) -> dict:
        decision = recovery_decision(
            prepared.base_state.base_decision_logits,
            min_old_probability=self.min_old_probability,
            min_old_margin=self.min_old_margin,
        )
        decision.update({"gate": 0.0, "competitor_canonical_ids": prepared.competitor_ids})
        return decision

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _hive_candidates(
        self,
        *,
        proposal: TriggerProposal,
        target_crop: np.ndarray,
        old_history: list[Observation],
        global_history: list[Observation],
    ) -> list[dict]:
        hive_config = self.config["hive"]
        history_config = self.config["history"]
        start = max(1, proposal.frame_id - int(history_config["max_history_frames"]))
        old_descriptors = [
            row.descriptor if row.descriptor is not None else appearance_descriptor(row.lr_crop)
            for row in old_history[-int(history_config["old_samples"]) :]
        ]
        if not old_descriptors:
            return []
        old_descriptor = np.mean(old_descriptors, axis=0)
        old_descriptor /= max(float(np.linalg.norm(old_descriptor)), 1e-12)
        target_descriptor = appearance_descriptor(target_crop)

        by_frame: dict[int, list[dict]] = defaultdict(list)
        for observation in global_history:
            if not start <= observation.frame_id < proposal.frame_id:
                continue
            descriptor = (
                observation.descriptor
                if observation.descriptor is not None
                else appearance_descriptor(observation.lr_crop)
            )
            old_similarity = float(np.dot(old_descriptor, descriptor))
            target_similarity = float(np.dot(target_descriptor, descriptor))
            by_frame[observation.frame_id].append(
                {
                    "candidate_frame": observation.frame_id,
                    "lr_pred_id": observation.token.raw_id,
                    "lr_confidence": observation.confidence,
                    "lr_bridge_similarity": min(old_similarity, target_similarity),
                    "lr_similarity_to_old": old_similarity,
                    "lr_similarity_to_target": target_similarity,
                    "observation": observation,
                }
            )
        frame_best = [
            max(rows, key=lambda row: (row["lr_bridge_similarity"], row["lr_confidence"]))
            for _, rows in sorted(by_frame.items())
            if rows
        ]
        args = SimpleNamespace(
            appearance_candidates=int(hive_config["appearance_candidates"]),
            recent_candidates=int(hive_config["recent_candidates"]),
            diverse_candidates=int(hive_config["diverse_candidates"]),
            max_candidates=int(hive_config["max_candidates"]),
            min_spacing_frames=int(hive_config["min_spacing_frames"]),
        )
        return select_candidate_pool(frame_best, args)

    def select_historical_observation(
        self,
        *,
        proposal: TriggerProposal,
        target_crop: np.ndarray,
        old_history: list[Observation],
        new_history: list[Observation],
        global_history: list[Observation],
    ) -> tuple[Observation | None, dict]:
        self._synchronize()
        started = time.perf_counter()
        candidates = self._hive_candidates(
            proposal=proposal,
            target_crop=target_crop,
            old_history=old_history,
            global_history=global_history,
        )
        max_history = int(self.config["history"]["old_samples"])
        max_candidates = int(self.config["hive"]["max_candidates"])
        old_tensor = torch.zeros(
            (1, max_history, 3, *self.crop_hw), dtype=torch.float32, device=self.device
        )
        old_mask = torch.zeros((1, max_history), dtype=torch.bool, device=self.device)
        for slot, observation in enumerate(old_history[-max_history:]):
            old_tensor[0, slot] = image_to_tensor(observation.lr_crop, self.crop_hw).to(self.device)
            old_mask[0, slot] = True

        candidate_tensor = torch.zeros(
            (1, max_candidates, 3, *self.crop_hw), dtype=torch.float32, device=self.device
        )
        candidate_mask = torch.zeros((1, max_candidates), dtype=torch.bool, device=self.device)
        metadata = torch.zeros((1, max_candidates, 3), dtype=torch.float32, device=self.device)
        for slot, candidate in enumerate(candidates[:max_candidates]):
            observation = candidate["observation"]
            candidate_tensor[0, slot] = image_to_tensor(observation.lr_crop, self.crop_hw).to(
                self.device
            )
            candidate_mask[0, slot] = True
            seconds = (proposal.frame_id - observation.frame_id) / float(
                self.config["input"]["fps"]
            )
            metadata[0, slot] = temporal_metadata(seconds).to(self.device)

        competitor_tensor = torch.zeros_like(old_tensor)
        competitor_mask = torch.zeros_like(old_mask)
        for slot, observation in enumerate(new_history[-max_history:]):
            competitor_tensor[0, slot] = image_to_tensor(
                observation.lr_crop, self.crop_hw
            ).to(self.device)
            competitor_mask[0, slot] = True
        target_tensor = image_to_tensor(target_crop, self.crop_hw).unsqueeze(0).to(self.device)

        if not bool(old_mask.any()) or not bool(candidate_mask.any()):
            return None, {
                "candidate_count": len(candidates),
                "reason": "missing_hive_history_or_candidates",
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        batch = {
            "old_crops": old_tensor,
            "old_mask": old_mask,
            "target_crop": target_tensor,
            "candidate_crops": candidate_tensor,
            "candidate_mask": candidate_mask,
            "metadata": metadata,
            "competitor_crops": competitor_tensor,
            "competitor_mask": competitor_mask,
        }
        with torch.inference_mode():
            logits = self.hive(batch)[0]
            selected_index = int(logits.argmax().item())
        self._synchronize()
        selected = candidates[selected_index]["observation"]
        return selected, {
            "candidate_count": len(candidates),
            "selected_index": selected_index,
            "selected_frame": selected.frame_id,
            "selected_raw_id": selected.token.raw_id,
            "selected_score": float(logits[selected_index].item()),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    def recover_prepared_with_hr(
        self,
        *,
        prepared: PreparedBaseEvent,
        proposal: TriggerProposal,
        target_crop: np.ndarray,
        old_history: list[Observation],
        new_history: list[Observation],
        global_history: list[Observation],
    ) -> dict:
        """Request one HIVE-selected HR crop and reuse the cached Base state."""

        selected, hive_diagnostics = self.select_historical_observation(
            proposal=proposal,
            target_crop=target_crop,
            old_history=old_history,
            new_history=new_history,
            global_history=global_history,
        )
        if selected is None:
            return {
                **self.decide_from_prepared_base(prepared),
                "merge": False,
                "reason": hive_diagnostics["reason"],
                "hive": hive_diagnostics,
                "timing": {
                    "base_ms": prepared.diagnostics["latency_ms"],
                    "hive_ms": hive_diagnostics["latency_ms"],
                    "hr_ms": 0.0,
                    "ecia_ms": 0.0,
                },
                "hr": {"requested": True, "crop_count": 0, "bytes_read": 0, "crop_pixels": 0},
            }

        hr_started = time.perf_counter()
        image = cv2.imread(str(selected.source_path), cv2.IMREAD_COLOR)
        hr_crop = clipped_crop(image, selected.bbox_hr) if image is not None else None
        try:
            bytes_read = selected.source_path.stat().st_size
        except OSError:
            bytes_read = 0
        hr_ms = (time.perf_counter() - hr_started) * 1000.0
        if hr_crop is None or not hr_crop.size:
            return {
                **self.decide_from_prepared_base(prepared),
                "merge": False,
                "reason": "hr_retrieval_failed",
                "hive": hive_diagnostics,
                "timing": {
                    "base_ms": prepared.diagnostics["latency_ms"],
                    "hive_ms": hive_diagnostics["latency_ms"],
                    "hr_ms": hr_ms,
                    "ecia_ms": 0.0,
                },
                "hr": {"requested": True, "crop_count": 0, "bytes_read": bytes_read, "crop_pixels": 0},
            }

        self._synchronize()
        ecia_started = time.perf_counter()
        encoded = self.reid.encode([hr_crop])
        hr_embedding = encoded[0] if encoded else None
        if hr_embedding is None:
            return {
                **self.decide_from_prepared_base(prepared),
                "merge": False,
                "reason": "missing_hr_embedding",
                "hive": hive_diagnostics,
                "timing": {
                    "base_ms": prepared.diagnostics["latency_ms"],
                    "hive_ms": hive_diagnostics["latency_ms"],
                    "hr_ms": hr_ms,
                    "ecia_ms": (time.perf_counter() - ecia_started) * 1000.0,
                },
                "hr": {"requested": True, "crop_count": 1, "bytes_read": bytes_read, "crop_pixels": int(hr_crop.shape[0] * hr_crop.shape[1])},
            }
        hr_value = np.asarray(hr_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(hr_value))
        if norm > 1e-12:
            hr_value /= norm
        hr_tensor = torch.from_numpy(hr_value).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.ecia.forward_from_base(
                prepared.base_state,
                hr=hr_tensor,
                hr_available=torch.ones(1, dtype=torch.bool, device=self.device),
            )
        self._synchronize()
        decision = recovery_decision(
            output.decision_logits,
            min_old_probability=self.min_old_probability,
            min_old_margin=self.min_old_margin,
        )
        decision.update(
            {
                "hive": hive_diagnostics,
                "gate": float(output.gate.detach().cpu().reshape(-1)[0]),
                "competitor_canonical_ids": prepared.competitor_ids,
                "base": prepared.diagnostics,
                "timing": {
                    "base_ms": prepared.diagnostics["latency_ms"],
                    "hive_ms": hive_diagnostics["latency_ms"],
                    "hr_ms": hr_ms,
                    "ecia_ms": (time.perf_counter() - ecia_started) * 1000.0,
                },
                "hr": {
                    "requested": True,
                    "crop_count": 1,
                    "bytes_read": bytes_read,
                    "crop_pixels": int(hr_crop.shape[0] * hr_crop.shape[1]),
                },
            }
        )
        return decision

    def recover(
        self,
        *,
        proposal: TriggerProposal,
        target_crop: np.ndarray,
        old_history: list[Observation],
        new_history: list[Observation],
        global_history: list[Observation],
        competitor_histories: list[tuple[int, list[Observation]]],
        hr_enabled: bool,
    ) -> dict:
        selected, hive_diagnostics = self.select_historical_observation(
            proposal=proposal,
            target_crop=target_crop,
            old_history=old_history,
            new_history=new_history,
            global_history=global_history,
        )
        if selected is None:
            return {
                "merge": False,
                "reason": hive_diagnostics["reason"],
                "hive": hive_diagnostics,
                "timing": {"hive_ms": hive_diagnostics["latency_ms"], "hr_ms": 0.0, "ecia_ms": 0.0},
                "hr": {"requested": False, "crop_count": 0, "bytes_read": 0, "crop_pixels": 0},
            }

        hr_started = time.perf_counter()
        hr_crop = None
        bytes_read = 0
        if hr_enabled:
            image = cv2.imread(str(selected.source_path), cv2.IMREAD_COLOR)
            if image is not None:
                hr_crop = clipped_crop(image, selected.bbox_hr)
                try:
                    bytes_read = selected.source_path.stat().st_size
                except OSError:
                    bytes_read = 0
        hr_ms = (time.perf_counter() - hr_started) * 1000.0
        if hr_enabled and (hr_crop is None or not hr_crop.size):
            return {
                "merge": False,
                "reason": "hr_retrieval_failed",
                "hive": hive_diagnostics,
                "timing": {"hive_ms": hive_diagnostics["latency_ms"], "hr_ms": hr_ms, "ecia_ms": 0.0},
                "hr": {"requested": True, "crop_count": 0, "bytes_read": bytes_read, "crop_pixels": 0},
            }

        self._synchronize()
        ecia_started = time.perf_counter()
        old_crops = [row.lr_crop for row in old_history[-int(self.config["history"]["old_samples"]) :]]
        crops = [*old_crops, target_crop]
        target_index = len(crops) - 1
        hr_index = None
        if hr_crop is not None:
            hr_index = len(crops)
            crops.append(hr_crop)
        competitor_groups = []
        for canonical_id, observations in competitor_histories:
            indices = []
            for observation in observations:
                indices.append(len(crops))
                crops.append(observation.lr_crop)
            competitor_groups.append((canonical_id, indices))
        encoded = self.reid.encode(crops)
        old_embedding = normalized_mean(encoded[: len(old_crops)])
        target_embedding = encoded[target_index]
        hr_embedding = encoded[hr_index] if hr_index is not None else None
        competitor_ids = []
        competitor_embeddings = []
        for canonical_id, indices in competitor_groups:
            embedding = normalized_mean([encoded[index] for index in indices])
            if embedding is not None:
                competitor_ids.append(canonical_id)
                competitor_embeddings.append(embedding)
        if old_embedding is None or target_embedding is None:
            return {
                "merge": False,
                "reason": "missing_ecia_old_or_target_embedding",
                "hive": hive_diagnostics,
                "timing": {"hive_ms": hive_diagnostics["latency_ms"], "hr_ms": hr_ms, "ecia_ms": (time.perf_counter() - ecia_started) * 1000.0},
                "hr": {"requested": hr_enabled, "crop_count": int(hr_crop is not None), "bytes_read": bytes_read, "crop_pixels": int(hr_crop.shape[0] * hr_crop.shape[1]) if hr_crop is not None else 0},
            }
        embedding_dim = int(old_embedding.size)
        embeddings = {
            "old": old_embedding,
            "target": np.asarray(target_embedding, dtype=np.float32),
            "hr": np.asarray(hr_embedding, dtype=np.float32) if hr_embedding is not None else np.zeros(embedding_dim, dtype=np.float32),
            "hr_available": hr_embedding is not None,
            "competitors": np.stack(competitor_embeddings).astype(np.float32) if competitor_embeddings else np.zeros((0, embedding_dim), dtype=np.float32),
        }
        tensors = tensorize_event(embeddings, self.device)
        with torch.inference_mode():
            output = self.ecia(
                **tensors,
                ablation="full",
                gate_override=None if hr_enabled else 0.0,
            )
        self._synchronize()
        decision = recovery_decision(
            output.decision_logits,
            min_old_probability=self.min_old_probability,
            min_old_margin=self.min_old_margin,
        )
        decision.update(
            {
                "hive": hive_diagnostics,
                "gate": float(output.gate.detach().cpu().reshape(-1)[0]),
                "competitor_canonical_ids": competitor_ids,
                "timing": {
                    "hive_ms": hive_diagnostics["latency_ms"],
                    "hr_ms": hr_ms,
                    "ecia_ms": (time.perf_counter() - ecia_started) * 1000.0,
                },
                "hr": {
                    "requested": hr_enabled,
                    "crop_count": int(hr_crop is not None),
                    "bytes_read": bytes_read,
                    "crop_pixels": int(hr_crop.shape[0] * hr_crop.shape[1]) if hr_crop is not None else 0,
                },
            }
        )
        return decision

