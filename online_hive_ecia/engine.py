"""Causal online identity-recovery controller around an unchanged BoT-SORT."""

from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from build_hive_history_ranking_v2 import appearance_descriptor
from hive_hr_reid.run_experiment import clipped_crop

from .base_trigger import BaseGuidedCandidateProposer, BaseGuidedHRPolicy
from .identity import CanonicalIdentityRegistry
from .models import OnlineModelBundle
from .trigger import AmbiguityTrigger
from .types import LostTrack, Observation, TrackToken, TrackerSignal


class OnlineRecoveryEngine:
    """Maintain bounded causal history and apply confirmed identity aliases."""

    def __init__(
        self,
        config: dict,
        *,
        mode: str,
        model_bundle: OnlineModelBundle | None,
    ) -> None:
        if mode not in {
            "baseline", "online", "hr_disabled",
            "base_guided_hr_disabled", "base_guided_online",
            "base_guided_safety_online",
            "base_guided_gating_off_safety_online",
        }:
            raise ValueError(f"Unsupported online mode: {mode}")
        if mode != "baseline" and model_bundle is None:
            raise ValueError("online modes require a frozen model bundle")
        self.config = config
        self.mode = mode
        self.model_bundle = model_bundle
        self.trigger = AmbiguityTrigger(config["trigger"])
        self.base_proposer = BaseGuidedCandidateProposer(config["base_guided_trigger"])
        self.base_policy = BaseGuidedHRPolicy(config["base_guided_trigger"]["policy"])
        self.registry = CanonicalIdentityRegistry(
            raw_reuse_gap=int(config["history"]["raw_reuse_gap"])
        )
        self.history_by_token: dict[TrackToken, deque[Observation]] = defaultdict(
            lambda: deque(maxlen=int(config["history"]["max_observations_per_identity"]))
        )
        self.global_history: deque[Observation] = deque(
            maxlen=int(config["history"]["max_global_observations"])
        )
        self.active_by_raw: dict[int, TrackToken] = {}
        self.lost: dict[TrackToken, LostTrack] = {}
        self.observation_counts: Counter[TrackToken] = Counter()
        self.recovered_tokens: set[TrackToken] = set()
        self.cooldown: dict[tuple[TrackToken, TrackToken], int] = {}
        self.counters = Counter()
        self.latency_ms = defaultdict(float)
        self.hr_cost = Counter()

    @staticmethod
    def _feature(signal: TrackerSignal | None) -> np.ndarray | None:
        if signal is None:
            return None
        return (
            signal.smooth_feature
            if signal.smooth_feature is not None
            else signal.current_feature
        )

    @staticmethod
    def _scale_box(
        box_lr: np.ndarray,
        lr_shape: tuple[int, int],
        hr_shape: tuple[int, int],
    ) -> np.ndarray:
        lr_height, lr_width = lr_shape
        hr_height, hr_width = hr_shape
        result = np.asarray(box_lr, dtype=np.float32).copy()
        result[[0, 2]] *= hr_width / max(lr_width, 1)
        result[[1, 3]] *= hr_height / max(lr_height, 1)
        return result

    def _make_observation(
        self,
        *,
        frame_id: int,
        token: TrackToken,
        row: np.ndarray,
        lr_frame: np.ndarray,
        hr_frame: np.ndarray,
        source_path: Path,
        signal: TrackerSignal | None,
    ) -> Observation:
        bbox_lr = np.asarray(row[:4], dtype=np.float32)
        lr_crop = clipped_crop(lr_frame, bbox_lr)
        if not lr_crop.size:
            raise ValueError(f"Empty LR crop for raw ID {token.raw_id} at frame {frame_id}")
        bbox_hr = self._scale_box(bbox_lr, lr_frame.shape[:2], hr_frame.shape[:2])
        return Observation(
            frame_id=frame_id,
            token=token,
            canonical_id=self.registry.canonical(token),
            bbox_lr=bbox_lr,
            bbox_hr=bbox_hr,
            confidence=float(row[5]),
            lr_crop=lr_crop,
            source_path=source_path.resolve(),
            native_feature=self._feature(signal),
            descriptor=appearance_descriptor(lr_crop),
        )

    def _competitors(
        self,
        *,
        frame_id: int,
        old_token: TrackToken,
        new_token: TrackToken,
    ) -> list[tuple[int, list[Observation]]]:
        history_config = self.config["history"]
        earliest = frame_id - int(history_config["competitor_history_frames"])
        groups: dict[int, list[Observation]] = defaultdict(list)
        for token, rows in self.history_by_token.items():
            if token in {old_token, new_token}:
                continue
            # Bounded histories can legitimately have no observation inside
            # the competitor time window.  Do not materialize an empty
            # canonical group: it has no prototype to encode and cannot be
            # ordered by a most-recent frame.
            recent_rows = [row for row in rows if row.frame_id >= earliest]
            if not recent_rows:
                continue
            canonical = self.registry.canonical(token)
            groups[canonical].extend(recent_rows)
        ordered = sorted(
            groups.items(),
            key=lambda item: (max(row.frame_id for row in item[1]), len(item[1])),
            reverse=True,
        )[: int(history_config["max_competitors"])]
        sample_count = int(history_config["competitor_samples"])
        return [
            (canonical, sorted(rows, key=lambda row: row.frame_id)[-sample_count:])
            for canonical, rows in ordered
        ]

    def _token_competitors(
        self,
        *,
        frame_id: int,
        old_token: TrackToken,
        new_token: TrackToken,
        current_observations: list[Observation],
    ) -> list[tuple[str, list[Observation]]]:
        """Stable raw-token groups unaffected by online canonical merges.

        This makes Base-guided HR-on/off proposal and HR-request schedules
        pairable even when the Full branch changes canonical identities.
        """

        history_config = self.config["history"]
        earliest = frame_id - int(history_config["competitor_history_frames"])
        current = {row.token: row for row in current_observations}
        groups: list[tuple[TrackToken, list[Observation]]] = []
        for token, history in self.history_by_token.items():
            if token in {old_token, new_token}:
                continue
            rows = [row for row in history if row.frame_id >= earliest]
            if token in current:
                rows.append(current[token])
            if rows:
                groups.append((token, rows))
        # Include established active tokens which do not yet have a retained
        # history entry (possible immediately after a generation change).
        known = {token for token, _ in groups}
        for token, row in current.items():
            if token not in {old_token, new_token} and token not in known:
                groups.append((token, [row]))
        groups.sort(
            key=lambda item: (max(row.frame_id for row in item[1]), len(item[1]), item[0].key()),
            reverse=True,
        )
        sample_count = int(history_config["competitor_samples"])
        return [
            (token.key(), sorted(rows, key=lambda row: row.frame_id)[-sample_count:])
            for token, rows in groups[: int(history_config["max_competitors"])]
        ]

    def _update_lifecycle(
        self,
        *,
        frame_id: int,
        current_by_raw: dict[int, TrackToken],
        lost_signals: dict[int, TrackerSignal],
    ) -> None:
        for raw_id, token in self.active_by_raw.items():
            if raw_id in current_by_raw:
                continue
            history = self.history_by_token.get(token)
            if not history:
                continue
            signal = lost_signals.get(raw_id)
            lost_feature = self._feature(signal)
            self.lost[token] = LostTrack(
                token=token,
                canonical_id=self.registry.canonical(token),
                lost_frame=frame_id,
                last_observation=history[-1],
                predicted_bbox_lr=signal.bbox_lr if signal is not None else None,
                native_feature=(
                    lost_feature
                    if lost_feature is not None
                    else history[-1].native_feature
                ),
            )
        for token in current_by_raw.values():
            self.lost.pop(token, None)
        # BoT-SORT keeps predicting lost Kalman states. Refresh those causal
        # motion/appearance signals before scoring a Lost->New hypothesis.
        for token, lost in self.lost.items():
            signal = lost_signals.get(token.raw_id)
            if signal is None:
                continue
            lost.predicted_bbox_lr = signal.bbox_lr
            feature = self._feature(signal)
            if feature is not None:
                lost.native_feature = feature
        max_age = int(self.config["trigger"]["max_lost_age"])
        self.lost = {
            token: row
            for token, row in self.lost.items()
            if frame_id - row.lost_frame <= max_age
        }

    def process_frame(
        self,
        *,
        frame_id: int,
        source_path: Path,
        lr_frame: np.ndarray,
        hr_frame: np.ndarray,
        raw_tracks: np.ndarray,
        active_signals: dict[int, TrackerSignal],
        lost_signals: dict[int, TrackerSignal],
    ) -> tuple[np.ndarray, list[dict]]:
        """Return current-frame canonical tracks and JSON-safe diagnostics."""

        started = time.perf_counter()
        rows = np.asarray(raw_tracks, dtype=float).reshape(-1, 6)
        # The baseline is the unmodified BoT-SORT output. Do not pay history,
        # crop, trigger, HIVE, ReID, or ECIA overhead in this branch.
        if self.mode == "baseline":
            self.counters["frames"] += 1
            self.counters["raw_tracks"] += len(rows)
            self.latency_ms["controller_ms"] += (time.perf_counter() - started) * 1000.0
            return rows.copy(), []
        tokens = []
        reused_tokens = []
        current_by_raw = {}
        for row in rows:
            raw_id = int(row[4])
            token, created = self.registry.token_for(raw_id, frame_id)
            tokens.append(token)
            if created and token.generation > 0:
                reused_tokens.append(token)
            current_by_raw[raw_id] = token

        self._update_lifecycle(
            frame_id=frame_id,
            current_by_raw=current_by_raw,
            lost_signals=lost_signals,
        )
        self.registry.begin_frame(set(tokens))

        current_observations = []
        valid_rows = []
        diagnostics = [
            {
                "type": "raw_id_reuse",
                "frame": frame_id,
                "raw_id": token.raw_id,
                "generation": token.generation,
                "canonical_id": self.registry.canonical(token),
            }
            for token in reused_tokens
        ]
        for row, token in zip(rows, tokens):
            try:
                observation = self._make_observation(
                    frame_id=frame_id,
                    token=token,
                    row=row,
                    lr_frame=lr_frame,
                    hr_frame=hr_frame,
                    source_path=source_path,
                    signal=active_signals.get(token.raw_id),
                )
            except ValueError as error:
                diagnostics.append(
                    {
                        "type": "observation_rejected",
                        "frame": frame_id,
                        "raw_id": token.raw_id,
                        "reason": str(error),
                    }
                )
                continue
            valid_rows.append(row)
            current_observations.append(observation)
        rows = np.asarray(valid_rows, dtype=float).reshape(-1, 6)
        tokens = [row.token for row in current_observations]

        # Original rule-based implementation is intentionally unchanged and
        # remains a selectable baseline.
        if self.mode in {"online", "hr_disabled"}:
            event_limit = int(self.config["trigger"]["max_events_per_frame"])
            attempted = 0
            for observation in current_observations:
                token = observation.token
                if token in self.recovered_tokens:
                    continue
                new_age = self.observation_counts[token] + 1
                if new_age > int(self.config["trigger"]["max_new_age"]):
                    continue
                proposal = self.trigger.propose(
                    frame_id=frame_id,
                    new_token=token,
                    new_bbox_lr=observation.bbox_lr,
                    new_feature=observation.native_feature,
                    new_age=new_age,
                    lost_tracks=list(self.lost.values()),
                )
                if proposal is None:
                    continue
                cooldown_key = (proposal.old.token, proposal.new_token)
                last_attempt = self.cooldown.get(cooldown_key, -10**9)
                if frame_id - last_attempt < int(self.config["trigger"]["cooldown_frames"]):
                    continue
                self.cooldown[cooldown_key] = frame_id
                attempted += 1
                self.counters["triggers"] += 1
                assert self.model_bundle is not None
                decision = self.model_bundle.recover(
                    proposal=proposal,
                    target_crop=observation.lr_crop,
                    old_history=list(self.history_by_token[proposal.old.token]),
                    new_history=list(self.history_by_token[proposal.new_token]),
                    global_history=list(self.global_history),
                    competitor_histories=self._competitors(
                        frame_id=frame_id,
                        old_token=proposal.old.token,
                        new_token=proposal.new_token,
                    ),
                    hr_enabled=self.mode == "online",
                )
                timing = decision.get("timing", {})
                for key in ("hive_ms", "hr_ms", "ecia_ms"):
                    self.latency_ms[key] += float(timing.get(key, 0.0))
                hr = decision.get("hr", {})
                self.hr_cost["requests"] += int(bool(hr.get("requested")))
                self.hr_cost["crops"] += int(hr.get("crop_count", 0))
                self.hr_cost["bytes_read"] += int(hr.get("bytes_read", 0))
                self.hr_cost["crop_pixels"] += int(hr.get("crop_pixels", 0))

                merge_result = None
                if decision.get("merge"):
                    merge_result = self.registry.merge_successor(
                        old_token=proposal.old.token,
                        new_token=proposal.new_token,
                        new_observation_count=new_age,
                        max_new_observations=int(
                            self.config["history"]["max_new_observations_for_merge"]
                        ),
                    )
                    if merge_result.applied:
                        self.counters["merges_applied"] += 1
                        self.recovered_tokens.add(proposal.new_token)
                    else:
                        self.counters["merges_rejected_by_safety"] += 1
                else:
                    self.counters["abstains"] += 1
                diagnostics.append(
                    {
                        "type": "recovery_event",
                        "frame": frame_id,
                        "old_token": proposal.old.token.key(),
                        "new_token": proposal.new_token.key(),
                        "old_canonical_id": proposal.old.canonical_id,
                        "trigger": self.trigger.diagnostics(proposal),
                        "decision": decision,
                        "safety": (
                            {
                                "applied": merge_result.applied,
                                "reason": merge_result.reason,
                                "canonical_id": merge_result.canonical_id,
                            }
                            if merge_result is not None
                            else {"applied": False, "reason": "ecia_abstain"}
                        ),
                    }
                )
                if attempted >= event_limit:
                    break

        if self.mode in {
            "base_guided_hr_disabled",
            "base_guided_online",
            "base_guided_safety_online",
            "base_guided_gating_off_safety_online",
        }:
            event_limit = int(self.config["base_guided_trigger"]["candidate"]["max_events_per_frame"])
            attempted = 0
            for observation in current_observations:
                token = observation.token
                new_age = self.observation_counts[token] + 1
                proposal = self.base_proposer.propose_candidates(
                    frame_id=frame_id,
                    new_observation=observation,
                    new_age=new_age,
                    lost_tracks=list(self.lost.values()),
                    active_observations=current_observations,
                    observation_counts=self.observation_counts,
                )
                if proposal is None:
                    continue
                cooldown_key = (proposal.old.token, proposal.new_token)
                last_attempt = self.cooldown.get(cooldown_key, -10**9)
                cooldown_frames = int(self.config["base_guided_trigger"]["candidate"]["cooldown_frames"])
                if frame_id - last_attempt < cooldown_frames:
                    continue
                self.cooldown[cooldown_key] = frame_id
                attempted += 1
                self.counters["base_proposed_events"] += 1
                assert self.model_bundle is not None

                old_history = list(self.history_by_token[proposal.old.token])
                if proposal.pair_kind == "active_to_active" and (
                    not old_history
                    or old_history[-1].frame_id != proposal.old.last_observation.frame_id
                ):
                    old_history.append(proposal.old.last_observation)
                prepared = self.model_bundle.prepare_base_event(
                    proposal=proposal,
                    target_observation=observation,
                    old_history=old_history,
                    competitor_histories=self._token_competitors(
                        frame_id=frame_id,
                        old_token=proposal.old.token,
                        new_token=proposal.new_token,
                        current_observations=current_observations,
                    ),
                )
                if prepared is None:
                    diagnostics.append(
                        {
                            "type": "base_guided_event",
                            "frame": frame_id,
                            "old_token": proposal.old.token.key(),
                            "new_token": proposal.new_token.key(),
                            "pair_kind": proposal.pair_kind,
                            "trigger": self.base_proposer.diagnostics(proposal),
                            "action": "abstain",
                            "would_request_hr": False,
                            "decision": {"merge": False, "reason": "missing_base_evidence"},
                            "safety": {"applied": False, "reason": "missing_base_evidence"},
                        }
                    )
                    if attempted >= event_limit:
                        break
                    continue

                policy = self.base_policy.decide(prepared.diagnostics)
                self.latency_ms["base_ms"] += float(prepared.diagnostics["latency_ms"])
                gating_off = self.mode == "base_guided_gating_off_safety_online"
                # G0 changes only the pre-query gate: every candidate with
                # valid Base evidence becomes an HR-query candidate. The
                # original Base decision is still logged for paired auditing.
                effective_request_hr = True if gating_off else policy.request_hr
                if policy.request_hr:
                    self.counters["base_policy_hr_requests"] += 1
                if gating_off and not policy.request_hr:
                    self.counters["gating_off_added_requests"] += 1
                if effective_request_hr:
                    self.counters["triggers"] += 1
                    self.counters["base_hr_requests"] += 1

                pre_query_safety = None
                if effective_request_hr and self.mode in {
                    "base_guided_safety_online",
                    "base_guided_gating_off_safety_online",
                }:
                    pre_query_safety = self.registry.precheck_successor(
                        old_token=proposal.old.token,
                        new_token=proposal.new_token,
                        new_observation_count=new_age,
                        max_new_observations=int(
                            self.config["history"]["max_new_observations_for_merge"]
                        ),
                    )

                if (
                    effective_request_hr
                    and self.mode in {
                        "base_guided_safety_online",
                        "base_guided_gating_off_safety_online",
                    }
                    and pre_query_safety is not None
                    and not pre_query_safety.applied
                ):
                    # Full ECIA could never produce an applicable merge under
                    # the current registry state.  Preserve the no-merge
                    # outcome while avoiding HIVE, image I/O and HR ReID.
                    self.counters["prequery_safety_skips"] += 1
                    self.counters[
                        f"prequery_skip_{pre_query_safety.reason}"
                    ] += 1
                    decision = self.model_bundle.decide_from_prepared_base(prepared)
                    decision.update(
                        {
                            "merge": False,
                            "reason": f"prequery_safety_reject:{pre_query_safety.reason}",
                            "base": prepared.diagnostics,
                            "timing": {
                                "base_ms": prepared.diagnostics["latency_ms"],
                                "hive_ms": 0.0,
                                "hr_ms": 0.0,
                                "ecia_ms": 0.0,
                            },
                            "hr": {
                                "requested": False,
                                "would_request": True,
                                "skipped_by_prequery": True,
                                "crop_count": 0,
                                "bytes_read": 0,
                                "crop_pixels": 0,
                            },
                        }
                    )
                elif effective_request_hr and self.mode in {
                    "base_guided_online",
                    "base_guided_safety_online",
                    "base_guided_gating_off_safety_online",
                }:
                    if self.mode in {
                        "base_guided_safety_online",
                        "base_guided_gating_off_safety_online",
                    }:
                        self.counters["prequery_feasible_requests"] += 1
                    decision = self.model_bundle.recover_prepared_with_hr(
                        prepared=prepared,
                        proposal=proposal,
                        target_crop=observation.lr_crop,
                        old_history=old_history,
                        new_history=list(self.history_by_token[proposal.new_token]),
                        global_history=list(self.global_history),
                    )
                else:
                    # HR-disabled logs the exact same would-request decision;
                    # gate=0 is represented by the already-computed Base logits.
                    decision = self.model_bundle.decide_from_prepared_base(prepared)
                    decision.update(
                        {
                            "base": prepared.diagnostics,
                            "timing": {
                                "base_ms": prepared.diagnostics["latency_ms"],
                                "hive_ms": 0.0,
                                "hr_ms": 0.0,
                                "ecia_ms": 0.0,
                            },
                            "hr": {
                                "requested": False,
                                "crop_count": 0,
                                "bytes_read": 0,
                                "crop_pixels": 0,
                            },
                        }
                    )
                timing = decision.get("timing", {})
                for key in ("hive_ms", "hr_ms", "ecia_ms"):
                    self.latency_ms[key] += float(timing.get(key, 0.0))
                hr = decision.get("hr", {})
                self.hr_cost["requests"] += int(bool(hr.get("requested")))
                self.hr_cost["crops"] += int(hr.get("crop_count", 0))
                self.hr_cost["bytes_read"] += int(hr.get("bytes_read", 0))
                self.hr_cost["crop_pixels"] += int(hr.get("crop_pixels", 0))

                merge_result = None
                if decision.get("merge"):
                    merge_result = self.registry.merge_successor(
                        old_token=proposal.old.token,
                        new_token=proposal.new_token,
                        new_observation_count=new_age,
                        max_new_observations=int(
                            self.config["history"]["max_new_observations_for_merge"]
                        ),
                    )
                    if merge_result.applied:
                        self.counters["merges_applied"] += 1
                        self.recovered_tokens.add(proposal.new_token)
                    else:
                        self.counters["merges_rejected_by_safety"] += 1
                else:
                    self.counters["abstains"] += 1
                diagnostics.append(
                    {
                        "type": "base_guided_event",
                        "frame": frame_id,
                        "old_token": proposal.old.token.key(),
                        "new_token": proposal.new_token.key(),
                        "old_canonical_id": proposal.old.canonical_id,
                        "pair_kind": proposal.pair_kind,
                        "trigger": self.base_proposer.diagnostics(proposal),
                        "base": prepared.diagnostics,
                        "policy": BaseGuidedHRPolicy.diagnostics(policy),
                        "gating": {
                            "enabled": not gating_off,
                            "base_policy_would_request_hr": policy.request_hr,
                            "effective_request_hr": effective_request_hr,
                        },
                        "pre_query_safety": (
                            {
                                "checked": True,
                                "feasible": pre_query_safety.applied,
                                "reason": pre_query_safety.reason,
                                "canonical_id": pre_query_safety.canonical_id,
                            }
                            if pre_query_safety is not None
                            else {"checked": False, "reason": "not_enabled_or_not_required"}
                        ),
                        "action": "request_hr_gating_off" if gating_off else policy.action,
                        "would_request_hr": effective_request_hr,
                        "decision": decision,
                        "safety": (
                            {
                                "applied": merge_result.applied,
                                "reason": merge_result.reason,
                                "canonical_id": merge_result.canonical_id,
                            }
                            if merge_result is not None
                            else {
                                "applied": False,
                                "reason": (
                                    f"prequery_safety_reject:{pre_query_safety.reason}"
                                    if pre_query_safety is not None
                                    and not pre_query_safety.applied
                                    else "ecia_abstain"
                                ),
                            }
                        ),
                    }
                )
                if attempted >= event_limit:
                    break

        canonical_ids, collisions = self.registry.resolve_frame(tokens)
        for collision in collisions:
            self.counters["identity_collisions_resolved"] += 1
            diagnostics.append({"type": "identity_collision", "frame": frame_id, **collision})
        output = rows.copy()
        if len(output):
            output[:, 4] = np.asarray(canonical_ids, dtype=float)

        for observation, canonical_id in zip(current_observations, canonical_ids):
            observation.canonical_id = int(canonical_id)
            self.history_by_token[observation.token].append(observation)
            self.global_history.append(observation)
            self.observation_counts[observation.token] += 1
        self.active_by_raw = current_by_raw
        self.counters["frames"] += 1
        self.counters["raw_tracks"] += len(raw_tracks)
        self.latency_ms["controller_ms"] += (time.perf_counter() - started) * 1000.0
        return output, diagnostics

    def summary(self) -> dict:
        frames = max(1, int(self.counters["frames"]))
        triggers = max(1, int(self.counters["triggers"]))
        return {
            "mode": self.mode,
            "counters": dict(self.counters),
            "latency": {
                **{key: float(value) for key, value in self.latency_ms.items()},
                "controller_mean_ms_per_frame": self.latency_ms["controller_ms"] / frames,
                "recovery_mean_ms_per_trigger": (
                    self.latency_ms["hive_ms"]
                    + self.latency_ms["hr_ms"]
                    + self.latency_ms["ecia_ms"]
                )
                / triggers,
            },
            "hr_cost": dict(self.hr_cost),
            "model_cache": (
                dict(self.model_bundle.cache_counters)
                if self.model_bundle is not None
                else {}
            ),
        }

