"""Collision-safe online canonical identity state.

BoT-SORT raw IDs are intentionally left untouched because they index internal
Kalman and appearance state. Confirmed recovery updates an output identity
alias from the current frame onward. This is an online tracker state update,
not an offline suffix replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import TrackToken


@dataclass
class MergeResult:
    applied: bool
    reason: str
    canonical_id: int | None = None


class CanonicalIdentityRegistry:
    """Map generation-safe raw tokens to unique online output identities."""

    def __init__(self, *, raw_reuse_gap: int = 60, virtual_id_offset: int = 1_000_000) -> None:
        self.raw_reuse_gap = int(raw_reuse_gap)
        self.virtual_id_offset = int(virtual_id_offset)
        self._generation: dict[int, int] = {}
        self._last_seen: dict[int, int] = {}
        self._token_to_canonical: dict[TrackToken, int] = {}
        self._active: set[TrackToken] = set()
        self._retired: set[TrackToken] = set()
        self._virtual_next = self.virtual_id_offset

    def token_for(self, raw_id: int, frame_id: int) -> tuple[TrackToken, bool]:
        raw_id = int(raw_id)
        previous = self._last_seen.get(raw_id)
        generation = self._generation.get(raw_id, 0)
        if previous is not None and frame_id - previous > self.raw_reuse_gap:
            generation += 1
            self._generation[raw_id] = generation
        else:
            self._generation.setdefault(raw_id, generation)
        token = TrackToken(raw_id, generation)
        created = token not in self._token_to_canonical
        if created:
            self._token_to_canonical[token] = raw_id if generation == 0 else self._allocate_virtual()
        self._last_seen[raw_id] = int(frame_id)
        return token, created

    def _allocate_virtual(self) -> int:
        value = self._virtual_next
        self._virtual_next += 1
        return value

    def canonical(self, token: TrackToken) -> int:
        if token not in self._token_to_canonical:
            self._token_to_canonical[token] = (
                token.raw_id if token.generation == 0 else self._allocate_virtual()
            )
        return self._token_to_canonical[token]

    def begin_frame(self, active_tokens: set[TrackToken]) -> None:
        self._active = set(active_tokens)

    def merge_successor(
        self,
        *,
        old_token: TrackToken,
        new_token: TrackToken,
        new_observation_count: int,
        max_new_observations: int,
    ) -> MergeResult:
        """Make ``new_token`` continue ``old_token`` without mutating BoT-SORT.

        Active ownership and output uniqueness are checked before the alias is
        committed. The old raw token is retired as the predecessor; if it later
        reappears while the successor is active, collision resolution gives it
        a fresh virtual ID rather than emitting duplicate identities.
        """

        if old_token == new_token:
            return MergeResult(False, "same_track_token")
        if new_observation_count > max_new_observations:
            return MergeResult(False, "new_track_too_old")
        if old_token in self._active:
            return MergeResult(False, "old_identity_still_active")
        old_canonical = self.canonical(old_token)
        owner = next(
            (
                token
                for token in self._active
                if token != new_token and self.canonical(token) == old_canonical
            ),
            None,
        )
        if owner is not None:
            return MergeResult(False, "canonical_id_active_conflict")
        self._token_to_canonical[new_token] = old_canonical
        self._retired.add(old_token)
        return MergeResult(True, "canonical_identity_transferred", old_canonical)

    def resolve_frame(self, tokens: list[TrackToken]) -> tuple[list[int], list[dict]]:
        """Guarantee one output identity per current detection."""

        groups: dict[int, list[tuple[int, TrackToken]]] = {}
        for index, token in enumerate(tokens):
            groups.setdefault(self.canonical(token), []).append((index, token))
        outputs = [0] * len(tokens)
        collisions = []
        for canonical, members in groups.items():
            # A confirmed successor always owns the recovered identity even if
            # Ultralytics returns the retired predecessor first in its array.
            owner_index, owner = next(
                (member for member in members if member[1] not in self._retired),
                members[0],
            )
            outputs[owner_index] = canonical
            for index, token in members:
                if index == owner_index:
                    continue
                replacement = self._allocate_virtual()
                self._token_to_canonical[token] = replacement
                collisions.append(
                    {
                        "token": token.key(),
                        "conflicting_token": owner.key(),
                        "canonical_id": canonical,
                        "replacement_id": replacement,
                        "reason": (
                            "retired_raw_id_reactivated"
                            if token in self._retired
                            else "duplicate_active_identity"
                        ),
                    }
                )
                outputs[index] = replacement
        return outputs, collisions

