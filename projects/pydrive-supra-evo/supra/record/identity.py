"""Edition-scoped identities and checkpoint compatibility rules."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

from ._canonical import canonicalize, require_sha256, sha256_object


IDENTITY_SCHEMA = "faithful-run-identity-v1"
CHECKPOINT_IDENTITY_SCHEMA = "faithful-checkpoint-identity-v1"
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


def _safe_component(value: str, field_name: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a lowercase artifact-safe identifier; got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Immutable identity for everything that can affect a record result."""

    edition: str
    run_id: str
    vehicle_spec_sha256: str
    track_surface_sha256: str
    physics_identity_sha256: str
    controller_sha256: str
    observation_schema_sha256: str
    action_schema_sha256: str
    source_sha256: str
    dependencies_sha256: str
    protocol_sha256: str
    schema: str = IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != IDENTITY_SCHEMA:
            raise ValueError(f"unsupported run identity schema: {self.schema}")
        _safe_component(self.edition, "edition")
        _safe_component(self.run_id, "run_id")
        for name in (
            "vehicle_spec_sha256", "track_surface_sha256",
            "physics_identity_sha256", "controller_sha256",
            "observation_schema_sha256", "action_schema_sha256",
            "source_sha256", "dependencies_sha256", "protocol_sha256",
        ):
            require_sha256(getattr(self, name), name)

    @property
    def sha256(self) -> str:
        return sha256_object(self)

    @property
    def semantic_hashes(self) -> tuple[str, ...]:
        """All record semantics, excluding the organizational run identifier."""
        return (
            self.vehicle_spec_sha256, self.track_surface_sha256,
            self.physics_identity_sha256, self.controller_sha256,
            self.observation_schema_sha256, self.action_schema_sha256,
            self.source_sha256, self.dependencies_sha256, self.protocol_sha256,
        )

    def to_dict(self) -> dict:
        return canonicalize(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RunIdentity":
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """The only legal storage namespace for a faithful-v2 run."""

    runtime_root: Path
    identity: RunIdentity

    @property
    def edition_root(self) -> Path:
        return self.runtime_root / "fable5" / "editions" / self.identity.edition

    @property
    def run_root(self) -> Path:
        return self.edition_root / "runs" / self.identity.run_id

    @property
    def manifest(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def latest_training_evaluation(self) -> Path:
        return self.run_root / "latest_training_evaluation.json"

    @property
    def events(self) -> Path:
        return self.run_root / "events.jsonl"

    @property
    def pit_state(self) -> Path:
        return self.run_root / "pit_state.json"

    @property
    def oracle_result(self) -> Path:
        return self.run_root / "oracle" / "result.json"

    @property
    def certificates(self) -> Path:
        return self.run_root / "certificates"

    @property
    def checkpoints(self) -> Path:
        return self.run_root / "checkpoints"

    @property
    def training_hof(self) -> Path:
        return self.checkpoints / "training_hof"

    @property
    def certified_champion(self) -> Path:
        return self.edition_root / "certified_champion.json"

    def assert_scoped(self, path: Path) -> Path:
        """Reject a path that escapes this exact run's namespace."""
        root = self.run_root.resolve()
        candidate = path.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"artifact escapes run namespace: {candidate}")
        return candidate


class TrainingStage(str, Enum):
    FOUNDATION = "foundation"
    FLOW = "flow"
    FINISH = "finish"
    FAST = "fast"
    FRONTIER = "frontier"


_STAGE_ORDER = tuple(TrainingStage)


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    run: RunIdentity
    stage: TrainingStage
    policy_architecture_sha256: str
    policy_state_sha256: str
    normalizer_schema_sha256: str
    normalizer_state_sha256: str
    optimizer_schema_sha256: str
    optimizer_state_sha256: str
    scheduler_state_sha256: str
    rng_state_sha256: str
    training_budget_sha256: str
    schema: str = CHECKPOINT_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", TrainingStage(self.stage))
        if self.schema != CHECKPOINT_IDENTITY_SCHEMA:
            raise ValueError(f"unsupported checkpoint identity schema: {self.schema}")
        for name in (
            "policy_architecture_sha256", "policy_state_sha256",
            "normalizer_schema_sha256", "normalizer_state_sha256",
            "optimizer_schema_sha256", "optimizer_state_sha256",
            "scheduler_state_sha256", "rng_state_sha256",
            "training_budget_sha256",
        ):
            require_sha256(getattr(self, name), name)

    @property
    def sha256(self) -> str:
        return sha256_object(self)

    def to_dict(self) -> dict:
        return canonicalize(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CheckpointIdentity":
        data = dict(payload)
        data["run"] = RunIdentity.from_dict(data["run"])
        data["stage"] = TrainingStage(data["stage"])
        return cls(**data)


class CheckpointUse(str, Enum):
    EXACT_RESUME = "exact_resume"
    STAGE_WARM_START = "stage_warm_start"
    EXPLICIT_MIGRATION = "explicit_migration"


@dataclass(frozen=True, slots=True)
class CompatibilityDecision:
    allowed: bool
    reason: str
    load_policy: bool = False
    load_normalizer: bool = False
    load_optimizer: bool = False
    load_scheduler: bool = False
    load_rng: bool = False
    carry_evaluation: bool = False
    certifiable_lineage: bool = False


def checkpoint_compatibility(
    source: CheckpointIdentity,
    target: CheckpointIdentity,
    use: CheckpointUse,
    *,
    migration_adapter_id: str | None = None,
) -> CompatibilityDecision:
    """Apply fail-closed import semantics to a checkpoint.

    Exact resumes carry all state only when every identity field matches.
    Stage warm starts carry policy and normalizer but reset all optimizer,
    scheduler, RNG and evaluation state. Explicit migrations require a named
    adapter and permanently break record-certifiable lineage.
    """
    if use is CheckpointUse.EXACT_RESUME:
        if source != target:
            return CompatibilityDecision(False, "exact resume identity mismatch")
        return CompatibilityDecision(
            True, "exact identity match", True, True, True, True, True, True, True
        )

    if use is CheckpointUse.STAGE_WARM_START:
        if source.run != target.run:
            return CompatibilityDecision(False, "stage warm start requires exact run identity")
        if source.stage == target.stage:
            return CompatibilityDecision(False, "same-stage load must use exact resume")
        if _STAGE_ORDER.index(target.stage) != _STAGE_ORDER.index(source.stage) + 1:
            return CompatibilityDecision(
                False, "stage warm start must advance exactly one stage"
            )
        if source.policy_architecture_sha256 != target.policy_architecture_sha256:
            return CompatibilityDecision(False, "policy architecture mismatch")
        if source.policy_state_sha256 != target.policy_state_sha256:
            return CompatibilityDecision(
                False,
                "stage warm-start target must identify the exact carried policy state",
            )
        if source.normalizer_schema_sha256 != target.normalizer_schema_sha256:
            return CompatibilityDecision(False, "normalizer schema mismatch")
        if source.normalizer_state_sha256 != target.normalizer_state_sha256:
            return CompatibilityDecision(
                False,
                "stage warm-start target must identify the exact carried normalizer state",
            )
        return CompatibilityDecision(
            True, "same-run stage warm start; optimizer and evidence reset",
            load_policy=True, load_normalizer=True, certifiable_lineage=True,
        )

    if not migration_adapter_id:
        return CompatibilityDecision(False, "explicit migration requires migration_adapter_id")
    _safe_component(migration_adapter_id, "migration_adapter_id")
    if source.policy_architecture_sha256 != target.policy_architecture_sha256:
        return CompatibilityDecision(False, "migration policy architecture mismatch")
    return CompatibilityDecision(
        True, f"migration via {migration_adapter_id}; lineage is noncertifiable",
        load_policy=True, certifiable_lineage=False,
    )
