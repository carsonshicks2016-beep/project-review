"""Faithful-v2 Porsche 919 record-program foundations.

Importing this package does not assert that the public-anchor baseline is a
faithful model.  Always inspect ``EvidenceReadinessEvaluator`` and the selected
backend's ``capability`` before presenting a fidelity or certification label.
"""
from .backends import (
    AuthorityModelMetadataV1,
    AuthoritySnapshotV1,
    EnergyLedgerV1,
    FIXED_TIMESTEP_S,
    MJXTrainingTwin,
    MuJoCoAuthority,
    TwinSnapshotV1,
)
from .defaults import (
    default_record_scenario,
    official_919evo_vehicle_spec,
    training_fallback_track,
)
from .schemas import (
    ActiveAeroMode,
    BackendCapability,
    Confidence,
    DistributionKind,
    DriverActionV2,
    DriverControllerStatusV1,
    DriverObservationV2,
    EvidenceReadinessEvaluator,
    EvidenceReadinessReport,
    EvidenceRef,
    MapPreviewPointV1,
    PermittedDriverControlsV1,
    PhysicalParameter,
    PhysicsBackend,
    PhysicsIdentity,
    PhysicsValidationEvidenceV1,
    RecordScenarioV1,
    ShiftRequest,
    TelemetryEvent,
    TimingPlaneV1,
    TrackSampleV2,
    TrackSurfaceV2,
    TrackSurveyValidationV1,
    UncertaintyDistribution,
    ValidationArtifactKind,
    ValidationArtifactV1,
    ValidationMetricV1,
    VehicleSpecV2,
    VehicleTelemetryV2,
    canonical_json,
    content_sha256,
)

# A/V is kept in a separate module so data-only consumers can depend on the
# schemas without using it, while the package surface remains convenient.
from .av import CarAssetRegistry, TruthfulAvAdapter
from .evidence import (
    DEFAULT_TRUST_STORE,
    EvidenceArtifactV1,
    EvidenceFreezeV1,
    EvidencePackageV1,
    EvidenceTrustStoreV1,
    REQUIRED_EVIDENCE_ROLES,
    TrustedEvidenceKeyV1,
    evidence_inventory,
    resolve_evidence_vault,
    verify_evidence_freeze,
    verify_package_directory,
)
__all__ = [
    "ActiveAeroMode", "AuthorityModelMetadataV1", "AuthoritySnapshotV1",
    "BackendCapability", "CarAssetRegistry", "Confidence", "DistributionKind",
    "DriverActionV2", "DriverControllerStatusV1", "DriverObservationV2", "EnergyLedgerV1",
    "DEFAULT_TRUST_STORE", "EvidenceArtifactV1", "EvidenceFreezeV1",
    "EvidencePackageV1", "EvidenceReadinessEvaluator", "EvidenceReadinessReport",
    "EvidenceRef", "EvidenceTrustStoreV1",
    "FIXED_TIMESTEP_S", "MJXTrainingTwin", "MapPreviewPointV1", "MuJoCoAuthority",
    "PermittedDriverControlsV1", "PhysicalParameter", "PhysicsBackend", "PhysicsIdentity",
    "PhysicsValidationEvidenceV1", "RecordScenarioV1", "ShiftRequest", "TelemetryEvent",
    "TimingPlaneV1", "TrackSampleV2", "TrackSurfaceV2", "TrackSurveyValidationV1",
    "REQUIRED_EVIDENCE_ROLES", "TrustedEvidenceKeyV1", "TruthfulAvAdapter",
    "TwinSnapshotV1", "UncertaintyDistribution",
    "ValidationArtifactKind", "ValidationArtifactV1", "ValidationMetricV1",
    "VehicleSpecV2", "VehicleTelemetryV2", "canonical_json", "content_sha256",
    "default_record_scenario", "evidence_inventory", "official_919evo_vehicle_spec",
    "resolve_evidence_vault", "training_fallback_track", "verify_evidence_freeze",
    "verify_package_directory",
]
