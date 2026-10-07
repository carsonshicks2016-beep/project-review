"""
HealthBridge AI - Clinical Action Scoring System
Prioritizes health findings by clinical significance and assigns action scores.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
import json

from blood_parser import BIOMARKER_DB


class ActionPriority(Enum):
    """Clinical action priority levels"""
    CRITICAL = ("Critical", "Immediate action required - consult healthcare provider", 100, "#dc2626")
    HIGH = ("High", "Address within 2-4 weeks", 80, "#ea580c")
    MODERATE = ("Moderate", "Address within 1-3 months", 60, "#d97706")
    LOW = ("Low", "Address within 6 months", 40, "#65a30d")
    INFORMATIONAL = ("Informational", "For awareness, no immediate action", 20, "#6b7280")

    def __init__(self, label, guidance, score, color):
        self.label = label
        self.guidance = guidance
        self.score = score
        self.color = color


class ClinicalImpact(Enum):
    """Types of clinical impact"""
    MORTALITY_RISK = ("Mortality Risk", 5.0)
    MORBIDITY_RISK = ("Morbidity Risk", 4.0)
    QUALITY_OF_LIFE = ("Quality of Life", 3.0)
    PREVENTIVE = ("Preventive Health", 2.0)
    OPTIMIZATION = ("Health Optimization", 1.0)

    def __init__(self, label, weight):
        self.label = label
        self.weight = weight


@dataclass
class ClinicalAction:
    """A specific clinical action recommendation"""
    action_id: str
    title: str
    description: str
    priority: ActionPriority
    impact: ClinicalImpact
    category: str
    estimated_benefit: str  # e.g., "Reduces risk by 30%"
    time_to_benefit: str  # e.g., "3-6 months"
    evidence_level: str  # "Strong", "Moderate", "Limited"
    costs: Dict[str, str] = field(default_factory=dict)  # financial, time, complexity
    prerequisites: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)


@dataclass
class ScoredFinding:
    """A health finding with computed clinical action score"""
    finding_id: str
    name: str
    category: str
    severity: str  # critical, warning, insight
    description: str
    implications: str

    # Scoring components
    genetic_risk: float  # 0-100
    biomarker_deviation: float  # 0-100
    clinical_evidence_score: float  # 0-100

    # Action items
    actions: List[ClinicalAction] = field(default_factory=list)

    # Metadata
    confidence: float = 50.0
    urgency_days: Optional[int] = None

    @property
    def composite_risk_score(self) -> float:
        """Calculate composite risk score 0-100"""
        # Weighted combination: genetic predisposition + current state
        genetic_weight = 0.3
        biomarker_weight = 0.4
        evidence_weight = 0.3

        return round(
            self.genetic_risk * genetic_weight +
            self.biomarker_deviation * biomarker_weight +
            self.clinical_evidence_score * evidence_weight,
            1
        )

    @property
    def action_priority(self) -> ActionPriority:
        """Determine action priority based on composite score"""
        score = self.composite_risk_score
        if self.severity == "critical":
            return ActionPriority.CRITICAL
        elif score >= 75:
            return ActionPriority.HIGH
        elif score >= 60:
            return ActionPriority.MODERATE
        elif score >= 40:
            return ActionPriority.LOW
        else:
            return ActionPriority.INFORMATIONAL

    @property
    def time_to_intervention(self) -> str:
        """Recommended time to intervention"""
        if self.urgency_days:
            if self.urgency_days <= 7:
                return "Within 1 week"
            elif self.urgency_days <= 30:
                return "Within 1 month"
            elif self.urgency_days <= 90:
                return "Within 3 months"

        priority = self.action_priority
        return priority.guidance

    def to_dict(self) -> Dict:
        return {
            "finding_id": self.finding_id,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "composite_risk_score": self.composite_risk_score,
            "action_priority": {
                "level": self.action_priority.label,
                "guidance": self.action_priority.guidance,
                "color": self.action_priority.color
            },
            "time_to_intervention": self.time_to_intervention,
            "description": self.description,
            "implications": self.implications,
            "actions": [
                {
                    "title": a.title,
                    "description": a.description,
                    "priority": a.priority.label,
                    "impact": a.impact.label,
                    "estimated_benefit": a.estimated_benefit,
                    "time_to_benefit": a.time_to_benefit,
                    "evidence_level": a.evidence_level
                }
                for a in self.actions
            ],
            "confidence": self.confidence
        }


class ActionScoringEngine:
    """Engine for scoring and prioritizing clinical findings"""

    # Risk modifiers based on biomarker patterns
    RISK_THRESHOLDS = {
        # Metabolic
        "glucose": {"optimal": 85, "warning": 100, "critical": 126},
        "hba1c": {"optimal": 5.3, "warning": 5.7, "critical": 6.5},
        "insulin": {"optimal": 6, "warning": 15, "critical": 25},
        "homa_ir": {"optimal": 1.0, "warning": 2.5, "critical": 4.0},

        # Lipids
        "ldl": {"optimal": 70, "warning": 100, "critical": 160},
        "apob": {"optimal": 60, "warning": 90, "critical": 130},
        "triglycerides": {"optimal": 100, "warning": 150, "critical": 200},
        "lpa": {"optimal": 30, "warning": 50, "critical": 100},

        # Inflammation
        "hscrp": {"optimal": 1.0, "warning": 3.0, "critical": 10.0},
        "homocysteine": {"optimal": 8, "warning": 12, "critical": 15},
    }

    # Genetic risk scores for key variants
    GENETIC_RISKS = {
        "rs7903146": {"heterozygous": 40, "homozygous": 70},  # TCF7L2
        "rs429358": {"heterozygous": 60, "homozygous": 90},  # APOE4
        "rs1801133": {"heterozygous": 35, "homozygous": 60},  # MTHFR
        "rs6025": {"heterozygous": 50, "homozygous": 95},  # Factor V Leiden
        "rs4244285": {"heterozygous": 45, "homozygous": 80},  # CYP2C19
        "rs1799963": {"heterozygous": 40, "homozygous": 75},  # Prothrombin
    }

    def __init__(self):
        self.findings: List[ScoredFinding] = []

    def calculate_biomarker_deviation(self, biomarker: str, value: float,
                                      optimal_range: tuple) -> float:
        """
        Calculate how far a biomarker deviates from optimal range (0-100 scale)
        Returns 0 if in optimal range, up to 100 if severely deviated
        """
        optimal_low, optimal_high = optimal_range
        if optimal_low <= 0:
            optimal_low = max(optimal_high * 0.8, 1e-6)
        if optimal_high <= 0:
            optimal_high = max(optimal_low * 1.2, 1e-6)

        warning = self.RISK_THRESHOLDS.get(biomarker, {}).get("warning", optimal_high * 1.2)
        critical = self.RISK_THRESHOLDS.get(biomarker, {}).get("critical", max(optimal_high * 1.5, warning * 1.2, 1e-6))

        if optimal_low <= value <= optimal_high:
            return 0.0

        if value < optimal_low:
            # Below optimal - usually less concerning than above
            deviation = (optimal_low - value) / optimal_low
            return min(50, deviation * 50)

        # Above optimal
        if value <= warning:
            # Between optimal and warning
            range_size = warning - optimal_high
            deviation = (value - optimal_high) / range_size if range_size > 0 else 0
            return 25 + deviation * 25
        elif value <= critical:
            # Between warning and critical
            range_size = critical - warning
            deviation = (value - warning) / range_size if range_size > 0 else 0
            return 50 + deviation * 30
        else:
            # Above critical
            return 80 + min(20, (value - critical) / critical * 20)

    def calculate_genetic_risk(self, rsid: str, zygosity: str) -> float:
        """Calculate genetic risk score for a variant"""
        risks = self.GENETIC_RISKS.get(rsid.lower().lstrip('rs'), {})

        if zygosity == "homozygous_risk":
            return risks.get("homozygous", 50)
        elif zygosity == "heterozygous":
            return risks.get("heterozygous", 30)
        return 0.0

    def score_pattern(self, pattern: Dict, snp_data: Dict,
                     biomarker_data: Dict, evidence_score: float = 50.0) -> ScoredFinding:
        """
        Score a cross-source pattern

        Args:
            pattern: Pattern dict from pattern_engine
            snp_data: Dict of rsid -> {zygosity, genotype}
            biomarker_data: Dict of biomarker -> value
            evidence_score: Clinical evidence confidence score (0-100)
        """
        pattern_id = pattern.get("id", "unknown")
        name = pattern.get("name", "Unknown Pattern")
        category = pattern.get("category", "General")
        severity = pattern.get("severity", "insight")

        # Calculate genetic risk component
        genetic_risk = 0
        required_snps = pattern.get("required_snps", [])
        for rsid in required_snps:
            if rsid in snp_data:
                zygosity = snp_data[rsid].get("zygosity", "unknown")
                genetic_risk = max(genetic_risk, self.calculate_genetic_risk(rsid, zygosity))

        # Calculate biomarker deviation
        biomarker_deviation = 0
        required_biomarkers = pattern.get("required_biomarkers", {})
        for bio, condition in required_biomarkers.items():
            if bio in biomarker_data:
                threshold = self.RISK_THRESHOLDS.get(bio, {})
                if "optimal" in threshold:
                    optimal_value = threshold["optimal"]
                    optimal_range = (optimal_value * 0.8, optimal_value * 1.2)
                else:
                    db_entry = BIOMARKER_DB.get(bio)
                    if db_entry:
                        optimal_range = (
                            db_entry.get("optimal_low", biomarker_data[bio] * 0.8),
                            db_entry.get("optimal_high", biomarker_data[bio] * 1.2),
                        )
                    else:
                        baseline = max(abs(float(biomarker_data[bio])), 1.0)
                        optimal_range = (baseline * 0.8, baseline * 1.2)
                deviation = self.calculate_biomarker_deviation(
                    bio, biomarker_data[bio], optimal_range
                )
                biomarker_deviation = max(biomarker_deviation, deviation)

        # Create actions from pattern
        actions = []
        for action_desc in pattern.get("actions", []):
            actions.append(ClinicalAction(
                action_id=f"{pattern_id}_{len(actions)}",
                title=action_desc[:50],
                description=action_desc,
                priority=ActionPriority.MODERATE,  # Will be adjusted based on score
                impact=self._determine_impact(category),
                category=category,
                estimated_benefit="See clinical literature",
                time_to_benefit="Variable",
                evidence_level="Moderate"
            ))

        # Determine urgency
        urgency = self._calculate_urgency(severity, genetic_risk, biomarker_deviation)

        finding = ScoredFinding(
            finding_id=pattern_id,
            name=name,
            category=category,
            severity=severity,
            description=pattern.get("description", ""),
            implications=pattern.get("implications", ""),
            genetic_risk=genetic_risk,
            biomarker_deviation=biomarker_deviation,
            clinical_evidence_score=evidence_score,
            actions=actions,
            confidence=evidence_score,
            urgency_days=urgency
        )

        return finding

    def _determine_impact(self, category: str) -> ClinicalImpact:
        """Map category to clinical impact type"""
        category_map = {
            "Metabolic": ClinicalImpact.MORBIDITY_RISK,
            "Cardiovascular": ClinicalImpact.MORTALITY_RISK,
            "Methylation": ClinicalImpact.MORBIDITY_RISK,
            "Vitamins": ClinicalImpact.QUALITY_OF_LIFE,
            "Nutrition": ClinicalImpact.PREVENTIVE,
            "Pharmacogenomics": ClinicalImpact.MORBIDITY_RISK,
            "Autoimmune": ClinicalImpact.MORBIDITY_RISK,
            "Cancer": ClinicalImpact.MORTALITY_RISK,
        }
        return category_map.get(category, ClinicalImpact.OPTIMIZATION)

    def _calculate_urgency(self, severity: str, genetic_risk: float,
                         biomarker_deviation: float) -> Optional[int]:
        """Calculate urgency in days"""
        if severity == "critical":
            if genetic_risk > 60 or biomarker_deviation > 70:
                return 7
            return 30
        elif severity == "warning":
            if genetic_risk > 50 or biomarker_deviation > 60:
                return 90
            return 180
        return None

    def prioritize_findings(self, findings: List[ScoredFinding]) -> List[ScoredFinding]:
        """Sort findings by clinical priority"""
        def sort_key(f: ScoredFinding) -> tuple:
            # Sort by: composite risk (desc), action priority score (desc), confidence (desc)
            return (
                -f.composite_risk_score,
                -f.action_priority.score,
                -f.confidence
            )

        return sorted(findings, key=sort_key)

    def generate_action_plan(self, findings: List[ScoredFinding]) -> Dict:
        """Generate a prioritized action plan"""
        prioritized = self.prioritize_findings(findings)

        immediate = [f for f in prioritized if f.action_priority == ActionPriority.CRITICAL]
        short_term = [f for f in prioritized if f.action_priority == ActionPriority.HIGH]
        medium_term = [f for f in prioritized if f.action_priority == ActionPriority.MODERATE]
        long_term = [f for f in prioritized if f.action_priority == ActionPriority.LOW]
        informational = [f for f in prioritized if f.action_priority == ActionPriority.INFORMATIONAL]

        return {
            "summary": {
                "total_findings": len(findings),
                "critical_count": len(immediate),
                "high_priority_count": len(short_term),
                "average_risk_score": round(
                    sum(f.composite_risk_score for f in findings) / len(findings), 1
                ) if findings else 0
            },
            "action_timeline": {
                "immediate_1_week": [f.to_dict() for f in immediate],
                "short_term_1_month": [f.to_dict() for f in short_term],
                "medium_term_3_months": [f.to_dict() for f in medium_term],
                "long_term_6_months": [f.to_dict() for f in long_term],
                "informational": [f.to_dict() for f in informational]
            },
            "top_3_priorities": [f.to_dict() for f in prioritized[:3]]
        }


# Clinical action rules database
CLINICAL_ACTIONS = {
    # Insulin resistance
    "insulin_resistance_triad": {
        "actions": [
            ClinicalAction(
                action_id="ir_1",
                title="Implement time-restricted eating",
                description="Eat within an 8-10 hour window daily to improve insulin sensitivity",
                priority=ActionPriority.HIGH,
                impact=ClinicalImpact.MORBIDITY_RISK,
                category="Metabolic",
                estimated_benefit="Reduces fasting insulin by 20-30%",
                time_to_benefit="4-8 weeks",
                evidence_level="Strong",
                costs={"financial": "Free", "time": "Daily discipline", "complexity": "Low"}
            ),
            ClinicalAction(
                action_id="ir_2",
                title="Post-meal walking",
                description="Take a 10-15 minute walk after each meal",
                priority=ActionPriority.MODERATE,
                impact=ClinicalImpact.MORBIDITY_RISK,
                category="Metabolic",
                estimated_benefit="Reduces post-meal glucose spikes by 20-30%",
                time_to_benefit="Immediate",
                evidence_level="Strong",
                costs={"financial": "Free", "time": "30-45 min/day", "complexity": "Low"}
            ),
        ]
    },
    # APOE4
    "apoe4_lipid_risk": {
        "actions": [
            ClinicalAction(
                action_id="apoe4_1",
                title="Request coronary artery calcium score",
                description="Non-invasive CT scan to assess subclinical atherosclerosis",
                priority=ActionPriority.HIGH,
                impact=ClinicalImpact.MORTALITY_RISK,
                category="Cardiovascular",
                estimated_benefit="Direct visualization of arterial plaque burden",
                time_to_benefit="Immediate result",
                evidence_level="Strong",
                costs={"financial": "$75-300", "time": "15 minutes", "complexity": "Low"}
            ),
            ClinicalAction(
                action_id="apoe4_2",
                title="Reduce saturated fat intake",
                description="Replace saturated fats with monounsaturated fats",
                priority=ActionPriority.HIGH,
                impact=ClinicalImpact.MORTALITY_RISK,
                category="Cardiovascular",
                estimated_benefit="Reduces LDL by 15-25 mg/dL in APOE4 carriers",
                time_to_benefit="4-12 weeks",
                evidence_level="Strong",
                costs={"financial": "Neutral", "time": "Ongoing", "complexity": "Medium"}
            ),
        ]
    },
}


if __name__ == "__main__":
    # Test the scoring engine
    print("Testing Action Scoring Engine...")

    engine = ActionScoringEngine()

    # Test biomarker deviation calculation
    deviation = engine.calculate_biomarker_deviation("glucose", 95, (72, 85))
    print(f"\nGlucose 95 mg/dL deviation score: {deviation}")

    deviation = engine.calculate_biomarker_deviation("glucose", 110, (72, 85))
    print(f"Glucose 110 mg/dL deviation score: {deviation}")

    # Test genetic risk
    risk = engine.calculate_genetic_risk("rs7903146", "heterozygous")
    print(f"\nTCF7L2 heterozygous risk score: {risk}")

    risk = engine.calculate_genetic_risk("rs429358", "homozygous_risk")
    print(f"APOE4 homozygous risk score: {risk}")

    print("\nAction Scoring Engine test complete.")
