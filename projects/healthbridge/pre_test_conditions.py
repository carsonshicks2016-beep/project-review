"""
HealthBridge AI - Pre-Test Conditions Module
Captures and accounts for factors that affect biomarker interpretation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum


class FastingStatus(Enum):
    """Fasting duration categories"""
    NOT_FASTED = ("Not fasted", 0, 4)
    SHORT_FAST = ("4-8 hours", 4, 8)
    STANDARD_FAST = ("8-12 hours (standard)", 8, 12)
    EXTENDED_FAST = ("12+ hours", 12, 24)

    def __init__(self, label, min_hours, max_hours):
        self.label = label
        self.min_hours = min_hours
        self.max_hours = max_hours


class ExerciseTiming(Enum):
    """Exercise timing relative to blood draw"""
    NONE = ("No recent exercise", 0, "No effect")
    SAME_DAY = ("Same day (2-8h prior)", 8, "Moderate effect")
    RECENT = ("Within 2 hours", 2, "Significant effect")

    def __init__(self, label, hours_prior, effect):
        self.label = label
        self.hours_prior = hours_prior
        self.effect = effect


class MenstrualPhase(Enum):
    """Menstrual cycle phase for female patients"""
    NOT_APPLICABLE = ("N/A", "Not applicable")
    FOLLICULAR = ("Follicular", "Days 1-14 - Baseline hormones")
    OVULATION = ("Ovulation", "Days 14-16 - Peak hormones")
    LUTEAL = ("Luteal", "Days 15-28 - Elevated progesterone")
    MENSES = ("Menses", "Days 1-5 - Hormones low")

    def __init__(self, label, description):
        self.label = label
        self.description = description


@dataclass
class MedicationEntry:
    """A medication that may affect biomarkers"""
    name: str
    dosage: str
    last_taken: datetime
    frequency: str  # e.g., "daily", "as needed"
    known_effects: List[str] = field(default_factory=list)

    def is_active(self, blood_draw_time: datetime) -> bool:
        """Check if medication is likely active at blood draw"""
        # Simplified - in practice would use pharmacokinetic data
        hours_since = (blood_draw_time - self.last_taken).total_seconds() / 3600
        return hours_since < 24


@dataclass
class PreTestConditions:
    """
    Comprehensive pre-test conditions for biomarker interpretation
    """
    # Timing
    collection_date: datetime
    collection_time: datetime
    fasting_hours: float

    # Physical state
    sleep_hours_prior: float
    exercise_timing: ExerciseTiming

    # Health state
    acute_illness: bool
    illness_description: Optional[str] = None
    injury_recent: bool = False
    inflammation_present: bool = False

    # Female-specific
    menstrual_phase: MenstrualPhase = MenstrualPhase.NOT_APPLICABLE
    hormonal_contraception: bool = False
    pregnancy_status: Optional[str] = None

    # Lifestyle
    alcohol_last_24h: bool = False
    caffeine_last_8h: bool = False
    smoking_status: str = "unknown"  # "never", "former", "current", "vaped_recently"

    # Environmental
    altitude: float = 0.0  # meters above sea level
    temperature_extreme: bool = False  # extreme heat or cold exposure
    travel_recent: bool = False  # international travel/jet lag

    # Medications and supplements
    medications: List[MedicationEntry] = field(default_factory=list)
    supplements_last_24h: List[str] = field(default_factory=list)

    # Lab-specific
    tourniquet_time_seconds: Optional[int] = None  # <60s ideal
    tube_order: Optional[str] = None  # Was glucose tube first?

    def get_fasting_status(self) -> FastingStatus:
        """Determine fasting status category"""
        if self.fasting_hours >= 12:
            return FastingStatus.EXTENDED_FAST
        elif self.fasting_hours >= 8:
            return FastingStatus.STANDARD_FAST
        elif self.fasting_hours >= 4:
            return FastingStatus.SHORT_FAST
        return FastingStatus.NOT_FASTED

    def to_dict(self) -> Dict:
        return {
            "collection_datetime": self.collection_time.isoformat(),
            "fasting_hours": self.fasting_hours,
            "fasting_status": self.get_fasting_status().label,
            "sleep_hours_prior": self.sleep_hours_prior,
            "exercise_timing": self.exercise_timing.label if self.exercise_timing else "Unknown",
            "acute_illness": self.acute_illness,
            "menstrual_phase": self.menstrual_phase.label if self.menstrual_phase else "N/A",
            "alcohol_last_24h": self.alcohol_last_24h,
            "caffeine_last_8h": self.caffeine_last_8h,
            "smoking_status": self.smoking_status,
            "medications_count": len([m for m in self.medications if m.is_active(self.collection_time)]),
            "supplements_count": len(self.supplements_last_24h),
        }


class BiomarkerAdjustments:
    """
    Adjustment factors for biomarkers based on pre-test conditions
    All adjustments are percentage changes from true value
    """

    # Known effects on biomarkers (percentage bias)
    # Values indicate how much the measured value may differ from true baseline
    ADJUSTMENTS = {
        "glucose": {
            "non_fasted": +15,  # +15% if not fasted
            "short_sleep": +8,
            "recent_exercise": -10,  # Can be transiently lower
            "acute_illness": +20,
            "caffeine_recent": +10,
        },
        "insulin": {
            "non_fasted": +200,  # Massive increase if not fasted
            "short_sleep": +30,
            "recent_exercise": -40,  # Exercise increases sensitivity
            "acute_illness": +25,
        },
        "triglycerides": {
            "non_fasted": +150,  # Very sensitive to recent meals
            "alcohol_recent": +30,
            "acute_illness": +15,
        },
        "ldl": {
            "acute_illness": -15,  # Illness can falsely lower LDL
            "recent_exercise": -5,
        },
        "hdl": {
            "acute_illness": -15,
        },
        "cortisol": {
            "short_sleep": +40,
            "exercise_recent": +25,
            "acute_illness": +50,
        },
        "hsCRP": {
            "acute_illness": +200,  # Major confounder
            "injury_recent": +100,
            "exercise_within_24h": +30,
        },
        "testosterone": {
            "short_sleep": -15,
            "acute_illness": -20,
            "luteal_phase_female": -50,  # If female
        },
        "iron": {
            "acute_illness": -40,  # Acute phase response
            "recent_exercise": +10,
        },
        "ferritin": {
            "acute_illness": +100,  # Acute phase reactant
        },
    }

    @classmethod
    def calculate_adjusted_value(cls, biomarker: str, measured_value: float,
                                conditions: PreTestConditions) -> Tuple[float, List[str]]:
        """
        Calculate adjusted value and list of applied adjustments
        Returns: (adjusted_value, list_of_factors)
        """
        adjustments = cls.ADJUSTMENTS.get(biomarker.lower(), {})
        total_adjustment_pct = 0
        applied_factors = []

        # Check fasting
        if conditions.get_fasting_status() == FastingStatus.NOT_FASTED:
            if "non_fasted" in adjustments:
                total_adjustment_pct -= adjustments["non_fasted"]
                applied_factors.append(f"Non-fasted ({adjustments['non_fasted']}%)")

        # Check sleep
        if conditions.sleep_hours_prior < 6:
            if "short_sleep" in adjustments:
                total_adjustment_pct -= adjustments["short_sleep"]
                applied_factors.append(f"Short sleep ({adjustments['short_sleep']}%)")

        # Check exercise
        if conditions.exercise_timing in [ExerciseTiming.RECENT, ExerciseTiming.SAME_DAY]:
            if "recent_exercise" in adjustments:
                total_adjustment_pct -= adjustments["recent_exercise"]
                applied_factors.append(f"Recent exercise ({adjustments['recent_exercise']}%)")

        # Check illness
        if conditions.acute_illness:
            if "acute_illness" in adjustments:
                total_adjustment_pct -= adjustments["acute_illness"]
                applied_factors.append(f"Acute illness ({adjustments['acute_illness']}%)")

        # Check caffeine
        if conditions.caffeine_last_8h:
            if "caffeine_recent" in adjustments:
                total_adjustment_pct -= adjustments["caffeine_recent"]
                applied_factors.append(f"Recent caffeine ({adjustments['caffeine_recent']}%)")

        # Check alcohol
        if conditions.alcohol_last_24h:
            if "alcohol_recent" in adjustments:
                total_adjustment_pct -= adjustments["alcohol_recent"]
                applied_factors.append(f"Recent alcohol ({adjustments['alcohol_recent']}%)")

        # Apply adjustment
        adjusted = measured_value * (1 + total_adjustment_pct / 100)

        return round(adjusted, 2), applied_factors

    @classmethod
    def get_interpretation_guidance(cls, biomarker: str,
                                   conditions: PreTestConditions) -> Dict:
        """Get specific interpretation guidance for a biomarker"""
        guidance = {
            "reliability": "Good",
            "caveats": [],
            "recommendations": []
        }

        # Check if biomarker is affected by conditions
        if biomarker.lower() in cls.ADJUSTMENTS:
            affected_by = []

            if conditions.get_fasting_status() == FastingStatus.NOT_FASTED and "non_fasted" in cls.ADJUSTMENTS[biomarker.lower()]:
                affected_by.append("non-fasting state")
                guidance["caveats"].append("Measured in non-fasted state - interpret with caution")
                guidance["reliability"] = "Questionable"

            if conditions.acute_illness:
                affected_by.append("acute illness")
                guidance["caveats"].append("Acute illness can significantly alter this marker")
                guidance["reliability"] = "Poor"
                guidance["recommendations"].append("Retest after illness resolution")

            if conditions.sleep_hours_prior < 6:
                affected_by.append("sleep deprivation")

            if affected_by:
                guidance["caveats"].insert(0, f"This marker affected by: {', '.join(affected_by)}")

        return guidance


class ConditionsForm:
    """Web form handling for pre-test conditions"""

    @staticmethod
    def get_html_form() -> str:
        """Generate HTML form for collecting conditions"""
        return """
        <div class="conditions-form">
            <h3>Pre-Test Conditions</h3>
            <p class="form-intro">Help us interpret your results accurately by providing context about your blood draw.</p>

            <div class="form-grid">
                <div class="form-group">
                    <label>Fasting Duration (hours)</label>
                    <input type="number" name="fasting_hours" min="0" max="24" value="12" step="0.5">
                    <span class="hint">Standard is 8-12 hours</span>
                </div>

                <div class="form-group">
                    <label>Sleep Prior Night (hours)</label>
                    <input type="number" name="sleep_hours" min="0" max="12" value="7" step="0.5">
                </div>

                <div class="form-group">
                    <label>Exercise Before Draw</label>
                    <select name="exercise_timing">
                        <option value="none">No recent exercise</option>
                        <option value="24h">Within last 24 hours</option>
                        <option value="same_day">Same day (2-8 hours prior)</option>
                        <option value="recent">Within 2 hours</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Current Health Status</label>
                    <select name="health_status">
                        <option value="healthy">Healthy - no illness</option>
                        <option value="mild">Mild cold/illness</option>
                        <option value="acute">Acute illness</option>
                        <option value="recovering">Recovering from illness</option>
                    </select>
                </div>
            </div>

            <div class="checkbox-group">
                <label class="checkbox-label">
                    <input type="checkbox" name="caffeine"> Consumed caffeine within 8 hours
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" name="alcohol"> Consumed alcohol within 24 hours
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" name="injury"> Recent injury or inflammation
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" name="travel"> Recent travel/jet lag
                </label>
            </div>

            <div class="form-group medications">
                <label>Medications Taken (one per line)</label>
                <textarea name="medications" rows="3" placeholder="e.g., Metformin 500mg - taken this morning
Fish oil supplement - taken yesterday"></textarea>
            </div>

            <div class="form-alert">
                <strong>⚠️ Important:</strong> Certain conditions can significantly affect biomarker results.
                Be as accurate as possible for the most reliable interpretation.
            </div>
        </div>

        <style>
            .conditions-form {
                background: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 24px;
                margin: 20px 0;
            }
            .form-intro {
                color: #6b7280;
                font-size: 0.9em;
                margin-bottom: 16px;
            }
            .form-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                margin-bottom: 20px;
            }
            .form-group label {
                display: block;
                font-weight: 600;
                font-size: 0.85em;
                color: #374151;
                margin-bottom: 6px;
            }
            .form-group input,
            .form-group select,
            .form-group textarea {
                width: 100%;
                padding: 10px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 0.9em;
            }
            .form-group .hint {
                display: block;
                font-size: 0.75em;
                color: #9ca3af;
                margin-top: 4px;
            }
            .checkbox-group {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                margin-bottom: 20px;
            }
            .checkbox-label {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 0.85em;
                color: #4b5563;
                cursor: pointer;
            }
            .checkbox-label input {
                width: 18px;
                height: 18px;
            }
            .form-alert {
                background: #fffbeb;
                border: 1px solid #fbbf24;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 0.85em;
                color: #92400e;
            }
        </style>
        """

    @staticmethod
    def parse_form_data(form_data: Dict) -> PreTestConditions:
        """Parse form data into PreTestConditions object"""
        # Parse exercise timing
        exercise_map = {
            "none": ExerciseTiming.NONE,
            "24h": ExerciseTiming.SAME_DAY,
            "same_day": ExerciseTiming.SAME_DAY,
            "recent": ExerciseTiming.RECENT
        }

        # Parse health status
        illness = form_data.get("health_status") in ["mild", "acute", "recovering"]

        # Parse medications
        medications = []
        med_text = form_data.get("medications", "")
        if med_text:
            for line in med_text.strip().split("\n"):
                if line.strip():
                    medications.append(MedicationEntry(
                        name=line.strip(),
                        dosage="unknown",
                        last_taken=datetime.now() - timedelta(hours=8),
                        frequency="unknown"
                    ))

        return PreTestConditions(
            collection_date=datetime.now(),
            collection_time=datetime.now(),
            fasting_hours=float(form_data.get("fasting_hours", 12)),
            sleep_hours_prior=float(form_data.get("sleep_hours", 7)),
            exercise_timing=exercise_map.get(form_data.get("exercise_timing", "none"), ExerciseTiming.NONE),
            acute_illness=illness,
            illness_description=form_data.get("health_status"),
            injury_recent=form_data.get("injury") == "on",
            inflammation_present=False,  # Would need assessment
            alcohol_last_24h=form_data.get("alcohol") == "on",
            caffeine_last_8h=form_data.get("caffeine") == "on",
            smoking_status="unknown",
            altitude=0,
            temperature_extreme=False,
            travel_recent=form_data.get("travel") == "on",
            medications=medications,
            tourniquet_time_seconds=None
        )


if __name__ == "__main__":
    # Test the module
    print("Testing Pre-Test Conditions Module...")

    # Create test conditions
    conditions = PreTestConditions(
        collection_date=datetime.now(),
        collection_time=datetime.now(),
        fasting_hours=4.5,  # Not properly fasted
        sleep_hours_prior=5,  # Short sleep
        exercise_timing=ExerciseTiming.SAME_DAY,
        acute_illness=False,
        alcohol_last_24h=True,
        caffeine_last_8h=True,
        smoking_status="never",
        altitude=0,
        temperature_extreme=False,
        travel_recent=False
    )

    print(f"\nFasting status: {conditions.get_fasting_status().label}")

    # Test glucose adjustment
    adjusted, factors = BiomarkerAdjustments.calculate_adjusted_value(
        "glucose", 95, conditions
    )
    print(f"\nGlucose adjustment:")
    print(f"  Measured: 95 mg/dL")
    print(f"  Adjusted: {adjusted} mg/dL")
    print(f"  Factors: {factors}")

    # Test triglycerides adjustment
    adjusted, factors = BiomarkerAdjustments.calculate_adjusted_value(
        "triglycerides", 180, conditions
    )
    print(f"\nTriglycerides adjustment:")
    print(f"  Measured: 180 mg/dL")
    print(f"  Adjusted: {adjusted} mg/dL")
    print(f"  Factors: {factors}")

    print("\nPre-Test Conditions Module test complete.")
