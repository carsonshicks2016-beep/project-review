"""
HealthBridge AI - Wearable Data Parser
Supports: Oura Ring, Apple Health, Whoop
Data types: Sleep, HRV, Resting HR, Activity, Recovery/Readiness, Stress

Each platform exports CSVs in different formats — this module normalizes
all of them into a unified wearable metrics dict for downstream analysis.

Export instructions per platform:
  Oura:        App → Profile → Account → Export Data → CSV files (sleep.csv, readiness.csv, activity.csv)
  Apple Health: Health app → Profile → Export All Health Data → export.zip → extract → HKExport.xml
                OR use a third-party exporter like Health Auto Export app → CSV
  Whoop:        App → Profile → Download My Data → CSV (sleep.csv, recovery.csv, workouts.csv, journal.csv)
"""

import csv
import json
import re
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from collections import defaultdict

def safe_mean(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None
    total = sum(v)
    return total / len(v)


# =============================================================================
# UNIFIED METRIC SCHEMA
# All parsers normalize to this structure
# =============================================================================

def empty_metrics():
    return {
        "platform": None,
        "date_range": {"start": None, "end": None, "days": 0},

        # Sleep
        "sleep": {
            "avg_total_hours": None,
            "avg_deep_hours": None,
            "avg_rem_hours": None,
            "avg_light_hours": None,
            "avg_awake_hours": None,
            "avg_efficiency_pct": None,
            "avg_latency_min": None,
            "avg_bedtime": None,
            "avg_wake_time": None,
            "sleep_consistency_score": None,  # 0-100, how consistent bedtimes are
            "nights_analyzed": 0,
        },

        # HRV
        "hrv": {
            "avg_rmssd": None,
            "avg_hrv_score": None,  # platform-specific normalized score (0-100)
            "hrv_trend": None,      # "improving", "declining", "stable"
            "low_hrv_nights_pct": None,  # % nights below personal baseline
        },

        # Heart Rate
        "heart_rate": {
            "avg_resting_hr": None,
            "min_resting_hr": None,
            "max_resting_hr": None,
            "resting_hr_trend": None,  # "improving" (decreasing), "declining" (increasing), "stable"
        },

        # Activity
        "activity": {
            "avg_daily_steps": None,
            "avg_active_calories": None,
            "avg_total_calories": None,
            "avg_activity_score": None,
            "sedentary_days_pct": None,  # % days with <5000 steps
            "high_activity_days_pct": None,  # % days with >10000 steps
            "days_analyzed": 0,
        },

        # Recovery / Readiness
        "recovery": {
            "avg_readiness_score": None,   # Oura readiness or Whoop recovery %
            "avg_recovery_score": None,
            "low_recovery_days_pct": None,  # % days with score < 33
            "high_recovery_days_pct": None, # % days with score > 66
        },

        # Stress
        "stress": {
            "avg_stress_score": None,
            "avg_body_battery_low": None,  # Garmin body battery minimum
            "high_stress_days_pct": None,
        },

        # Computed summary flags
        "flags": [],
        "raw_summary": {},
    }


# =============================================================================
# OURA PARSER
# =============================================================================

def parse_oura(files: dict) -> dict:
    """
    Parse Oura CSV exports.
    files: dict of {filename: file_content_string}
    Expected: sleep.csv, readiness.csv, activity.csv (any subset works)
    """
    metrics = empty_metrics()
    metrics["platform"] = "Oura Ring"

    all_dates = []

    # --- SLEEP ---
    sleep_file = next((c for n, c in files.items() if "sleep" in n.lower()), None)
    if sleep_file:
        sleep_data = []
        reader = csv.DictReader(io.StringIO(sleep_file))
        for row in reader:
            try:
                date = row.get("date") or row.get("Day") or row.get("Summary Date", "")
                total = float(row.get("Total Sleep Duration", row.get("total_sleep_duration", 0)) or 0)
                deep = float(row.get("Deep Sleep Duration", row.get("deep_sleep_duration", 0)) or 0)
                rem = float(row.get("REM Sleep Duration", row.get("rem_sleep_duration", 0)) or 0)
                light = float(row.get("Light Sleep Duration", row.get("light_sleep_duration", 0)) or 0)
                efficiency = float(row.get("Sleep Efficiency", row.get("efficiency", 0)) or 0)
                hrv = float(row.get("Average HRV", row.get("average_hrv", 0)) or 0)
                rhr = float(row.get("Lowest Resting Heart Rate", row.get("lowest_resting_heart_rate", 0)) or 0)
                bedtime = row.get("Bedtime Start", row.get("bedtime_start", ""))
                latency = float(row.get("Sleep Latency", row.get("sleep_latency", 0)) or 0)

                if total > 0:
                    sleep_data.append({
                        "date": date, "total": total / 3600, "deep": deep / 3600,
                        "rem": rem / 3600, "light": light / 3600,
                        "efficiency": efficiency, "hrv": hrv, "rhr": rhr,
                        "bedtime": bedtime, "latency": latency / 60
                    })
                    if date:
                        all_dates.append(date)
            except (ValueError, TypeError):
                continue

        if sleep_data:
            metrics["sleep"]["nights_analyzed"] = len(sleep_data)
            metrics["sleep"]["avg_total_hours"] = round(safe_mean(d["total"] for d in sleep_data) or 0, 2) or None
            metrics["sleep"]["avg_deep_hours"] = round(safe_mean(d["deep"] for d in sleep_data) or 0, 2) or None
            metrics["sleep"]["avg_rem_hours"] = round(safe_mean(d["rem"] for d in sleep_data) or 0, 2) or None
            metrics["sleep"]["avg_light_hours"] = round(safe_mean(d["light"] for d in sleep_data) or 0, 2) or None
            metrics["sleep"]["avg_efficiency_pct"] = round(safe_mean(d["efficiency"] for d in sleep_data if d["efficiency"] > 0) or 0, 1) or None
            metrics["sleep"]["avg_latency_min"] = round(safe_mean(d["latency"] for d in sleep_data if d["latency"] > 0) or 0, 1) or None

            hrv_vals = [d["hrv"] for d in sleep_data if d["hrv"] > 0]
            if hrv_vals:
                metrics["hrv"]["avg_rmssd"] = round(safe_mean(hrv_vals) or 0, 1) or None
                metrics["hrv"]["hrv_trend"] = _compute_trend(hrv_vals)

            rhr_vals = [d["rhr"] for d in sleep_data if d["rhr"] > 0]
            if rhr_vals:
                metrics["heart_rate"]["avg_resting_hr"] = round(safe_mean(rhr_vals) or 0, 1) or None
                metrics["heart_rate"]["min_resting_hr"] = min(rhr_vals)
                metrics["heart_rate"]["max_resting_hr"] = max(rhr_vals)
                metrics["heart_rate"]["resting_hr_trend"] = _compute_trend(rhr_vals, invert=True)

    # --- READINESS ---
    readiness_file = next((c for n, c in files.items() if "readiness" in n.lower()), None)
    if readiness_file:
        readiness_scores = []
        reader = csv.DictReader(io.StringIO(readiness_file))
        for row in reader:
            try:
                score = float(row.get("Score", row.get("readiness_score", 0)) or 0)
                if score > 0:
                    readiness_scores.append(score)
            except (ValueError, TypeError):
                continue

        if readiness_scores:
            metrics["recovery"]["avg_readiness_score"] = round(safe_mean(readiness_scores) or 0, 1) or None
            metrics["recovery"]["low_recovery_days_pct"] = round(
                sum(1 for s in readiness_scores if s < 60) / len(readiness_scores) * 100, 1)
            metrics["recovery"]["high_recovery_days_pct"] = round(
                sum(1 for s in readiness_scores if s >= 80) / len(readiness_scores) * 100, 1)

    # --- ACTIVITY ---
    activity_file = next((c for n, c in files.items() if "activity" in n.lower()), None)
    if activity_file:
        activity_data = []
        reader = csv.DictReader(io.StringIO(activity_file))
        for row in reader:
            try:
                steps = float(row.get("Steps", row.get("steps", 0)) or 0)
                active_cal = float(row.get("Active Calories", row.get("active_calories", 0)) or 0)
                score = float(row.get("Activity Score", row.get("activity_score", 0)) or 0)
                if steps >= 0:
                    activity_data.append({"steps": steps, "active_cal": active_cal, "score": score})
            except (ValueError, TypeError):
                continue

        if activity_data:
            metrics["activity"]["days_analyzed"] = len(activity_data)
            metrics["activity"]["avg_daily_steps"] = (lambda v: round(v) if v else None)(safe_mean(d["steps"] for d in activity_data))
            metrics["activity"]["avg_active_calories"] = (lambda v: round(v) if v else None)(safe_mean(d["active_cal"] for d in activity_data))
            metrics["activity"]["sedentary_days_pct"] = round(
                sum(1 for d in activity_data if d["steps"] < 5000) / len(activity_data) * 100, 1)
            metrics["activity"]["high_activity_days_pct"] = round(
                sum(1 for d in activity_data if d["steps"] >= 10000) / len(activity_data) * 100, 1)
            score_vals = [d["score"] for d in activity_data if d["score"] > 0]
            if score_vals:
                metrics["activity"]["avg_activity_score"] = round(safe_mean(score_vals) or 0, 1) or None

    _set_date_range(metrics, all_dates)
    _compute_flags(metrics)
    return metrics


# =============================================================================
# WHOOP PARSER
# =============================================================================

def parse_whoop(files: dict) -> dict:
    """
    Parse Whoop CSV exports.
    Expected files: sleep.csv, recovery.csv, workouts.csv
    """
    metrics = empty_metrics()
    metrics["platform"] = "Whoop"

    all_dates = []

    # --- SLEEP ---
    sleep_file = next((c for n, c in files.items() if "sleep" in n.lower()), None)
    if sleep_file:
        sleep_data = []
        reader = csv.DictReader(io.StringIO(sleep_file))
        for row in reader:
            try:
                date = row.get("Cycle start time", row.get("date", ""))[:10]
                total = float(row.get("Sleep duration (min)", row.get("total_sleep_minutes", 0)) or 0)
                deep = float(row.get("Slow wave sleep duration (min)", row.get("sws_minutes", 0)) or 0)
                rem = float(row.get("REM sleep duration (min)", row.get("rem_minutes", 0)) or 0)
                light = float(row.get("Light sleep duration (min)", row.get("light_sleep_minutes", 0)) or 0)
                efficiency = float(row.get("Sleep efficiency %", row.get("sleep_efficiency", 0)) or 0)
                hrv = float(row.get("Heart rate variability (ms)", row.get("hrv_rmssd_milli", 0)) or 0)
                rhr = float(row.get("Resting heart rate (bpm)", row.get("resting_heart_rate", 0)) or 0)
                latency = float(row.get("Sleep latency (min)", 0) or 0)

                if total > 0:
                    sleep_data.append({
                        "date": date, "total": total / 60, "deep": deep / 60,
                        "rem": rem / 60, "light": light / 60,
                        "efficiency": efficiency, "hrv": hrv, "rhr": rhr,
                        "latency": latency
                    })
                    if date:
                        all_dates.append(date)
            except (ValueError, TypeError):
                continue

        if sleep_data:
            metrics["sleep"]["nights_analyzed"] = len(sleep_data)
            metrics["sleep"]["avg_total_hours"] = round(safe_mean(d["total"] for d in sleep_data) or 0, 2) or None
            metrics["sleep"]["avg_deep_hours"] = round(safe_mean(d["deep"] for d in sleep_data) or 0, 2) or None
            metrics["sleep"]["avg_rem_hours"] = round(safe_mean(d["rem"] for d in sleep_data) or 0, 2) or None
            metrics["sleep"]["avg_light_hours"] = round(safe_mean(d["light"] for d in sleep_data) or 0, 2) or None

            eff_vals = [d["efficiency"] for d in sleep_data if d["efficiency"] > 0]
            if eff_vals:
                metrics["sleep"]["avg_efficiency_pct"] = round(safe_mean(eff_vals) or 0, 1) or None

            lat_vals = [d["latency"] for d in sleep_data if d["latency"] > 0]
            if lat_vals:
                metrics["sleep"]["avg_latency_min"] = round(safe_mean(lat_vals) or 0, 1) or None

            hrv_vals = [d["hrv"] for d in sleep_data if d["hrv"] > 0]
            if hrv_vals:
                metrics["hrv"]["avg_rmssd"] = round(safe_mean(hrv_vals) or 0, 1) or None
                metrics["hrv"]["hrv_trend"] = _compute_trend(hrv_vals)

            rhr_vals = [d["rhr"] for d in sleep_data if d["rhr"] > 0]
            if rhr_vals:
                metrics["heart_rate"]["avg_resting_hr"] = round(safe_mean(rhr_vals) or 0, 1) or None
                metrics["heart_rate"]["min_resting_hr"] = min(rhr_vals)
                metrics["heart_rate"]["max_resting_hr"] = max(rhr_vals)
                metrics["heart_rate"]["resting_hr_trend"] = _compute_trend(rhr_vals, invert=True)

    # --- RECOVERY ---
    recovery_file = next((c for n, c in files.items() if "recovery" in n.lower()), None)
    if recovery_file:
        recovery_scores = []
        reader = csv.DictReader(io.StringIO(recovery_file))
        for row in reader:
            try:
                score = float(row.get("Recovery score %", row.get("recovery_score", 0)) or 0)
                if score > 0:
                    recovery_scores.append(score)
                    date = row.get("Cycle start time", "")[:10]
                    if date:
                        all_dates.append(date)
            except (ValueError, TypeError):
                continue

        if recovery_scores:
            metrics["recovery"]["avg_recovery_score"] = round(safe_mean(recovery_scores) or 0, 1) or None
            metrics["recovery"]["avg_readiness_score"] = metrics["recovery"]["avg_recovery_score"]
            metrics["recovery"]["low_recovery_days_pct"] = round(
                sum(1 for s in recovery_scores if s < 33) / len(recovery_scores) * 100, 1)
            metrics["recovery"]["high_recovery_days_pct"] = round(
                sum(1 for s in recovery_scores if s >= 67) / len(recovery_scores) * 100, 1)

    # --- WORKOUTS (activity proxy) ---
    workout_file = next((c for n, c in files.items() if "workout" in n.lower()), None)
    if workout_file:
        strain_vals = []
        cal_vals = []
        reader = csv.DictReader(io.StringIO(workout_file))
        for row in reader:
            try:
                strain = float(row.get("Strain", row.get("strain_score", 0)) or 0)
                cals = float(row.get("Kilojoules", row.get("kilojoules", 0)) or 0) * 0.239
                if strain > 0:
                    strain_vals.append(strain)
                if cals > 0:
                    cal_vals.append(cals)
            except (ValueError, TypeError):
                continue

        if cal_vals:
            metrics["activity"]["avg_active_calories"] = (lambda v: round(v) if v else None)(safe_mean(cal_vals))
            metrics["activity"]["days_analyzed"] = len(cal_vals)

    _set_date_range(metrics, all_dates)
    _compute_flags(metrics)
    return metrics


# =============================================================================
# APPLE HEALTH PARSER
# Handles CSV exports from Health Auto Export or similar apps
# Also handles the native XML export format
# =============================================================================

def parse_apple_health(files: dict) -> dict:
    """
    Parse Apple Health CSV exports (from Health Auto Export app or similar).
    Expected files: sleep_analysis.csv, heart_rate.csv, steps.csv, hrv.csv, etc.
    Also handles single combined CSV or XML format.
    """
    metrics = empty_metrics()
    metrics["platform"] = "Apple Health"

    all_dates = []
    sleep_data = []
    hrv_vals = []
    rhr_vals = []
    step_vals = []
    cal_vals = []

    for fname, content in files.items():
        fname_lower = fname.lower()

        # HRV
        if "hrv" in fname_lower or "heart_rate_variability" in fname_lower:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                try:
                    val = float(row.get("HRV (ms)", row.get("value", row.get("HRV", 0))) or 0)
                    date = _extract_date(row)
                    if val > 0:
                        hrv_vals.append(val)
                        if date:
                            all_dates.append(date)
                except (ValueError, TypeError):
                    continue

        # Resting Heart Rate
        elif "resting" in fname_lower and "heart" in fname_lower:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                try:
                    val = float(row.get("Resting Heart Rate (bpm)", row.get("value", 0)) or 0)
                    date = _extract_date(row)
                    if val > 0:
                        rhr_vals.append(val)
                        if date:
                            all_dates.append(date)
                except (ValueError, TypeError):
                    continue

        # Steps
        elif "step" in fname_lower:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                try:
                    val = float(row.get("Steps (count)", row.get("value", row.get("Steps", 0))) or 0)
                    date = _extract_date(row)
                    if val > 0:
                        step_vals.append(val)
                        if date:
                            all_dates.append(date)
                except (ValueError, TypeError):
                    continue

        # Active Calories
        elif "active" in fname_lower and ("calori" in fname_lower or "energy" in fname_lower):
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                try:
                    val = float(row.get("Active Energy (kcal)", row.get("value", 0)) or 0)
                    if val > 0:
                        cal_vals.append(val)
                except (ValueError, TypeError):
                    continue

        # Sleep
        elif "sleep" in fname_lower:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                try:
                    stage = row.get("Value", row.get("sleep_stage", row.get("Stage", "")))
                    duration = float(row.get("Duration (min)", row.get("duration", 0)) or 0)
                    date = _extract_date(row)
                    if duration > 0 and date:
                        sleep_data.append({"stage": str(stage).lower(), "duration": duration, "date": date})
                        all_dates.append(date)
                except (ValueError, TypeError):
                    continue

    # Process sleep data (Apple Health stores individual sleep stages per entry)
    if sleep_data:
        sleep_by_date = defaultdict(lambda: {"total": 0, "deep": 0, "rem": 0, "light": 0, "awake": 0})
        for entry in sleep_data:
            d = entry["date"]
            stage = entry["stage"]
            dur = entry["duration"]
            if any(x in stage for x in ["deep", "slow", "n3"]):
                sleep_by_date[d]["deep"] += dur
            elif any(x in stage for x in ["rem", "r "]):
                sleep_by_date[d]["rem"] += dur
            elif any(x in stage for x in ["light", "core", "n1", "n2"]):
                sleep_by_date[d]["light"] += dur
            elif any(x in stage for x in ["awake", "wake", "in bed"]):
                sleep_by_date[d]["awake"] += dur
            else:
                sleep_by_date[d]["total"] += dur

        daily_sleep = list(sleep_by_date.values())
        if daily_sleep:
            metrics["sleep"]["nights_analyzed"] = len(daily_sleep)
            deep_vals = [d["deep"] / 60 for d in daily_sleep if d["deep"] > 0]
            rem_vals = [d["rem"] / 60 for d in daily_sleep if d["rem"] > 0]
            light_vals = [d["light"] / 60 for d in daily_sleep if d["light"] > 0]
            total_vals = [(d["deep"] + d["rem"] + d["light"]) / 60 for d in daily_sleep]

            if total_vals:
                metrics["sleep"]["avg_total_hours"] = round(safe_mean(total_vals) or 0, 2) or None
            if deep_vals:
                metrics["sleep"]["avg_deep_hours"] = round(safe_mean(deep_vals) or 0, 2) or None
            if rem_vals:
                metrics["sleep"]["avg_rem_hours"] = round(safe_mean(rem_vals) or 0, 2) or None
            if light_vals:
                metrics["sleep"]["avg_light_hours"] = round(safe_mean(light_vals) or 0, 2) or None

    if hrv_vals:
        metrics["hrv"]["avg_rmssd"] = round(safe_mean(hrv_vals) or 0, 1) or None
        metrics["hrv"]["hrv_trend"] = _compute_trend(hrv_vals)

    if rhr_vals:
        metrics["heart_rate"]["avg_resting_hr"] = round(safe_mean(rhr_vals) or 0, 1) or None
        metrics["heart_rate"]["min_resting_hr"] = min(rhr_vals)
        metrics["heart_rate"]["max_resting_hr"] = max(rhr_vals)
        metrics["heart_rate"]["resting_hr_trend"] = _compute_trend(rhr_vals, invert=True)

    if step_vals:
        metrics["activity"]["avg_daily_steps"] = (lambda v: round(v) if v else None)(safe_mean(step_vals))
        metrics["activity"]["days_analyzed"] = len(step_vals)
        metrics["activity"]["sedentary_days_pct"] = round(
            sum(1 for s in step_vals if s < 5000) / len(step_vals) * 100, 1)
        metrics["activity"]["high_activity_days_pct"] = round(
            sum(1 for s in step_vals if s >= 10000) / len(step_vals) * 100, 1)

    if cal_vals:
        metrics["activity"]["avg_active_calories"] = (lambda v: round(v) if v else None)(safe_mean(cal_vals))

    _set_date_range(metrics, all_dates)
    _compute_flags(metrics)
    return metrics


# =============================================================================
# AUTO-DETECT PLATFORM FROM FILES
# =============================================================================

def detect_platform(files: dict) -> str:
    """Detect wearable platform from file contents and names."""
    filenames = " ".join(files.keys()).lower()
    contents_sample = " ".join(list(files.values())[:2])[:500].lower()

    if "whoop" in filenames or "strain" in contents_sample or "recovery score %" in contents_sample:
        return "whoop"
    elif "readiness" in filenames or "oura" in filenames or "contributors" in contents_sample:
        return "oura"
    elif "hkquantitytypeidentifier" in contents_sample or "apple" in filenames or "health auto export" in contents_sample:
        return "apple"
    else:
        # Try to guess from column headers
        for content in files.values():
            first_line = content.split("\n")[0].lower()
            if "strain" in first_line or "recovery score" in first_line:
                return "whoop"
            elif "readiness" in first_line or "contributors" in first_line:
                return "oura"
            elif "hk" in first_line or "sourcename" in first_line:
                return "apple"
    return "unknown"


def parse_wearable(files: dict) -> dict:
    """
    Main entry point. Auto-detects platform and routes to correct parser.
    files: dict of {filename: file_content_string}
    """
    platform = detect_platform(files)
    print(f"  Detected wearable platform: {platform}")

    if platform == "oura":
        return parse_oura(files)
    elif platform == "whoop":
        return parse_whoop(files)
    elif platform == "apple":
        return parse_apple_health(files)
    else:
        # Try all parsers and return the one with most data
        results = [parse_oura(files), parse_whoop(files), parse_apple_health(files)]
        best = max(results, key=lambda r: r["sleep"]["nights_analyzed"] or 0)
        if best["sleep"]["nights_analyzed"] == 0:
            # Fall back to generic CSV parsing
            return parse_generic_csv(files)
        return best


def parse_generic_csv(files: dict) -> dict:
    """Fallback generic CSV parser — tries to extract common metric names."""
    metrics = empty_metrics()
    metrics["platform"] = "Unknown (Generic)"

    hrv_keys = ["hrv", "rmssd", "heart rate variability"]
    rhr_keys = ["resting heart rate", "resting hr", "rhr"]
    steps_keys = ["steps", "step count", "daily steps"]
    sleep_keys = ["sleep duration", "total sleep", "sleep hours"]

    for fname, content in files.items():
        try:
            reader = csv.DictReader(io.StringIO(content))
            headers = [h.lower() for h in (reader.fieldnames or [])]

            hrv_col = next((h for h in headers if any(k in h for k in hrv_keys)), None)
            rhr_col = next((h for h in headers if any(k in h for k in rhr_keys)), None)
            steps_col = next((h for h in headers if any(k in h for k in steps_keys)), None)
            sleep_col = next((h for h in headers if any(k in h for k in sleep_keys)), None)

            hrv_vals, rhr_vals, step_vals, sleep_vals = [], [], [], []

            for row in reader:
                row_lower = {k.lower(): v for k, v in row.items()}
                try:
                    if hrv_col and row_lower.get(hrv_col):
                        hrv_vals.append(float(row_lower[hrv_col]))
                    if rhr_col and row_lower.get(rhr_col):
                        rhr_vals.append(float(row_lower[rhr_col]))
                    if steps_col and row_lower.get(steps_col):
                        step_vals.append(float(row_lower[steps_col]))
                    if sleep_col and row_lower.get(sleep_col):
                        sleep_vals.append(float(row_lower[sleep_col]))
                except (ValueError, TypeError):
                    continue

            if hrv_vals:
                metrics["hrv"]["avg_rmssd"] = round(safe_mean(hrv_vals) or 0, 1) or None
            if rhr_vals:
                metrics["heart_rate"]["avg_resting_hr"] = round(safe_mean(rhr_vals) or 0, 1) or None
            if step_vals:
                metrics["activity"]["avg_daily_steps"] = (lambda v: round(v) if v else None)(safe_mean(step_vals))
            if sleep_vals:
                metrics["sleep"]["avg_total_hours"] = round(safe_mean(sleep_vals) or 0, 2) or None

        except Exception:
            continue

    _compute_flags(metrics)
    return metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_date(row: dict) -> str:
    """Try to extract a date string from a row."""
    for key in ["Date", "date", "Start Date", "startDate", "End Date", "Cycle start time",
                "Timestamp", "timestamp", "Day", "day"]:
        val = row.get(key, "")
        if val and len(str(val)) >= 10:
            return str(val)[:10]
    return ""


def _compute_trend(values: list, invert: bool = False, window: int = 7) -> str:
    """
    Compute trend from a time-ordered list of values.
    invert=True means lower is better (e.g. resting HR).
    Returns "improving", "declining", or "stable".
    """
    if len(values) < window * 2:
        return "stable"
    first_half = safe_mean(values[:len(values)//2])
    second_half = safe_mean(values[len(values)//2:])
    pct_change = (second_half - first_half) / first_half * 100 if first_half != 0 else 0

    threshold = 3.0  # % change required to call a trend
    if abs(pct_change) < threshold:
        return "stable"
    if invert:
        return "improving" if pct_change < 0 else "declining"
    else:
        return "improving" if pct_change > 0 else "declining"


def _set_date_range(metrics: dict, all_dates: list):
    """Set date range from collected dates."""
    valid_dates = [d for d in all_dates if d and len(d) >= 8]
    if valid_dates:
        valid_dates.sort()
        metrics["date_range"]["start"] = valid_dates[0]
        metrics["date_range"]["end"] = valid_dates[-1]
        try:
            start = datetime.strptime(valid_dates[0][:10], "%Y-%m-%d")
            end = datetime.strptime(valid_dates[-1][:10], "%Y-%m-%d")
            metrics["date_range"]["days"] = (end - start).days + 1
        except ValueError:
            metrics["date_range"]["days"] = len(set(valid_dates))


def _compute_flags(metrics: dict):
    """Generate summary flags based on computed metrics."""
    flags = []

    sleep = metrics["sleep"]
    hrv = metrics["hrv"]
    hr = metrics["heart_rate"]
    activity = metrics["activity"]
    recovery = metrics["recovery"]

    # Sleep flags
    if sleep["avg_total_hours"] and sleep["avg_total_hours"] < 7.0:
        flags.append({
            "type": "warning",
            "metric": "Sleep Duration",
            "message": f"Average sleep of {sleep['avg_total_hours']}h is below the 7-9h optimal range — one of the most impactful modifiable health factors"
        })
    if sleep["avg_deep_hours"] and sleep["avg_deep_hours"] < 1.0:
        flags.append({
            "type": "warning",
            "metric": "Deep Sleep",
            "message": f"Average deep sleep of {sleep['avg_deep_hours']}h is low — deep sleep drives growth hormone release, physical repair, and immune function"
        })
    if sleep["avg_rem_hours"] and sleep["avg_rem_hours"] < 1.5:
        flags.append({
            "type": "warning",
            "metric": "REM Sleep",
            "message": f"Average REM of {sleep['avg_rem_hours']}h is below optimal — REM drives memory consolidation, emotional regulation, and BDNF production"
        })
    if sleep["avg_efficiency_pct"] and sleep["avg_efficiency_pct"] < 85:
        flags.append({
            "type": "warning",
            "metric": "Sleep Efficiency",
            "message": f"Sleep efficiency of {sleep['avg_efficiency_pct']}% is below optimal — time in bed is not converting to restorative sleep"
        })
    if sleep["avg_latency_min"] and sleep["avg_latency_min"] > 20:
        flags.append({
            "type": "insight",
            "metric": "Sleep Latency",
            "message": f"Average sleep latency of {sleep['avg_latency_min']} min is elevated — may reflect elevated cortisol, anxiety, or late light exposure"
        })

    # HRV flags
    if hrv["avg_rmssd"] and hrv["avg_rmssd"] < 30:
        flags.append({
            "type": "warning",
            "metric": "HRV",
            "message": f"Average HRV (RMSSD) of {hrv['avg_rmssd']} ms is low — reflects reduced autonomic nervous system resilience and recovery capacity"
        })
    if hrv["hrv_trend"] == "declining":
        flags.append({
            "type": "warning",
            "metric": "HRV Trend",
            "message": "HRV is trending downward over the analysis period — accumulating physiological stress or overtraining"
        })

    # Resting HR flags
    if hr["avg_resting_hr"] and hr["avg_resting_hr"] > 70:
        flags.append({
            "type": "warning",
            "metric": "Resting Heart Rate",
            "message": f"Average resting HR of {hr['avg_resting_hr']} bpm is above optimal (<60 bpm for cardiovascular fitness) — reflects lower aerobic capacity"
        })
    if hr["resting_hr_trend"] == "declining":
        flags.append({
            "type": "warning",
            "metric": "Resting HR Trend",
            "message": "Resting heart rate is trending upward — may indicate overtraining, illness, or increasing cardiovascular stress"
        })

    # Activity flags
    if activity["avg_daily_steps"] and activity["avg_daily_steps"] < 7000:
        flags.append({
            "type": "warning",
            "metric": "Daily Steps",
            "message": f"Average of {activity['avg_daily_steps']:,} steps/day is below the 7,500-10,000 step target associated with all-cause mortality reduction"
        })
    if activity["sedentary_days_pct"] and activity["sedentary_days_pct"] > 30:
        flags.append({
            "type": "warning",
            "metric": "Sedentary Days",
            "message": f"{activity['sedentary_days_pct']}% of days had fewer than 5,000 steps — prolonged sedentary behavior is an independent cardiovascular risk factor"
        })

    # Recovery flags
    readiness = recovery.get("avg_readiness_score") or recovery.get("avg_recovery_score")
    if readiness and readiness < 65:
        flags.append({
            "type": "warning",
            "metric": "Recovery Score",
            "message": f"Average recovery/readiness score of {readiness} is below optimal — body is not fully recovering between days"
        })

    metrics["flags"] = flags


# =============================================================================
# WEARABLE SUMMARY FOR AI ANALYSIS
# =============================================================================

def build_wearable_summary(metrics: dict) -> str:
    """Build a structured text summary of wearable metrics for AI analysis."""
    lines = [f"WEARABLE DATA SUMMARY ({metrics['platform']})"]

    dr = metrics["date_range"]
    if dr["start"] and dr["end"]:
        lines.append(f"Period: {dr['start']} to {dr['end']} ({dr['days']} days)")

    s = metrics["sleep"]
    if s["nights_analyzed"] > 0:
        lines.append(f"\nSLEEP ({s['nights_analyzed']} nights):")
        if s["avg_total_hours"]:
            lines.append(f"  Total sleep: {s['avg_total_hours']}h avg (optimal: 7.5-9h)")
        if s["avg_deep_hours"]:
            lines.append(f"  Deep sleep: {s['avg_deep_hours']}h avg (optimal: >1.5h)")
        if s["avg_rem_hours"]:
            lines.append(f"  REM sleep: {s['avg_rem_hours']}h avg (optimal: >1.5h)")
        if s["avg_efficiency_pct"]:
            lines.append(f"  Sleep efficiency: {s['avg_efficiency_pct']}% (optimal: >85%)")
        if s["avg_latency_min"]:
            lines.append(f"  Sleep latency: {s['avg_latency_min']} min (optimal: 10-20 min)")

    h = metrics["hrv"]
    if h["avg_rmssd"]:
        lines.append(f"\nHRV:")
        lines.append(f"  Average RMSSD: {h['avg_rmssd']} ms (optimal varies by age; higher is better)")
        if h["hrv_trend"]:
            lines.append(f"  Trend: {h['hrv_trend']}")

    hr = metrics["heart_rate"]
    if hr["avg_resting_hr"]:
        lines.append(f"\nHEART RATE:")
        lines.append(f"  Resting HR: {hr['avg_resting_hr']} bpm (optimal: <60 bpm)")
        if hr["resting_hr_trend"]:
            lines.append(f"  Trend: {hr['resting_hr_trend']}")

    a = metrics["activity"]
    if a["days_analyzed"] > 0:
        lines.append(f"\nACTIVITY ({a['days_analyzed']} days):")
        if a["avg_daily_steps"]:
            lines.append(f"  Daily steps: {a['avg_daily_steps']:,} avg (optimal: 7,500-10,000+)")
        if a["avg_active_calories"]:
            lines.append(f"  Active calories: {a['avg_active_calories']} avg/day")
        if a["sedentary_days_pct"] is not None:
            lines.append(f"  Sedentary days (<5k steps): {a['sedentary_days_pct']}%")
        if a["high_activity_days_pct"] is not None:
            lines.append(f"  High activity days (>10k steps): {a['high_activity_days_pct']}%")

    r = metrics["recovery"]
    readiness = r.get("avg_readiness_score") or r.get("avg_recovery_score")
    if readiness:
        lines.append(f"\nRECOVERY:")
        lines.append(f"  Avg readiness/recovery score: {readiness} (optimal: >70)")
        if r["low_recovery_days_pct"] is not None:
            lines.append(f"  Low recovery days: {r['low_recovery_days_pct']}%")
        if r["high_recovery_days_pct"] is not None:
            lines.append(f"  High recovery days: {r['high_recovery_days_pct']}%")

    if metrics["flags"]:
        lines.append(f"\nFLAGGED FINDINGS ({len(metrics['flags'])}):")
        for flag in metrics["flags"]:
            prefix = "⚠" if flag["type"] == "warning" else "→"
            lines.append(f"  {prefix} {flag['metric']}: {flag['message']}")

    return "\n".join(lines)
