"""
HealthBridge AI - Interactive Visualization Module
Generates charts and visualizations for biomarkers and health data
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BiomarkerPoint:
    """Single biomarker measurement"""
    date: str
    value: float
    unit: str
    confidence: float = 100.0
    conditions: Optional[Dict] = None


class ChartGenerator:
    """Generate interactive charts for health data"""

    # Optimal ranges from blood_parser
    OPTIMAL_RANGES = {
        "glucose": (72, 85, "mg/dL"),
        "hba1c": (4.6, 5.3, "%"),
        "insulin": (2, 6, "uIU/mL"),
        "ldl": (40, 70, "mg/dL"),
        "hdl": (60, 90, "mg/dL"),
        "triglycerides": (50, 100, "mg/dL"),
        "hscrp": (0.1, 1.0, "mg/L"),
        "homocysteine": (6, 8, "µmol/L"),
        "vitamin_d": (60, 80, "ng/mL"),
        "apob": (40, 70, "mg/dL"),
        "lpa": (0, 30, "mg/dL"),
    }

    # Standard reference ranges
    STANDARD_RANGES = {
        "glucose": (70, 99, "mg/dL"),
        "hba1c": (0, 5.6, "%"),
        "insulin": (2, 25, "uIU/mL"),
        "ldl": (0, 99, "mg/dL"),
        "hdl": (40, 1000, "mg/dL"),
        "triglycerides": (0, 149, "mg/dL"),
        "hscrp": (0, 3.0, "mg/L"),
        "homocysteine": (0, 15, "µmol/L"),
        "vitamin_d": (30, 100, "ng/mL"),
        "apob": (0, 90, "mg/dL"),
    }

    @staticmethod
    def generate_biomarker_gauge(biomarker: str, value: float,
                                   optimal_range: Tuple[float, float],
                                   standard_range: Tuple[float, float]) -> Dict:
        """Generate gauge chart configuration"""
        opt_low, opt_high = optimal_range
        std_low, std_high = standard_range

        # Calculate normalized position (0-100)
        if value < std_low:
            normalized = 0
        elif value > std_high:
            normalized = 100
        else:
            normalized = ((value - std_low) / (std_high - std_low)) * 100

        # Determine color
        if opt_low <= value <= opt_high:
            color = "#059669"  # Green - optimal
            status = "Optimal"
        elif std_low <= value <= std_high:
            color = "#f59e0b"  # Yellow - suboptimal
            status = "Suboptimal"
        else:
            color = "#dc2626"  # Red - out of range
            status = "Needs Attention"

        return {
            "type": "gauge",
            "data": {
                "value": round(value, 1),
                "min": std_low,
                "max": std_high * 1.2,
                "optimal_low": opt_low,
                "optimal_high": opt_high,
                "unit": ""
            },
            "config": {
                "normalized_position": round(normalized, 1),
                "color": color,
                "status": status,
                "segments": [
                    {"range": [std_low, opt_low], "color": "#fbbf24", "label": "Below Optimal"},
                    {"range": [opt_low, opt_high], "color": "#10b981", "label": "Optimal"},
                    {"range": [opt_high, std_high], "color": "#fbbf24", "label": "Above Optimal"},
                ]
            }
        }

    @staticmethod
    def generate_trend_chart(biomarker: str, data_points: List[BiomarkerPoint],
                              show_optimal: bool = True) -> Dict:
        """Generate trend chart with optimal range overlay"""
        if not data_points:
            return {"error": "No data points provided"}

        dates = [p.date for p in data_points]
        values = [p.value for p in data_points]

        # Get range info
        optimal = ChartGenerator.OPTIMAL_RANGES.get(biomarker.lower())
        standard = ChartGenerator.STANDARD_RANGES.get(biomarker.lower())

        chart_config = {
            "type": "line",
            "biomarker": biomarker,
            "data": {
                "dates": dates,
                "values": values,
                "unit": data_points[0].unit if data_points else ""
            },
            "annotations": []
        }

        if optimal and show_optimal:
            opt_low, opt_high, unit = optimal
            chart_config["annotations"].extend([
                {
                    "type": "range",
                    "y_min": opt_low,
                    "y_max": opt_high,
                    "label": "Optimal Range",
                    "color": "rgba(16, 185, 129, 0.2)"
                },
                {
                    "type": "line",
                    "y": opt_low,
                    "label": f"Optimal: {opt_low}",
                    "color": "#059669",
                    "style": "dashed"
                },
                {
                    "type": "line",
                    "y": opt_high,
                    "label": f"Optimal: {opt_high}",
                    "color": "#059669",
                    "style": "dashed"
                }
            ])

        if standard:
            std_low, std_high, _ = standard
            chart_config["annotations"].append({
                "type": "line",
                "y": std_high,
                "label": f"Standard Max: {std_high}",
                "color": "#dc2626",
                "style": "dotted"
            })

        return chart_config

    @staticmethod
    def generate_snp_heatmap(snp_data: List[Dict]) -> Dict:
        """Generate heatmap of SNP risk across categories"""
        categories = {}

        for snp in snp_data:
            cat = snp.get("category", "Unknown")
            zygosity = snp.get("zygosity", "unknown")

            if cat not in categories:
                categories[cat] = {"risk_count": 0, "total": 0, "snps": []}

            categories[cat]["total"] += 1
            categories[cat]["snps"].append(snp.get("rsid", ""))

            if zygosity == "homozygous_risk":
                categories[cat]["risk_count"] += 2
            elif zygosity == "heterozygous":
                categories[cat]["risk_count"] += 1

        # Calculate risk scores
        heatmap_data = []
        for cat, data in categories.items():
            risk_score = data["risk_count"] / (data["total"] * 2) * 100 if data["total"] > 0 else 0

            # Determine color
            if risk_score >= 40:
                color = "#dc2626"
            elif risk_score >= 25:
                color = "#f59e0b"
            elif risk_score >= 10:
                color = "#3b82f6"
            else:
                color = "#10b981"

            heatmap_data.append({
                "category": cat,
                "risk_score": round(risk_score, 1),
                "color": color,
                "variant_count": data["total"],
                "snps": data["snps"]
            })

        # Sort by risk score
        heatmap_data.sort(key=lambda x: x["risk_score"], reverse=True)

        return {
            "type": "heatmap",
            "title": "Genetic Risk Distribution",
            "data": heatmap_data
        }

    @staticmethod
    def generate_composite_dashboard(biomarkers: Dict[str, float],
                                      snp_categories: Dict[str, int],
                                      wearable_metrics: Optional[Dict] = None) -> Dict:
        """Generate a composite health dashboard"""
        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "summary_score": 0,
            "sections": []
        }

        # Calculate health scores
        scores = {}

        # Metabolic score
        metabolic_markers = ["glucose", "hba1c", "insulin"]
        metabolic_score = ChartGenerator._calculate_category_score(
            biomarkers, metabolic_markers
        )
        scores["metabolic"] = metabolic_score

        # Cardiovascular score
        cardio_markers = ["ldl", "hdl", "triglycerides", "hscrp"]
        cardio_score = ChartGenerator._calculate_category_score(
            biomarkers, cardio_markers
        )
        scores["cardiovascular"] = cardio_score

        # Overall score
        dashboard["summary_score"] = round(
            sum(scores.values()) / len(scores), 1
        ) if scores else 50

        # Add sections
        for category, score in scores.items():
            dashboard["sections"].append({
                "category": category.title(),
                "score": score,
                "status": ChartGenerator._score_to_status(score),
                "color": ChartGenerator._score_to_color(score)
            })

        return dashboard

    @staticmethod
    def _calculate_category_score(biomarkers: Dict, relevant: List[str]) -> float:
        """Calculate category health score"""
        scores = []
        for marker in relevant:
            if marker in biomarkers:
                value = biomarkers[marker]
                optimal = ChartGenerator.OPTIMAL_RANGES.get(marker)
                if optimal:
                    opt_low, opt_high, _ = optimal
                    if opt_low <= value <= opt_high:
                        scores.append(100)
                    else:
                        # Calculate distance from optimal
                        distance = min(abs(value - opt_low), abs(value - opt_high))
                        penalty = min(50, distance / opt_low * 50)
                        scores.append(100 - penalty)

        return round(sum(scores) / len(scores), 1) if scores else 50

    @staticmethod
    def _score_to_status(score: float) -> str:
        if score >= 80:
            return "Optimal"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Needs Attention"

    @staticmethod
    def _score_to_color(score: float) -> str:
        if score >= 80:
            return "#10b981"
        elif score >= 60:
            return "#3b82f6"
        elif score >= 40:
            return "#f59e0b"
        else:
            return "#dc2626"


class HTMLChartRenderer:
    """Render charts to HTML using Chart.js"""

    CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"

    @staticmethod
    def render_gauge_html(biomarker: str, value: float, config: Dict) -> str:
        """Render a gauge/meter visualization"""
        status = config.get("status", "Unknown")
        color = config.get("color", "#6b7280")
        normalized = config.get("normalized_position", 50)

        return f"""
        <div class="biomarker-gauge" style="margin: 20px 0; padding: 20px; background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0; font-size: 1.1em; color: #1f2937;">{biomarker.title()}</h4>
                <span style="font-size: 1.5em; font-weight: 700; color: {color};">{value}</span>
            </div>

            <div style="position: relative; height: 12px; background: linear-gradient(to right, #dc2626 0%, #fbbf24 30%, #10b981 40%, #10b981 60%, #fbbf24 70%, #dc2626 100%); border-radius: 6px; margin: 15px 0;">
                <div style="position: absolute; left: {normalized}%; top: -4px; width: 4px; height: 20px; background: #1f2937; border-radius: 2px; transform: translateX(-50%); box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
            </div>

            <div style="display: flex; justify-content: space-between; font-size: 0.75em; color: #6b7280; margin-top: 8px;">
                <span>Standard Range</span>
                <span style="color: {color}; font-weight: 600;">{status}</span>
                <span>Standard Range</span>
            </div>

            <div style="margin-top: 12px; padding: 10px; background: #f9fafb; border-radius: 8px; font-size: 0.85em;">
                <div style="display: flex; gap: 20px;">
                    <div>
                        <span style="color: #6b7280;">Optimal:</span>
                        <span style="font-weight: 600; color: #059669;">{' - '.join(map(str, config.get('optimal_range', ['?', '?'])))}</span>
                    </div>
                    <div>
                        <span style="color: #6b7280;">Your value:</span>
                        <span style="font-weight: 600; color: {color};">{value}</span>
                    </div>
                </div>
            </div>
        </div>
        """

    @staticmethod
    def render_dashboard_html(dashboard: Dict) -> str:
        """Render composite dashboard"""
        overall = dashboard.get("summary_score", 50)
        sections = dashboard.get("sections", [])

        # Overall score color
        if overall >= 80:
            overall_color = "#10b981"
            overall_status = "Excellent"
        elif overall >= 60:
            overall_color = "#3b82f6"
            overall_status = "Good"
        elif overall >= 40:
            overall_color = "#f59e0b"
            overall_status = "Fair"
        else:
            overall_color = "#dc2626"
            overall_status = "Needs Attention"

        sections_html = ""
        for section in sections:
            sections_html += f"""
            <div style="flex: 1; min-width: 200px; padding: 20px; background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 0.85em; color: #6b7280; margin-bottom: 8px;">{section['category']}</div>
                <div style="font-size: 2em; font-weight: 700; color: {section['color']}; margin-bottom: 4px;">{section['score']}</div>
                <div style="font-size: 0.8em; color: {section['color']};">{section['status']}</div>
                <div style="margin-top: 10px; height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden;">
                    <div style="width: {section['score']}%; height: 100%; background: {section['color']}; transition: width 0.5s ease;"></div>
                </div>
            </div>
            """

        return f"""
        <div class="health-dashboard" style="margin: 30px 0;">
            <div style="text-align: center; margin-bottom: 30px; padding: 30px; background: linear-gradient(135deg, {overall_color}20, {overall_color}05); border-radius: 16px; border: 2px solid {overall_color}40;">
                <div style="font-size: 0.9em; color: {overall_color}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Overall Health Score</div>
                <div style="font-size: 4em; font-weight: 800; color: {overall_color}; line-height: 1;">{overall}</div>
                <div style="font-size: 1.2em; color: {overall_color}; margin-top: 8px; font-weight: 600;">{overall_status}</div>
            </div>

            <div style="display: flex; flex-wrap: wrap; gap: 16px;">
                {sections_html}
            </div>
        </div>
        """


if __name__ == "__main__":
    # Test the module
    print("Testing Visualization Module...")

    # Test gauge generation
    gauge = ChartGenerator.generate_biomarker_gauge(
        "glucose", 95, (72, 85), (70, 99)
    )
    print(f"\nGlucose gauge:")
    print(f"  Status: {gauge['config']['status']}")
    print(f"  Color: {gauge['config']['color']}")
    print(f"  Normalized position: {gauge['config']['normalized_position']}")

    # Test trend chart
    points = [
        BiomarkerPoint("2024-01", 88, "mg/dL"),
        BiomarkerPoint("2024-04", 92, "mg/dL"),
        BiomarkerPoint("2024-07", 95, "mg/dL"),
    ]
    trend = ChartGenerator.generate_trend_chart("glucose", points)
    print(f"\nTrend chart has {len(trend['data']['values'])} points")

    # Test dashboard
    dashboard = ChartGenerator.generate_composite_dashboard(
        {"glucose": 88, "hba1c": 5.4, "ldl": 85},
        {"Metabolic": 2, "Cardiovascular": 1}
    )
    print(f"\nDashboard overall score: {dashboard['summary_score']}")

    print("\nVisualization Module test complete.")
