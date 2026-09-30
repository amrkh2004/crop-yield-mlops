"""
Data Drift Detection Module using Scipy Kolmogorov-Smirnov tests and Evidently AI.
Monitors feature distribution shifts between reference (training) and current (inference) datasets.
"""

import json
import os

import pandas as pd
import scipy.stats as stats
from prodml.logging import get_logger
from prodml.metrics import DATA_DRIFT_SCORE, DRIFT_DETECTED

logger = get_logger("prodml.drift_detector")


class DataDriftDetector:
    """
    Data Drift Detector for tabular crop yield features.
    Computes Kolmogorov-Smirnov distribution tests for continuous features
    and frequency distribution distance for categorical features.
    """

    NUMERIC_FEATURES = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]
    CATEGORICAL_FEATURES = ["Area", "Item"]

    def __init__(self, p_value_threshold: float = 0.05):
        """
        Initialize drift detector with significance threshold p-value.
        """
        self.p_value_threshold = p_value_threshold

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        export_json_path: str = "reports/drift_report.json",
    ) -> dict:
        """
        Detect distribution drift across numeric and categorical features.

        :param reference_df: Reference baseline DataFrame (e.g., training data).
        :param current_df: Current operational/inference DataFrame.
        :param export_json_path: Path to save JSON report.
        :return: Dict summary of drift analysis.
        """
        feature_results = {}
        drift_count = 0
        total_features = 0
        stat_scores = []

        # 1. Numerical Features Drift (Kolmogorov-Smirnov Test)
        for col in self.NUMERIC_FEATURES:
            if col in reference_df.columns and col in current_df.columns:
                total_features += 1
                ref_data = reference_df[col].dropna()
                cur_data = current_df[col].dropna()

                stat, p_val = stats.ks_2samp(ref_data, cur_data)
                stat_scores.append(float(stat))
                is_drift = bool(p_val < self.p_value_threshold)
                if is_drift:
                    drift_count += 1

                feature_results[col] = {
                    "type": "numerical",
                    "test": "kolmogorov_smirnov",
                    "ks_statistic": float(stat),
                    "p_value": float(p_val),
                    "drift_detected": is_drift,
                }

        # 2. Categorical Features Drift (Frequency Distance / Chi-Square)
        for col in self.CATEGORICAL_FEATURES:
            if col in reference_df.columns and col in current_df.columns:
                total_features += 1
                ref_counts = reference_df[col].value_counts(normalize=True)
                cur_counts = current_df[col].value_counts(normalize=True)

                # Total Variation Distance (TVD)
                all_cats = set(ref_counts.index).union(set(cur_counts.index))
                tvd = (
                    0.5
                    * sum(
                        abs(ref_counts.get(cat, 0.0) - cur_counts.get(cat, 0.0))
                        for cat in all_cats
                    )
                )
                stat_scores.append(float(tvd))
                is_drift = bool(tvd > 0.2)  # Threshold for TVD drift
                if is_drift:
                    drift_count += 1

                feature_results[col] = {
                    "type": "categorical",
                    "test": "total_variation_distance",
                    "tvd_statistic": float(tvd),
                    "drift_detected": is_drift,
                }

        overall_drift_score = (
            sum(stat_scores) / len(stat_scores) if stat_scores else 0.0
        )
        overall_drift_detected = drift_count > 0

        report = {
            "drift_detected": overall_drift_detected,
            "overall_drift_score": round(overall_drift_score, 4),
            "drifted_features_count": drift_count,
            "total_features_evaluated": total_features,
            "p_value_threshold": self.p_value_threshold,
            "retraining_recommended": overall_drift_detected,
            "feature_metrics": feature_results,
        }

        # Update Prometheus Gauges
        DATA_DRIFT_SCORE.set(overall_drift_score)
        DRIFT_DETECTED.set(1.0 if overall_drift_detected else 0.0)

        # Save JSON Report if output path specified
        if export_json_path:
            os.makedirs(os.path.dirname(export_json_path), exist_ok=True)
            with open(export_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("drift_report_exported", path=export_json_path)

        return report

    def generate_evidently_html_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        export_html_path: str = "reports/drift_report.html",
    ) -> bool:
        """
        Generates interactive HTML Evidently AI Data Drift report if installed.
        """
        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=reference_df, current_data=current_df)

            os.makedirs(os.path.dirname(export_html_path), exist_ok=True)
            drift_report.save_html(export_html_path)
            logger.info("evidently_html_exported", path=export_html_path)
            return True
        except Exception as e:
            logger.warning(
                "evidently_report_fallback",
                error=str(e),
                message="Falling back to standard HTML summary.",
            )
            # Create a clean fallback HTML file
            os.makedirs(os.path.dirname(export_html_path), exist_ok=True)
            drift_summary = self.detect_drift(
                reference_df, current_df, export_json_path=None
            )
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Data Drift Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; background-color: #0f172a; color: #f8fafc; }}
                h1 {{ color: #38bdf8; }}
                .card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .status-pass {{ color: #4ade80; font-weight: bold; }}
                .status-drift {{ color: #f87171; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ border: 1px solid #334155; padding: 8px 12px; text-align: left; }}
                th {{ background: #334155; }}
            </style>
            </head>
            <body>
                <h1>Crop Yield Model - Data Drift Monitoring Report</h1>
                <div class="card">
                    <h2>Overall Status</h2>
                    <p>Drift Detected: <span class="{ 'status-drift' if drift_summary['drift_detected'] else 'status-pass' }">
                        { drift_summary['drift_detected'] }</span></p>
                    <p>Overall Drift Score: { drift_summary['overall_drift_score'] }</p>
                    <p>Retraining Recommended: { drift_summary['retraining_recommended'] }</p>
                </div>
            </body>
            </html>
            """
            with open(export_html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return False
