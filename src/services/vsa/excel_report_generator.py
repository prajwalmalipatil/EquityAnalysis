"""
excel_report_generator.py
Generates professional styled multi-sheet Excel reports for VSA results.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from .formatters import ExcelFormatter


class VSAExcelReportGenerator:
    """Generates professional styled multi-sheet Excel reports for VSA results."""

    @staticmethod
    def generate(
        df: pd.DataFrame, file_path: Path, out_path: Path, conf: int, fail: int
    ) -> None:
        """Writes dataframes to sheets and applies standard formatting and styling."""
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)

            # Signal Summary sheet
            sig_counts = df["Signal_Type"].value_counts().reset_index()
            sig_counts.columns = ["Signal", "Count"]
            sig_counts.to_excel(writer, sheet_name="Signal_Summary", index=False)

            # Processing Log sheet
            log_df = pd.DataFrame(
                [
                    {"Artifact": "Processing Engine", "Value": "V² Money V2.1 Prod"},
                    {"Artifact": "Analysis Timestamp", "Value": datetime.now().isoformat()},
                    {"Artifact": "Source File", "Value": str(file_path.name)},
                    {"Artifact": "Rows Analyzed", "Value": len(df)},
                    {"Artifact": "Confirmed Signals", "Value": conf},
                    {"Artifact": "Failed Signals", "Value": fail},
                ]
            )
            log_df.to_excel(writer, sheet_name="Processing_Log", index=False)

            # Indicator Meta sheet
            meta_df = pd.DataFrame(
                [
                    {"Indicator": "Volume_MA", "Description": "20-period volume average"},
                    {"Indicator": "Spread", "Description": "High - Low"},
                    {"Indicator": "Fire", "Description": "Volume-spread anomaly (🔥)"},
                ]
            )
            meta_df.to_excel(writer, sheet_name="Indicator_Meta", index=False)

        # Apply UI Styling
        wb = load_workbook(out_path)
        ExcelFormatter.apply_standard_styling(wb, "VSA_Analysis")
        ExcelFormatter.add_vsa_legend(wb)
        wb.save(out_path)
