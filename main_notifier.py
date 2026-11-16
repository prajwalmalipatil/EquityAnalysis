"""
main_notifier.py
Sends the daily Gap Momentum Analysis email.

The email carries the Gap Momentum signals of the latest session and the Gap-Up
Rejection watch (failed gap-up follow-through, tracked for ten sessions), read from
the `gap_momentum_filters` and `gap_up_rejection_filters` blocks of dashboard/data.json.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.clients.smtp_client import SMTPClient
from src.services.reporting.gap_momentum_email_renderer import (
    render_gap_momentum_html,
    render_gap_momentum_text,
)
from src.services.reporting.gap_up_rejection_email_renderer import (
    count_new,
    render_gap_up_rejection_html,
    render_gap_up_rejection_text,
)
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("notifier-main")

DATA_PATH = "dashboard/data.json"
PREVIEW_PATH = "local_report_preview.html"
EMAIL_TITLE = "Gap Momentum Analysis"
UNAVAILABLE_MESSAGE = "Gap Momentum data was not produced by this pipeline run."
FOOTER = "Generated automatically by V² Money Automation"


def load_dashboard_data(path: str = DATA_PATH) -> Dict:
    """Reads data.json once; returns {} when it is missing or unreadable."""
    try:
        data_path = Path(path)
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("FAILED_TO_LOAD_DATA_JSON", extra={"error": str(e)})
    return {}


def session_date(filters: Optional[Dict], fallback: str) -> str:
    """The trading day the signals belong to; the run date when there are no signals."""
    items = (filters or {}).get("daily", [])
    dates = sorted(str(i["signal_date"]) for i in items if isinstance(i, dict) and i.get("signal_date"))
    return dates[-1] if dates else fallback


def build_email(data: Dict, run_date: str) -> Tuple[str, str, str]:
    """Returns (subject, html_body, text_body) for the Gap Momentum email."""
    filters = data.get("gap_momentum_filters")
    count = len((filters or {}).get("daily", []))
    session = session_date(filters, run_date)

    section_html = render_gap_momentum_html(filters) or (
        f'<p style="color: #fbbf24; text-align: center; font-size: 14px; margin: 30px 0;">{UNAVAILABLE_MESSAGE}</p>')
    section_text = render_gap_momentum_text(filters) or f"\n{UNAVAILABLE_MESSAGE}\n"

    # Gap-Up Rejection watch; renders nothing when the block is absent from data.json.
    rejections = data.get("gap_up_rejection_filters")
    section_html += render_gap_up_rejection_html(rejections)
    section_text += render_gap_up_rejection_text(rejections)
    new_rejections = count_new(rejections)

    subject = (f"{EMAIL_TITLE} - {session} | {count} signal{'' if count == 1 else 's'} | "
               f"{new_rejections} new rejection{'' if new_rejections == 1 else 's'} | V² Money")
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px;">
        <div style="max-width: 680px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px;">
          <h2 style="color: #38bdf8; text-align: center; font-size: 24px; margin: 0 0 6px 0;">{EMAIL_TITLE}</h2>
          <p style="font-size: 13px; color: #94a3b8; text-align: center; margin: 0;">Session {session}</p>
          {section_html}
          <p style="font-size: 13px; color: #64748b; text-align: center; margin-top: 30px; border-top: 1px solid #334155; padding-top: 20px;">
            {FOOTER}
          </p>
        </div>
      </body>
    </html>
    """
    text_body = f"{EMAIL_TITLE} - Session {session}\n{section_text}\n{FOOTER}\n"
    return subject, html_body, text_body


def main():
    parser = argparse.ArgumentParser(description="Gap Momentum Analysis Notifier - V² Money Edition")
    parser.add_argument("--base-dir", "--base_dir", required=False, help="Base directory for equity data (deprecated, kept for compatibility)")
    parser.add_argument("--to", required=False, help="Recipient email address")
    parser.add_argument("--report-only", action="store_true", help="Only generate and save the report locally")
    args = parser.parse_args()

    subject, html_report, text_report = build_email(load_dashboard_data(), datetime.now().strftime("%Y-%m-%d"))

    if args.report_only:
        report_path = Path(PREVIEW_PATH)
        report_path.write_text(html_report)
        logger.info("REPORT_SAVED_LOCALLY", extra={"path": str(report_path)})
        return

    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        logger.error("MISSING_EMAIL_CREDENTIALS")
        sys.exit(1)

    if not args.to:
        logger.error("MISSING_RECIPIENT")
        sys.exit(1)

    client = SMTPClient()
    success = client.send_email(
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=args.to,
        subject=subject,
        html_body=html_report,
        text_body=text_report
    )

    if success:
        logger.info("NOTIFICATION_JOB_COMPLETE")
    else:
        logger.error("NOTIFICATION_JOB_FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
