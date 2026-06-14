import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from src.clients.smtp_client import SMTPClient
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("notifier-main")

def main():
    parser = argparse.ArgumentParser(description="Automated Equity Notifier - V² Money Edition")
    parser.add_argument("--base-dir", "--base_dir", required=False, help="Base directory for equity data (deprecated, kept for compatibility)")
    parser.add_argument("--to", required=False, help="Recipient email address")
    parser.add_argument("--report-only", action="store_true", help="Only generate and save the report locally")
    
    args = parser.parse_args()
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    dashboard_url = "https://prajwalmalipatil.github.io/EquityAnalysis/"
    
    # Extract Macro Intelligence from data.json if available
    macro_html = ""
    macro_text = ""
    try:
        data_path = Path("dashboard/data.json")
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                macro_events = data.get("macro_intelligence", {}).get("recent_events", [])
                
                if macro_events:
                    # Look for strictly NEW events
                    new_events = [e for e in macro_events if e.get("is_new_since_last_session") or e.get("metadata", {}).get("lifecycle_status") == "NEW"]
                    if not new_events:
                        new_events = macro_events[:3] # Fallback to latest 3
                        
                    macro_html = f"""
                    <div style="background-color: #334155; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #f59e0b;">
                        <h3 style="color: #fbbf24; margin-top: 0;">🌍 Latest Macro Intelligence</h3>
                        <ul style="padding-left: 20px; margin-bottom: 0;">
                            {''.join([f'<li style="margin-bottom: 10px;"><strong>{e.get("official_data", {}).get("title", e.get("title", ""))}</strong> - {e.get("derived_data", {}).get("ai_summary", e.get("summary", "")).split(".")[0]}...</li>' for e in new_events])}
                        </ul>
                    </div>
                    """
                    
                    macro_text = "\nLatest Macro Intelligence:\n" + "\n".join([f"- {e.get('official_data', {}).get('title', e.get('title', ''))}" for e in new_events]) + "\n"
    except Exception as e:
        logger.warning("FAILED_TO_LOAD_MACRO_FOR_EMAIL", extra={"error": str(e)})

    html_report = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);">
          <h2 style="color: #38bdf8; text-align: center; font-size: 24px; margin-bottom: 20px;">Daily Equity Analysis Ready</h2>
          <p style="font-size: 16px; line-height: 1.6; color: #cbd5e1; text-align: center;">
            The automated pipeline has completed processing data for <strong>{date_str}</strong>. <br/><br/>
            Detailed insights including VSA, EigenFilters, Consensus Ratings, and <strong>Macro Intelligence</strong> are now live on your interactive dashboard.
          </p>
          
          {macro_html}
          
          <div style="text-align: center; margin: 40px 0;">
            <a href="{dashboard_url}" style="background-color: #0ea5e9; color: #ffffff; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-size: 16px; font-weight: bold; display: inline-block; box-shadow: 0 4px 6px rgba(14,165,233,0.3);">
              View Interactive Dashboard &rarr;
            </a>
          </div>
          <p style="font-size: 13px; color: #64748b; text-align: center; margin-top: 30px; border-top: 1px solid #334155; padding-top: 20px;">
            Generated automatically by V² Money Automation
          </p>
        </div>
      </body>
    </html>
    """

    text_report = f"""Daily Equity Analysis Ready
    
The automated pipeline has completed processing data for {date_str}.
Detailed analysis including VSA, EigenFilters, Consensus Ratings, and Macro Intelligence are now live on your interactive dashboard.
{macro_text}
View your Interactive Dashboard here:
{dashboard_url}

Generated automatically by V² Money Automation
"""

    if args.report_only:
        report_path = Path("local_report_preview.html")
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
        subject=f"Trade Analysis Dashboard Updated - {date_str} IST | V² Money",
        html_body=html_report,
        text_body=text_report
    )
    
    if success:
        logger.info("NOTIFICATION_JOB_COMPLETE")
    else:
        logger.error("NOTIFICATION_JOB_FAILED")

if __name__ == "__main__":
    main()
