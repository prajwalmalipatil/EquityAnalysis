"""
main_notifier.py
New CLI entrypoint for Automated Equity Reporting.
Uses the refactored Reporting Services (Aggregator, Renderer, SMTP).
"""

import argparse
import os
import sys
from pathlib import Path
from src.services.reporting.data_aggregator import DataAggregator
from src.services.reporting.html_renderer import HTMLRenderer
from src.clients.smtp_client import SMTPClient
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("notifier-main")

def main():
    parser = argparse.ArgumentParser(description="Refactored Automated Equity Notifier")
    parser.add_argument("--base-dir", "--base_dir", required=True, help="Base directory for equity data")
    parser.add_argument("--to", required=True, help="Recipient email address")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        logger.error("BASE_DIR_NOT_FOUND", extra={"path": str(base_dir)})
        sys.exit(1)
        
    # Get credentials from environment
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    if not sender_email or not sender_password:
        logger.error("MISSING_EMAIL_CREDENTIALS", 
                    extra={"help": "Set SENDER_EMAIL and SENDER_PASSWORD environment variables."})
        sys.exit(1)
        
    # 1. Aggregate Data
    aggregator = DataAggregator(base_dir)
    stats = aggregator.aggregate_pipeline_stats()
    symbol_data = aggregator.get_symbol_lists()
    
    # Enrich anomalies
    anomaly_details = []
    for sym in symbol_data["anomaly"]:
        details = aggregator.get_anomaly_details(sym)
        if details:
            anomaly_details.append(details)
            
    # 2. Render HTML
    renderer = HTMLRenderer()
    html_report = renderer.render_full_report(stats, symbol_data, anomaly_details)
    
    # 3. Send Email
    client = SMTPClient()
    success = client.send_email(
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=args.to,
        subject=f"Automated Equity Analysis - {renderer.date_str}",
        html_body=html_report
    )
    
    if success:
        logger.info("NOTIFICATION_JOB_COMPLETE")
    else:
        logger.error("NOTIFICATION_JOB_FAILED")

if __name__ == "__main__":
    main()
