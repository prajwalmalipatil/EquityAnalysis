"""
main_notifier.py
Fully Wired Entrypoint for V² Money Publications.
Passes detailed ticker, trigger, and anomaly data to the premium renderer.
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
    parser = argparse.ArgumentParser(description="Automated Equity Notifier - V² Money Edition")
    parser.add_argument("--base-dir", "--base_dir", required=True, help="Base directory for equity data")
    parser.add_argument("--to", required=False, help="Recipient email address")
    parser.add_argument("--report-only", action="store_true", help="Only generate and save the report locally")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        logger.error("BASE_DIR_NOT_FOUND", extra={"path": str(base_dir)})
        sys.exit(1)
        
    # 1. Aggregate Deep Metrics
    aggregator = DataAggregator(base_dir)
    stats = aggregator.aggregate_pipeline_stats()
    symbol_data = aggregator.get_symbol_lists()
    
    # 2. Enrich for Deep Tables
    ticker_details = []
    for sym in symbol_data["ticker"]:
        details = aggregator.get_ticker_details(sym)
        if details: ticker_details.append(details)
        
    trigger_details = []
    for sym in symbol_data["triggers"]:
        details = aggregator.get_trigger_details(sym)
        if details: trigger_details.append(details)
            
    anomaly_details = []
    for sym in symbol_data["anomaly"]:
        details = aggregator.get_anomaly_details(sym)
        if details: anomaly_details.append(details)

    eigen_details = []
    for sym in symbol_data["eigen_filter"]:
        details = aggregator.get_eigen_details(sym)
        if details: eigen_details.append(details)
            
    # 3. Render Premium HTML
    renderer = HTMLRenderer()
    html_report = renderer.render_full_report(
        stats=stats,
        ticker_details=ticker_details,
        trigger_details=trigger_details,
        anomaly_details=anomaly_details,
        trending_symbols=symbol_data["trending"],
        eigen_details=eigen_details
    )
    
    # 4. Handle Output
    if args.report_only:
        report_path = Path("local_report_preview.html")
        report_path.write_text(html_report)
        logger.info("REPORT_SAVED_LOCALLY", extra={"path": str(report_path)})
        return

    # Standard email flow (requires credentials)
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
        subject=f"Trade Analysis Report - {renderer.date_str} IST | V² Money",
        html_body=html_report
    )
    
    if success:
        logger.info("NOTIFICATION_JOB_COMPLETE")
    else:
        logger.error("NOTIFICATION_JOB_FAILED")

if __name__ == "__main__":
    main()
