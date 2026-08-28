import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from src.clients.smtp_client import SMTPClient
from src.utils.observability import get_tenant_logger

from typing import List, Dict

logger = get_tenant_logger("notifier-main")


def _render_vt_html_table(items: List[Dict], timeframe_label: str) -> str:
    """Renders a responsive HTML table for a single timeframe's Volume Trap stocks."""
    if not items:
        return ""

    rows_html = ""
    for i, item in enumerate(items):
        bg_color = "#1e293b" if i % 2 == 0 else "#253347"
        sym = item.get("symbol", "--")
        sentiment = item.get("sentiment", "Neutral")
        if sentiment == "Bullish":
            sent_badge = '<span style="color: #4ade80; font-weight: bold;">▲ Bullish</span>'
        elif sentiment == "Bearish":
            sent_badge = '<span style="color: #f87171; font-weight: bold;">▼ Bearish</span>'
        else:
            sent_badge = f'<span style="color: #94a3b8;">{sentiment}</span>'

        vol_pct = f"{item.get('vol_delta_pct'):+0.1f}%" if item.get("vol_delta_pct") is not None else "--"
        spread_pct = f"{item.get('spread_delta_pct'):+0.1f}%" if item.get("spread_delta_pct") is not None else "--"
        body_ratio = f"{item.get('body_ratio'):.4f}" if item.get("body_ratio") is not None else "--"

        rows_html += f"""
        <tr style="background-color: {bg_color}; border-bottom: 1px solid #334155;">
            <td style="padding: 7px 10px; font-weight: bold; color: #f8fafc;">{sym}</td>
            <td style="padding: 7px 10px; font-size: 12px;">{sent_badge}</td>
            <td style="padding: 7px 10px; color: #38bdf8; text-align: right;">{vol_pct}</td>
            <td style="padding: 7px 10px; color: #fbbf24; text-align: right;">{spread_pct}</td>
            <td style="padding: 7px 10px; color: #cbd5e1; text-align: right;">{body_ratio}</td>
        </tr>"""

    return f"""
    <div style="margin-top: 14px;">
        <div style="font-weight: bold; color: #f1f5f9; font-size: 13px; margin-bottom: 6px;">
            📅 {timeframe_label} ({len(items)} stock{'s' if len(items) > 1 else ''})
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; background-color: #1e293b; border-radius: 6px; overflow: hidden;">
            <thead>
                <tr style="background-color: #0f172a; color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
                    <th style="padding: 7px 10px;">Symbol</th>
                    <th style="padding: 7px 10px;">Sentiment</th>
                    <th style="padding: 7px 10px; text-align: right;">Vol Δ%</th>
                    <th style="padding: 7px 10px; text-align: right;">Spread Δ%</th>
                    <th style="padding: 7px 10px; text-align: right;">Body Ratio</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>"""


def _format_volume_trap_email_html(vt_filters: Dict) -> str:
    """Generates the full HTML section with summary and detailed tables for Volume Trap."""
    daily_vt = vt_filters.get("daily", [])
    weekly_vt = vt_filters.get("weekly", [])
    monthly_vt = vt_filters.get("monthly", [])
    total_vt = len(daily_vt) + len(weekly_vt) + len(monthly_vt)

    if total_vt == 0:
        return ""

    all_vt = daily_vt + weekly_vt + monthly_vt
    bullish_count = sum(1 for v in all_vt if v.get("sentiment") == "Bullish")
    bearish_count = sum(1 for v in all_vt if v.get("sentiment") == "Bearish")

    daily_table = _render_vt_html_table(daily_vt, "Daily")
    weekly_table = _render_vt_html_table(weekly_vt, "Weekly")
    monthly_table = _render_vt_html_table(monthly_vt, "Monthly")

    return f"""
    <div style="background-color: #334155; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #2dd4bf;">
        <h3 style="color: #2dd4bf; margin: 0 0 6px 0; font-size: 18px;">🎯 Volume Trap Filter Insights</h3>
        <p style="color: #cbd5e1; margin: 0; font-size: 14px;">
            <strong>{total_vt}</strong> stocks detected across timeframes
            (Daily: {len(daily_vt)}, Weekly: {len(weekly_vt)}, Monthly: {len(monthly_vt)})
        </p>
        <p style="color: #94a3b8; margin: 6px 0 4px 0; font-size: 13px;">
            <span style="color: #4ade80; font-weight: bold;">▲ Bullish: {bullish_count}</span> &nbsp;|&nbsp;
            <span style="color: #f87171; font-weight: bold;">▼ Bearish: {bearish_count}</span>
        </p>
        {daily_table}
        {weekly_table}
        {monthly_table}
    </div>"""


def _render_vt_text_table(items: List[Dict], timeframe_label: str) -> str:
    """Renders a formatted text table for a single timeframe."""
    if not items:
        return ""
    lines = [
        f"  📅 {timeframe_label} ({len(items)} stocks):",
        f"  {'SYMBOL':<12} {'SENTIMENT':<10} {'VOL Δ%':>10} {'SPREAD Δ%':>10} {'BODY RATIO':>12}",
        f"  {'-'*56}"
    ]
    for item in items:
        sym = item.get("symbol", "--")
        sent = item.get("sentiment", "Neutral")
        vol = f"{item.get('vol_delta_pct'):+0.1f}%" if item.get("vol_delta_pct") is not None else "--"
        spread = f"{item.get('spread_delta_pct'):+0.1f}%" if item.get("spread_delta_pct") is not None else "--"
        body = f"{item.get('body_ratio'):.4f}" if item.get("body_ratio") is not None else "--"
        lines.append(f"  {sym:<12} {sent:<10} {vol:>10} {spread:>10} {body:>12}")
    lines.append("")
    return "\n".join(lines)


def _format_volume_trap_email_text(vt_filters: Dict) -> str:
    """Generates the full plain-text report for Volume Trap."""
    daily_vt = vt_filters.get("daily", [])
    weekly_vt = vt_filters.get("weekly", [])
    monthly_vt = vt_filters.get("monthly", [])
    total_vt = len(daily_vt) + len(weekly_vt) + len(monthly_vt)

    if total_vt == 0:
        return ""

    all_vt = daily_vt + weekly_vt + monthly_vt
    bullish_count = sum(1 for v in all_vt if v.get("sentiment") == "Bullish")
    bearish_count = sum(1 for v in all_vt if v.get("sentiment") == "Bearish")

    lines = [
        f"\n🎯 Volume Trap Filter Insights: {total_vt} stocks detected (Daily: {len(daily_vt)}, Weekly: {len(weekly_vt)}, Monthly: {len(monthly_vt)})",
        f"   Bullish: {bullish_count} | Bearish: {bearish_count}\n"
    ]
    for label, items in [("Daily", daily_vt), ("Weekly", weekly_vt), ("Monthly", monthly_vt)]:
        tbl = _render_vt_text_table(items, label)
        if tbl:
            lines.append(tbl)

    return "\n".join(lines)


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
                    seen_titles = []
                    deduped_events = []
                    
                    raw_new_events = [
                        e for e in macro_events 
                        if e.get("is_new_since_last_session") or e.get("processing_state") == "NEW"
                    ]
                    if not raw_new_events:
                        raw_new_events = macro_events[:3]
                        
                    def clean_title(t: str) -> set:
                        fillers = {"rbi", "issues", "announces", "to", "on", "for", "a", "an", "the", "and", "of", "in", "with", "under"}
                        words = [w for w in t.lower().split() if w.isalnum()]
                        return set(w for w in words if w not in fillers)

                    def is_duplicate(t1: str, t2: str) -> bool:
                        s1 = clean_title(t1)
                        s2 = clean_title(t2)
                        if not s1 or not s2:
                            return False
                        common = s1.intersection(s2)
                        overlap = len(common) / min(len(s1), len(s2))
                        return overlap >= 0.7

                    for e in raw_new_events:
                        title = e.get("title", "")
                        duplicate = False
                        for seen in seen_titles:
                            if is_duplicate(title, seen):
                                duplicate = True
                                break
                        if not duplicate:
                            seen_titles.append(title)
                            deduped_events.append(e)
                            
                    new_events = deduped_events[:3]
                    
                    macro_html = f"""
                    <div style="background-color: #334155; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #f59e0b;">
                        <h3 style="color: #fbbf24; margin-top: 0;">🌍 Latest Macro Intelligence</h3>
                        <ul style="padding-left: 20px; margin-bottom: 0;">
                            {''.join([f'<li style="margin-bottom: 10px;"><strong>{e.get("title", "")}</strong> - {(e.get("summary", "") or "No summary available").split(".")[0]}...</li>' for e in new_events])}
                        </ul>
                    </div>
                    """
                    
                    macro_text = "\nLatest Macro Intelligence:\n" + "\n".join([f"- {e.get('title', '')}" for e in new_events]) + "\n"
    except Exception as e:
        logger.warning("FAILED_TO_LOAD_MACRO_FOR_EMAIL", extra={"error": str(e)})

    # Extract Volume Trap Filter details from data.json
    volume_trap_html = ""
    volume_trap_text = ""
    try:
        data_path = Path("dashboard/data.json")
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                vt_filters = data.get("volume_trap_filters", {})
                volume_trap_html = _format_volume_trap_email_html(vt_filters)
                volume_trap_text = _format_volume_trap_email_text(vt_filters)
    except Exception as e:
        logger.warning("FAILED_TO_LOAD_VOLUME_TRAP_FOR_EMAIL", extra={"error": str(e)})

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
          
          {volume_trap_html}
          
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
{volume_trap_text}
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
        sys.exit(1)

if __name__ == "__main__":
    main()
