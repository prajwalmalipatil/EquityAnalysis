#!/usr/bin/env python3
"""
Automation Notifier for Equity Analysis Pipeline
Sends HTML email reports with analysis summary
"""
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta, timezone

def generate_html_report(stats, results_list, trending_list, ticker_list):
    """Generate a premium HTML report with CSS and professional layout."""
    # Ensure IST time regardless of server location (UTC -> IST)
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    date_str = ist_now.strftime('%d-%m-%Y %H:%M')
    
    # CSS remains inline for email compatibility
    styles = """
    <style>
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f0f4f0; margin: 0; padding: 20px; color: #2d3748; }
        .container { max-width: 850px; margin: 0 auto; background: #ffffff; border-radius: 20px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); overflow: hidden; border: 1px solid #e2e8f0; }
        .header { background: linear-gradient(135deg, #4F46E5, #7C3AED); color: white; padding: 40px 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 28px; font-weight: 800; }
        .header p { margin: 15px 0 0; opacity: 0.95; font-size: 15px; }
        .content { padding: 35px; }
        .section { margin-bottom: 30px; background: white; border-radius: 16px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #f1f5f9; }
        .section-title { font-size: 18px; font-weight: 800; color: #1e293b; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .stat-card { background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; }
        .stat-card .label { font-size: 12px; color: #6b7280; text-transform: uppercase; font-weight: 700; }
        .stat-card .value { font-size: 24px; color: #4F46E5; font-weight: 800; margin-top: 8px; }
        .ticker-item { background: #fef2f2; border-left: 5px solid #ef4444; padding: 15px; margin-bottom: 12px; border-radius: 0 12px 12px 0; font-weight: 600; color: #991b1b; }
        .footer { background: #f8fafc; padding: 30px; text-align: center; font-size: 13px; color: #64748b; border-top: 1px solid #e2e8f0; }
    </style>
    """

    header_html = f"""
    <div class="header">
        <h1>📊 Equity Analysis Report</h1>
        <p>Generated on {date_str} IST | Automated Pipeline</p>
    </div>
    """

    ticker_html = "".join([f'<div class="ticker-item">⚠️ SIGNAL: {sym}</div>' for sym in ticker_list]) \
                  if ticker_list else '<p style="color: #64748b; font-style: italic;">No high-probability signals detected.</p>'
    
    trending_html = ", ".join(trending_list[:20]) + ("..." if len(trending_list) > 20 else "") \
                    if trending_list else "No trending symbols identified."

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>{styles}</head>
    <body>
        <div class="container">
            {header_html}
            <div class="content">
                <div class="section">
                    <div class="section-title">📊 Pipeline Statistics</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="label">Extraction Success</div>
                            <div class="value">{stats['extraction']} / 208</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">VSA Symbols Analyzed</div>
                            <div class="value">{stats['vsa']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Trending Identified</div>
                            <div class="value">{stats['trending']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">High-Prob Signals</div>
                            <div class="value">{stats['ticker']}</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">🎯 Ticker Signals</div>
                    {ticker_html}
                </div>

                <div class="section">
                    <div class="section-title">📈 Trending Symbols</div>
                    <p>{trending_html}</p>
                </div>
            </div>

            <div class="footer">
                <p>Automated Equity Analysis Pipeline</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def get_pipeline_data(base_dir: Path):
    """Scan all output folders and return data for the HTML report."""
    equity_data = base_dir / "equity_data"
    results = equity_data / "Results"
    ticker = equity_data / "Ticker"
    trending = equity_data / "Trending"

    def count_files(path, ext="*"):
        return len(list(path.glob(ext))) if path.exists() else 0
    
    def get_symbol_list(path, suffix="_VSA.xlsx"):
        if not path.exists(): return []
        raw_list = sorted([f.name for f in path.glob("*.xlsx")])
        clean_list = []
        for sym in raw_list:
            clean = sym.upper().replace(suffix.upper(), "").replace(".XLSX", "").replace(".CSV", "")
            clean = clean.replace("_1Y_", "_").split("_")[0]
            clean_list.append(clean.strip())
        return sorted(list(set(clean_list)))

    extraction_count = count_files(equity_data, "*.csv")
    results_list = get_symbol_list(results)
    trending_list = get_symbol_list(trending)
    ticker_list = get_symbol_list(ticker)

    stats = {
        "extraction": extraction_count,
        "vsa": len(results_list),
        "trending": len(trending_list),
        "ticker": len(ticker_list)
    }
    
    return stats, results_list, trending_list, ticker_list

def send_email(subject, html_body, to_email):
    """Send HTML email using SMTP."""
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")

    if not all([sender_email, sender_password]):
        print("Error: SMTP credentials missing")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email
    msg.add_alternative(html_body, subtype='html')

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

if __name__ == "__main__":
    base_dir = Path(".")
    stats, results_list, trending_list, ticker_list = get_pipeline_data(base_dir)
    
    html_content = generate_html_report(stats, results_list, trending_list, ticker_list)
    
    recipient = os.environ.get("RECIPIENT_EMAIL", "Prajwalmalipatil@gmail.com")
    
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("SEND_EMAIL") == "true":
        send_email(
            subject=f"Equity Analysis Report: {datetime.now().strftime('%Y-%m-%d')}",
            html_body=html_content,
            to_email=recipient
        )
    else:
        print("Email sending skipped (not in CI environment)")
        print(f"Stats: {stats}")
