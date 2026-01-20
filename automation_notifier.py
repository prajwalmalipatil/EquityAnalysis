#!/usr/bin/env python3
"""
Automation Notifier for Equity Analysis Pipeline
Sends HTML email reports with professional Trade Analysis layout
"""
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta, timezone

def generate_smart_summary(stats, ticker_list):
    """Generate an intelligent, context-aware analysis summary."""
    extraction_rate = (stats['extraction'] / 208) * 100
    ticker_count = len(ticker_list)
    trending_count = stats['trending']
    
    # 1. Opening Statement based on signals
    if ticker_count > 10:
        opening = "The latest market analysis reveals a highly dynamic landscape with significant momentum across multiple sectors."
    elif ticker_count > 0:
        opening = "The latest market analysis reveals a selective landscape with robust trends and high-probability entry points identifying specific sector strength."
    else:
        opening = "The latest session reflects a consolidative phase across the benchmark universe, with limited high-probability breakouts detected."

    # 2. Middle detail based on processing
    if extraction_rate >= 95:
        detail = f"Our pipeline has successfully processed {stats['vsa']} symbols, identifying {trending_count} robust trends."
    else:
        detail = f"Our pipeline has analyzed {stats['vsa']} symbols during this session, focusing on high-liquidity segments."

    # 3. Closing tone
    closing = "The data suggests a healthy market participation with localized strength in trending assets. We remain highly optimistic about the attention for potential breakouts. The overall market breadth remains supportive of strategic long-term publications."
    
    return f"{opening} {detail} {closing}"

def generate_html_report(stats, results_list, trending_list, ticker_list):
    """Generate a premium Trade Analysis Report in green theme."""
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    date_str = ist_now.strftime('%d-%m-%Y %H:%M')
    
    smart_summary = generate_smart_summary(stats, ticker_list)
    
    styles = """
    <style>
        body { font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif; background-color: #fcfdfc; margin: 0; padding: 20px; color: #1a202c; }
        .container { max-width: 800px; margin: 0 auto; background: #ffffff; border-radius: 4px; border: 1px solid #e2e8f0; }
        .header { background-color: #7cb342; color: white; padding: 30px 20px; text-align: center; }
        .header h1 { margin: 0; font-size: 24px; font-weight: 700; text-transform: none; }
        .header p { margin: 10px 0 0; font-size: 13px; opacity: 0.9; }
        .content { padding: 25px; }
        .section { margin-bottom: 30px; }
        .section-header { font-size: 15px; font-weight: 800; color: #2d3748; padding-bottom: 8px; border-bottom: 1.5px solid #edf2f7; margin-bottom: 15px; display: flex; align-items: center; }
        .section-header span { margin-right: 8px; }
        .stat-box { border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 12px; padding: 15px; background: #fff; }
        .stat-label { font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 700; letter-spacing: 0.5px; }
        .stat-value { font-size: 20px; font-weight: 800; color: #2d3748; margin-top: 4px; }
        
        /* 3-Column Table Logic */
        .symbol-table { width: 100%; border-collapse: collapse; font-size: 11px; }
        .symbol-table td { padding: 8px 12px; border-bottom: 1px solid #f7fafc; color: #4a5568; font-weight: 600; width: 33.33%; text-transform: uppercase; }
        
        .ticker-bubble { background: #fffcfc; border-left: 4px solid #f56565; padding: 12px 15px; margin-bottom: 10px; font-size: 13px; color: #4a5568; }
        .summary-box { background: #f9fafb; padding: 20px; border-radius: 4px; line-height: 1.6; font-size: 13px; color: #4a5568; }
        .footer { padding: 20px; text-align: center; font-size: 11px; color: #a0aec0; border-top: 1px solid #edf2f7; }
    </style>
    """

    # Format Trending Symbols into 3 columns
    table_rows = ""
    for i in range(0, len(trending_list), 3):
        chunk = trending_list[i:i+3]
        row = "<tr>"
        for sym in chunk:
            row += f"<td>{sym}</td>"
        # Pad row if less than 3 columns
        for _ in range(3 - len(chunk)):
            row += "<td></td>"
        row += "</tr>"
        table_rows += row

    ticker_content = "".join([f'<div class="ticker-bubble">🎯 PATTERN ALERT: <b>{sym}</b> detected with high confirmation.</div>' for sym in ticker_list]) \
                     if ticker_list else '<p style="color: #718096; font-style: italic; font-size: 12px;">No high-probability signals detected in the latest session.</p>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>{styles}</head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Trade Analysis Report</h1>
                <p>Generated on {date_str} IST | V² Money Publications</p>
            </div>
            
            <div class="content">
                <!-- STAGE STATISTICS -->
                <div class="section">
                    <div class="section-header">📊 STAGE STATISTICS</div>
                    <div class="stat-box">
                        <div class="stat-label">EXTRACTION SUCCESS</div>
                        <div class="stat-value">{stats['extraction']} / 208</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">VSA SYMBOLS ANALYZED</div>
                        <div class="stat-value">{stats['vsa']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">TRENDING IDENTIFIED</div>
                        <div class="stat-value">{stats['trending']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">HIGH-PROB SIGNALS</div>
                        <div class="stat-value">{stats['ticker']}</div>
                    </div>
                </div>

                <!-- TICKER SIGNALS -->
                <div class="section">
                    <div class="section-header">🎯 TICKER SIGNALS (Action Required)</div>
                    {ticker_content}
                </div>

                <!-- TRENDING SYMBOLS -->
                <div class="section">
                    <div class="section-header">📈 TRENDING SYMBOLS</div>
                    <p style="font-size: 11px; color: #718096; margin-bottom: 10px;">TRENDING STOCKS LIST ▼</p>
                    <table class="symbol-table">
                        {table_rows}
                    </table>
                </div>

                <!-- RESULTS SUMMARY -->
                <div class="section">
                    <div class="section-header">📒 Results: Analysis Summary</div>
                    <div class="summary-box">
                        {smart_summary}
                    </div>
                </div>
            </div>

            <div class="footer">
                <p>&copy; {datetime.now().year} V² Money Publications - Automated Equity Pipeline</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def get_pipeline_data(base_dir: Path):
    """Scan output folders and return data for the report."""
    # Use equity_data if it exists, otherwise fall back to current directory scan
    # Priority: equity_data_manual (for test runs) > equity_data (production)
    search_dirs = ["equity_data_manual", "equity_data", "."]
    
    equity_data = None
    for d in search_dirs:
        if (base_dir / d / "Results").exists():
            equity_data = base_dir / d
            break
    
    if not equity_data:
        # Fallback to current directory
        equity_data = base_dir

    results = equity_data / "Results"
    ticker = equity_data / "Ticker"
    trending = equity_data / "Trending"

    def count_files(path, ext="*.csv"):
        return len(list(path.glob(ext))) if path.exists() else 0
    
    def get_symbol_list(path, suffix="_VSA.xlsx"):
        if not path.exists(): return []
        raw_list = sorted([f.name for f in path.glob("*.xlsx")])
        clean_list = []
        for sym in raw_list:
            clean = sym.upper().replace(suffix.upper(), "").replace(".XLSX", "").replace(".CSV", "")
            # Remove date if present (e.g., RELIANCE_1Y_20260121 -> RELIANCE)
            parts = clean.split('_')
            clean_list.append(parts[0].strip())
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
    msg["From"] = f"Trade Analysis System <{sender_email}>"
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
            subject=f"Trade Analysis Report: {datetime.now().strftime('%Y-%m-%d')}",
            html_body=html_content,
            to_email=recipient
        )
    else:
        # Create local debug file
        debug_path = Path("debug_report.html")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Local debug report created: {debug_path.absolute()}")
        print(f"Stats: {stats}")
        print(f"Summary: {generate_smart_summary(stats, ticker_list)}")
