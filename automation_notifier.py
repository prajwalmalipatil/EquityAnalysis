#!/usr/bin/env python3
"""
Automation Notifier for Equity Analysis Pipeline
Sends HTML email reports with professional Trade Analysis layout
"""
import os
import smtplib
import ssl
import pandas as pd
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta, timezone

def generate_smart_summary(stats, ticker_list):
    """Generate an intelligent, multi-paragraph analysis summary."""
    ticker_count = len(ticker_list)
    trending_count = stats['trending']
    vsa_count = stats['vsa']
    
    # 1. Market Landscape Paragraph
    if trending_count > 25:
        landscape = f"The latest market analysis reveals a dynamic landscape with significant momentum across key sectors. Our pipeline has successfully processed {vsa_count} symbols, identifying robust trends and high-probability entry points."
    elif trending_count > 10:
        landscape = f"The latest market analysis reveals a selective landscape with emerging momentum in specific sectors. Our pipeline has successfully processed {vsa_count} symbols, identifying {trending_count} robust trends."
    else:
        landscape = f"The latest market analysis reveals a consolidative landscape with limited directional momentum. Our pipeline has processed {vsa_count} symbols, focusing on identifying underlying structural support."

    # 2. Market Participation & Signal Emphasis Paragraph
    participation = "The data suggests a healthy market participation with localized strength in trending assets."
    if ticker_count > 0:
        signal_focus = f"We remain highly optimistic about the identified signals, particularly the {ticker_count} high-probability ticker alerts which warrant immediate attention for potential breakouts."
    else:
        signal_focus = "While high-probability ticker alerts were absent this session, we remain vigilant for emerging patterns that may warrant immediate attention."
    
    breadth = "The overall market breadth remains supportive of strategic long-term publications."

    # 3. Dynamic Outlook Line
    if ticker_count > 5 or trending_count > 30:
        outlook = "🚀 Outlook: Strong Positive Momentum. Market continues to present lucrative opportunities for the disciplined trader."
    elif ticker_count > 0:
        outlook = "🚀 Outlook: Cautiously Optimistic. Selective opportunities present high-conviction entry points for strategic positioning."
    else:
        outlook = "🚀 Outlook: Neutral/Consolidative. Patience is advised as the market prepares for its next directional move."

    return f"""
    <p>{landscape}</p>
    <p>{participation} {signal_focus} {breadth}</p>
    <p style="font-weight: 800; color: #2d3748; margin-top: 15px;">{outlook}</p>
    """

def generate_html_report(stats, results_list, trending_list, ticker_list):
    """Generate a premium Trade Analysis Report in green theme."""
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    date_str = ist_now.strftime('%d-%m-%Y %H:%M')
    
    smart_summary_html = generate_smart_summary(stats, ticker_list)
    
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
        
        .ticker-card { background: #fff; border: 1.5px solid #edf2f7; border-left: 5px solid #e53e3e; border-radius: 6px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        .ticker-symbol { font-size: 18px; font-weight: 800; color: #2d3748; letter-spacing: -0.5px; }
        .ticker-badge { background: #fff5f5; color: #c53030; font-size: 10px; font-weight: 800; padding: 4px 10px; border-radius: 99px; text-transform: uppercase; margin-left: 10px; border: 1px solid #feb2b2; }
        .ticker-status { font-size: 11px; color: #718096; margin-top: 8px; font-weight: 600; display: flex; align-items: center; }
        .ticker-status::before { content: "●"; color: #e53e3e; margin-right: 6px; font-size: 14px; }
        
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

    ticker_content = ""
    if ticker_list:
        for ticker_data in ticker_list:
            sym = ticker_data['symbol']
            sig_type = ticker_data.get('signal_type', 'VSA Pattern Detected')
            effort = ticker_data.get('effort', 'N/A')
            spread = ticker_data.get('spread', 'N/A')
            confidence = ticker_data.get('confidence', 'N/A')
            
            # Formatting spread and confidence if they are floats
            try:
                if isinstance(spread, (int, float)): spread = f"{spread:.2f}"
                if isinstance(confidence, (int, float)): confidence = f"{confidence:.2f}"
            except: pass

            ticker_content += f"""
            <div class="ticker-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span class="ticker-symbol">{sym}</span>
                    <span class="ticker-badge">{sig_type}</span>
                </div>
                <div style="margin-top: 12px; font-size: 13px; color: #4a5568; border-top: 1px solid #edf2f7; padding-top: 10px;">
                    <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                        <tr>
                            <td style="color: #718096; padding: 4px 0;">Effort vs Result:</td>
                            <td style="font-weight: 700; text-align: right; color: #2d3748;">{effort}</td>
                        </tr>
                        <tr>
                            <td style="color: #718096; padding: 4px 0;">Spread Ratio:</td>
                            <td style="font-weight: 700; text-align: right; color: #2d3748;">{spread}</td>
                        </tr>
                        <tr>
                            <td style="color: #718096; padding: 4px 0;">Pattern Confidence:</td>
                            <td style="font-weight: 700; text-align: right; color: #2d3748;">{confidence}</td>
                        </tr>
                    </table>
                </div>
                <div style="font-size: 12px; color: #718096; margin-top: 10px; font-style: italic;">
                    Asset is showing structural strength with institutional demand absorption.
                </div>
                <div class="ticker-status">Requires Immediate Review</div>
            </div>
            """
    else:
        ticker_content = """
        <div style="text-align: center; padding: 30px; border: 1px dashed #cbd5e0; border-radius: 8px; color: #718096; font-size: 12px;">
            <div style="font-size: 24px; margin-bottom: 10px;">🛡️</div>
            No high-probability "No Demand/Supply" signals detected in the latest session.
        </div>
        """

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
                        {smart_summary_html}
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

    def get_symbol_details(path, symbol_name):
        """Read latest signal details from Processing_Log sheet"""
        excel_files = list(path.glob(f"{symbol_name}*.xlsx"))
        if not excel_files: return None
        
        try:
            # Use openpyxl engine specifically for multi-sheet reads
            df_log = pd.read_excel(excel_files[0], sheet_name="Processing_Log")
            if df_log.empty: return None
            
            # Sort by Date descending to ensure latest is first (C2 row)
            if 'Date' in df_log.columns:
                df_log['Date'] = pd.to_datetime(df_log['Date'])
                df_log = df_log.sort_values('Date', ascending=False)
            
            latest = df_log.iloc[0]
            return {
                "symbol": symbol_name,
                "signal_type": latest.get('Signal_Type', 'Unknown'),
                "effort": latest.get('Effort_Result', 'N/A'),
                "spread": latest.get('Spread_Ratio', 'N/A'),
                "confidence": latest.get('Pattern_Confidence', 'N/A')
            }
        except Exception as e:
            print(f"Error reading details for {symbol_name}: {e}")
            return {"symbol": symbol_name}

    extraction_count = count_files(equity_data, "*.csv")
    results_list = get_symbol_list(results)
    trending_list = get_symbol_list(trending)
    
    # Enrich ticker list with technical details
    ticker_symbols = get_symbol_list(ticker)
    ticker_list = []
    for sym in ticker_symbols:
        details = get_symbol_details(ticker, sym)
        if details:
            ticker_list.append(details)
        else:
            ticker_list.append({"symbol": sym})

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
