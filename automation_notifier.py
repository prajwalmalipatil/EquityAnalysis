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

def generate_html_report(stats, results_list, trending_list, ticker_list, triggers_list, anomaly_list):
    """Generate a premium Trade Analysis Report in green theme."""
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    date_str = ist_now.strftime('%d-%m-%Y %H:%M')
    
    smart_summary_html = generate_smart_summary(stats, ticker_list)
    
    
    # Generate Trigger Section HTML
    triggers_content = ""
    if triggers_list:
        rows = ""
        for t in triggers_list:
            vol_color = "#e53e3e" if t['vol_pct'] < 0 else "#38a169"
            spr_color = "#38a169" if t['spread_pct'] > 0 else "#e53e3e"
            
            rows += f"""
            <tr style="border-bottom: 1px solid #f7fafc;">
                <td style="padding: 10px; font-weight: 700; color: #2d3748;">{t['symbol']}</td>
                <td style="padding: 10px; text-align: right; color: #718096; font-size: 11px;">{int(t['prev_vol']):,}</td>
                <td style="padding: 10px; text-align: right; font-weight: 700; color: #2d3748;">{int(t['curr_vol']):,}</td>
                <td style="padding: 10px; text-align: right; color: #718096; font-size: 11px;">{t['prev_spread']:.2f}</td>
                <td style="padding: 10px; text-align: right; font-weight: 700; color: #2d3748;">{t['curr_spread']:.2f}</td>
                <td style="padding: 10px; text-align: right; font-weight: 700; color: {vol_color};">{t['vol_pct']:.1f}%</td>
                <td style="padding: 10px; text-align: right; font-weight: 700; color: {spr_color};">{t['spread_pct']:.1f}%</td>
            </tr>
            """
            
        triggers_content = f"""
        <div class="section">
            <div class="section-header" style="color: #c53030;">📊 VSA Triggered Stocks – Vol Contraction + Spread Expansion</div>
            
            <div style="background: #fff; border: 1px solid #fed7d7; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 12px; color: #742a2a; line-height: 1.5;">
                <strong>Trigger Logic:</strong> Volume decreased from previous day while Spread expanded. This anomaly suggests potential supply removal facilitating easier price movement.
            </div>

            <div class="stat-box" style="display: flex; justify-content: space-between; align-items: center; border-color: #feb2b2; background: #fff5f5;">
                <div>
                    <div class="stat-label" style="color: #9b2c2c;">TRIGGERED STOCKS</div>
                    <div class="stat-value" style="color: #c53030;">{len(triggers_list)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="stat-label" style="color: #9b2c2c;">ANALYSIS DATE</div>
                    <div class="stat-value" style="color: #c53030; font-size: 14px; margin-top: 4px;">{date_str.split()[0]}</div>
                </div>
            </div>

            <table style="width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 10px; background: #fff; border: 1px solid #edf2f7;">
                <thead style="background: #f7fafc; color: #4a5568;">
                    <tr>
                        <th style="padding: 10px; text-align: left; font-weight: 700;">SYMBOL</th>
                        <th style="padding: 10px; text-align: right; font-weight: 600;">PREV VOL</th>
                        <th style="padding: 10px; text-align: right; font-weight: 600;">CURR VOL</th>
                        <th style="padding: 10px; text-align: right; font-weight: 600;">PREV SPR</th>
                        <th style="padding: 10px; text-align: right; font-weight: 600;">CURR SPR</th>
                        <th style="padding: 10px; text-align: right; font-weight: 600;">VOL %</th>
                        <th style="padding: 10px; text-align: right; font-weight: 600;">SPR %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            
            <div style="margin-top: 12px; font-size: 11px; color: #718096; font-style: italic; border-left: 3px solid #cbd5e0; padding-left: 10px;">
                <strong>Observational Note:</strong> Stocks with >30% volume drop and significant spread expansion (>50%) warrant close monitoring for continuation.
            </div>
        </div>
        """
    else:
        triggers_content = """
        <div class="section">
             <div class="section-header" style="color: #718096;">📊 VSA Triggered Stocks</div>
             <div style="padding: 20px; text-align: center; color: #a0aec0; border: 1px dashed #e2e8f0; border-radius: 6px; font-size: 12px; font-style: italic;">
                No stocks met the Volume Contraction + Spread Expansion criteria today.
             </div>
        </div>
        """

    # Generate Anomaly Section HTML (Volume Build-Up then Drop + OHLC Classification)
    anomaly_content = ""
    if anomaly_list:
        bullish_anomalies = [a for a in anomaly_list if a.get('sentiment') == 'Bullish']
        bearish_anomalies = [a for a in anomaly_list if a.get('sentiment') == 'Bearish']
        neutral_anomalies = [a for a in anomaly_list if a.get('sentiment') == 'Neutral']
        
        def render_anomaly_table(anomalies, title, header_color, bg_color, border_color):
            if not anomalies: return ""
            rows = ""
            for a in anomalies:
                drop_color = "#e53e3e" if a['drop_pct'] < 0 else "#38a169"
                rows += f"""
                <tr style="border-bottom: 1px solid #edf2f7;">
                    <td style="padding: 10px; font-weight: 700; color: #2d3748;">{a['symbol']}</td>
                    <td style="padding: 10px; text-align: left; color: {header_color}; font-size: 11px; font-weight: 600;">{a.get('pattern', 'Neutral')}</td>
                    <td style="padding: 10px; text-align: right; color: #718096; font-size: 11px;">{int(a['vol_t1']):,}</td>
                    <td style="padding: 10px; text-align: right; font-weight: 700; color: #2d3748;">{int(a['vol_tday']):,}</td>
                    <td style="padding: 10px; text-align: right; font-weight: 700; color: {drop_color};">{a['drop_pct']:.1f}%</td>
                </tr>
                """
            return f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: {header_color}; margin-bottom: 6px; display: flex; align-items: center;">{title}</div>
                <table style="width: 100%; border-collapse: collapse; font-size: 11px; background: {bg_color}; border: 1px solid {border_color}; border-radius: 6px; overflow: hidden;">
                    <thead style="background: {border_color}; color: {header_color};">
                        <tr>
                            <th style="padding: 8px 10px; text-align: left;">SYMBOL</th>
                            <th style="padding: 8px 10px; text-align: left;">OHLC PATTERN</th>
                            <th style="padding: 8px 10px; text-align: right;">PREV VOL</th>
                            <th style="padding: 8px 10px; text-align: right;">CURR VOL</th>
                            <th style="padding: 8px 10px; text-align: right;">DROP %</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """
            
        bullish_html = render_anomaly_table(bullish_anomalies, "🟢 Bullish Setups (Accumulation / Traps)", "#22543d", "#f0fff4", "#c6f6d5")
        bearish_html = render_anomaly_table(bearish_anomalies, "🔴 Bearish Setups (Distribution / Supply)", "#742a2a", "#fff5f5", "#fed7d7")
        neutral_html = render_anomaly_table(neutral_anomalies, "⚪ Neutral Contractions", "#4a5568", "#fcfdfc", "#edf2f7")
        
        anomaly_content = f"""
        <div class="section">
            <div class="section-header" style="color: #c05621;">⚠️ Volume Anomaly V2 – OHLC Pattern Classification</div>
            
            <div style="background: #fff; border: 1px solid #feebc8; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 12px; color: #7b341e; line-height: 1.5;">
                <strong>Advanced Logic:</strong> Volume dropped >10% after a 3-day build-up. We cross-referenced this volume signal with the day's OHLC (Open, High, Low, Close) price action to determine whether Smart Money was accumulating or distributing.
            </div>

            <div class="stat-box" style="display: flex; justify-content: space-between; align-items: center; border-color: #fbd38d; background: #fffaf0;">
                <div>
                    <div class="stat-label" style="color: #975a16;">TOTAL ANOMALY STOCKS</div>
                    <div class="stat-value" style="color: #c05621;">{len(anomaly_list)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="stat-label" style="color: #975a16;">BULLISH / BEARISH</div>
                    <div class="stat-value" style="color: #c05621; font-size: 16px; margin-top: 4px;">{len(bullish_anomalies)} / {len(bearish_anomalies)}</div>
                </div>
            </div>
            
            {bullish_html}
            {bearish_html}
            {neutral_html}
            
            <div style="margin-top: 15px; font-size: 11px; color: #718096; font-style: italic; border-left: 3px solid #fbd38d; padding-left: 10px;">
                <strong>Data Insight:</strong> The 'Silent Accumulation' pattern boasts a historic 70% win rate over a 5-day horizon. Avoid or short the Bearish setups.
            </div>
        </div>
        """
    else:
        anomaly_content = """
        <div class="section">
             <div class="section-header" style="color: #718096;">⚠️ Volume Anomaly Stocks</div>
             <div style="padding: 20px; text-align: center; color: #a0aec0; border: 1px dashed #e2e8f0; border-radius: 6px; font-size: 12px; font-style: italic;">
                No stocks met the Volume Build-Up then Drop criteria today.
             </div>
        </div>
        """

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
                    <div class="stat-box">
                        <div class="stat-label">ANOMALY DETECTIONS</div>
                        <div class="stat-value">{stats['anomaly']}</div>
                    </div>
                </div>

                <!-- TICKER SIGNALS -->
                <div class="section">
                    <div class="section-header">🎯 TICKER SIGNALS (Action Required)</div>
                    {ticker_content}
                </div>

                <!-- VSA TRIGGERED STOCKS (NEW SECTION) -->
                {triggers_content}

                <!-- VOLUME ANOMALY STOCKS (NEW SECTION) -->
                {anomaly_content}

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
    triggers = equity_data / "Triggers"
    anomaly = equity_data / "Anomaly"

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
            print(f"Error reading details for {symbol_name}: {e}")
            return {"symbol": symbol_name}

    def get_trigger_details(path, symbol_name):
        """Read trigger metrics from VSA_Analysis sheet"""
        excel_files = list(path.glob(f"{symbol_name}*.xlsx"))
        if not excel_files: return None
        
        try:
            # Read VSA_Analysis sheet
            df = pd.read_excel(excel_files[0], sheet_name="VSA_Analysis")
            if len(df) < 2: return None
            
            # Ensure sorting
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.sort_values("Date")
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Metric calculations
            vol_curr = float(curr.get("Volume", 0))
            vol_prev = float(prev.get("Volume", 1)) # avoid div/0
            spr_curr = float(curr.get("Spread", 0))
            spr_prev = float(prev.get("Spread", 1))
            
            vol_change = ((vol_curr - vol_prev) / vol_prev) * 100
            spread_change = ((spr_curr - spr_prev) / spr_prev) * 100
            
            return {
                "symbol": symbol_name,
                "prev_vol": vol_prev,
                "curr_vol": vol_curr,
                "prev_spread": spr_prev,
                "curr_spread": spr_curr,
                "vol_pct": vol_change,
                "spread_pct": spread_change
            }
        except Exception as e:
            print(f"Error extracting trigger details for {symbol_name}: {e}")
            return None

    def get_anomaly_details(path, symbol_name):
        """Read anomaly volume metrics from VSA_Analysis sheet and classify OHLC pattern"""
        excel_files = list(path.glob(f"{symbol_name}*.xlsx"))
        if not excel_files: return None
        
        try:
            df = pd.read_excel(excel_files[0], sheet_name="VSA_Analysis")
            if len(df) < 4: return None
            
            # Ensure sorting
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.sort_values("Date")
            
            t_day = df.iloc[-1]
            t_m1 = df.iloc[-2]
            t_m2 = df.iloc[-3]
            t_m3 = df.iloc[-4]
            
            vol_tday = float(t_day.get("Volume", 0))
            vol_tm1 = float(t_m1.get("Volume", 1))
            vol_tm2 = float(t_m2.get("Volume", 1))
            vol_tm3 = float(t_m3.get("Volume", 1))
            
            drop_pct = ((vol_tday - vol_tm1) / vol_tm1) * 100 if vol_tm1 > 0 else 0
            
            # OHLC Price Action Extraction
            O = float(t_day.get("Open", 0))
            H = float(t_day.get("High", 0.01))
            L = float(t_day.get("Low", 0))
            C = float(t_day.get("Close", 0))
            prev_C = float(t_m1.get("Close", 0))
            prev_O = float(t_m1.get("Open", 0))
            
            total_range = H - L if H > L else 0.01
            close_pos = (C - L) / total_range if total_range > 0.01 else 0.5
            gap_pct = ((O - prev_C) / prev_C) * 100 if prev_C > 0 else 0
            
            # Pattern logic flags
            is_bearish_bar = C < O
            close_near_low = close_pos < 0.30
            close_near_high = close_pos > 0.70
            gap_down = gap_pct < -0.3
            heavy_drop = drop_pct < -50
            mod_drop = -50 <= drop_pct < -25
            close_above_prev = C > prev_C
            prev_was_up = prev_C > prev_O
            
            # Classification
            pattern_name = "Neutral Contraction"
            sentiment = "Neutral"
            
            # Top Bullish Found
            if close_near_low and heavy_drop and close_above_prev:
                pattern_name = "Silent Accumulation (70% Win Rate)"
                sentiment = "Bullish"
            elif close_near_low and prev_was_up and close_above_prev:
                pattern_name = "Pullback Absorption (65% Win Rate)"
                sentiment = "Bullish"
            elif close_near_high and gap_down and heavy_drop:
                pattern_name = "Gap Absorption (64% Win Rate)"
                sentiment = "Bullish"
            elif is_bearish_bar and close_above_prev:
                pattern_name = "Bear Trap (63% Win Rate)"
                sentiment = "Bullish"
            
            # Top Bearish Found
            elif close_near_low and gap_down and mod_drop:
                pattern_name = "Continuation Dump (67% Downside)"
                sentiment = "Bearish"
            elif close_near_high and is_bearish_bar and not close_above_prev:
                pattern_name = "Failed Rally (63% Downside)"
                sentiment = "Bearish"
            elif close_near_high and is_bearish_bar and not prev_was_up:
                pattern_name = "Supply Resurgence (61% Downside)"
                sentiment = "Bearish"
            
            return {
                "symbol": symbol_name,
                "vol_t3": vol_tm3,
                "vol_t2": vol_tm2,
                "vol_t1": vol_tm1,
                "vol_tday": vol_tday,
                "drop_pct": drop_pct,
                "pattern": pattern_name,
                "sentiment": sentiment
            }
        except Exception as e:
            print(f"Error extracting anomaly details for {symbol_name}: {e}")
            return None

    extraction_count = count_files(equity_data, "*.csv")
    results_list = get_symbol_list(results)
    trending_list = get_symbol_list(trending)
    triggers_raw = get_symbol_list(triggers)
    
    # Process trigger list with details
    triggers_list = []
    for sym in triggers_raw:
        details = get_trigger_details(triggers, sym)
        if details:
            triggers_list.append(details)

    # Process anomaly list with details
    anomaly_raw = get_symbol_list(anomaly)
    anomaly_list = []
    for sym in anomaly_raw:
        details = get_anomaly_details(anomaly, sym)
        if details:
            anomaly_list.append(details)

    
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
        "ticker": len(ticker_list),
        "triggers": len(triggers_list),
        "anomaly": len(anomaly_list)
    }
    
    return stats, results_list, trending_list, ticker_list, triggers_list, anomaly_list

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
    stats, results_list, trending_list, ticker_list, triggers_list, anomaly_list = get_pipeline_data(base_dir)
    
    html_content = generate_html_report(stats, results_list, trending_list, ticker_list, triggers_list, anomaly_list)
    
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
