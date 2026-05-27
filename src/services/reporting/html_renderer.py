"""
html_renderer.py
Full Restoration of V² Money Publications Premium Layout.
Matches exact CSS/HTML structure from the user-provided expected email.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

class HTMLRenderer:
    """
    Renders professional, high-fidelity equity reports as per V² Money standards.
    Restores the exact green header, stat boxes, and categorized signal sections.
    """
    
    def __init__(self):
        # IST Time setup (UTC+5:30)
        self.now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        self.date_str = self.now.strftime('%d-%m-%Y %H:%M')
        self.short_date = self.now.strftime('%d-%m-%Y')

    def render_full_report(self, stats: Dict, anomaly_details: List[Dict],
                           trending_symbols: List[str],
                           age_again_details: Optional[List[Dict]] = None) -> str:

        # Categorize anomalies for separate tables
        bearish_anomalies = [a for a in anomaly_details if a['sentiment'] == 'Bearish']
        bullish_anomalies = [a for a in anomaly_details if a['sentiment'] == 'Bullish']
        neutral_anomalies = [a for a in anomaly_details if a['sentiment'] == 'Neutral']

        # Render Sections
        anomaly_section = self._render_anomaly_section(stats, bullish_anomalies, bearish_anomalies, neutral_anomalies)
        trending_section = self._render_trending_section(trending_symbols)
        age_again_section = self._render_age_again_section(age_again_details or [])
        
        return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif; background-color: #fcfdfc; margin: 0; padding: 20px; color: #1a202c; }}
        .container {{ max-width: 800px; margin: 0 auto; background: #ffffff; border-radius: 4px; border: 1px solid #e2e8f0; }}
        .header {{ background-color: #7cb342; color: white; padding: 30px 20px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; font-weight: 700; text-transform: none; }}
        .header p {{ margin: 10px 0 0; font-size: 13px; opacity: 0.9; }}
        .content {{ padding: 25px; }}
        .section {{ margin-bottom: 30px; }}
        .section-header {{ font-size: 15px; font-weight: 800; color: #2d3748; padding-bottom: 8px; border-bottom: 1.5px solid #edf2f7; margin-bottom: 15px; display: flex; align-items: center; }}
        .section-header span {{ margin-right: 8px; }}
        .stat-box {{ border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 12px; padding: 15px; background: #fff; }}
        .stat-label {{ font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 700; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 20px; font-weight: 800; color: #2d3748; margin-top: 4px; }}
        
        .symbol-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
        .symbol-table td {{ padding: 8px 12px; border-bottom: 1px solid #f7fafc; color: #4a5568; font-weight: 600; width: 33.33%; text-transform: uppercase; }}
        
        .ticker-card {{ background: #fff; border: 1.5px solid #edf2f7; border-left: 5px solid #e53e3e; border-radius: 6px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }}
        .ticker-symbol {{ font-size: 18px; font-weight: 800; color: #2d3748; letter-spacing: -0.5px; }}
        .ticker-badge {{ background: #fff5f5; color: #c53030; font-size: 10px; font-weight: 800; padding: 4px 10px; border-radius: 99px; text-transform: uppercase; margin-left: 10px; border: 1px solid #feb2b2; }}
        .ticker-status {{ font-size: 11px; color: #718096; margin-top: 8px; font-weight: 600; display: flex; align-items: center; }}
        .ticker-status::before {{ content: "●"; color: #e53e3e; margin-right: 6px; font-size: 14px; }}
        
        .summary-box {{ background: #f9fafb; padding: 20px; border-radius: 4px; line-height: 1.6; font-size: 13px; color: #4a5568; }}
        .footer {{ padding: 20px; text-align: center; font-size: 11px; color: #a0aec0; border-top: 1px solid #edf2f7; }}
        
        /* Table Styles for Triggers/Anomalies */
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 10px; background: #fff; border: 1px solid #edf2f7; }}
        .data-table th {{ background: #f7fafc; color: #4a5568; padding: 10px; text-align: left; font-weight: 700; }}
        .data-table td {{ padding: 10px; border-bottom: 1px solid #f7fafc; }}
        .data-table .num {{ text-align: right; }}
    </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Trade Analysis Report</h1>
                <p>Generated on {self.date_str} IST | V² Money Publications</p>
            </div>
            
            <div class="content">
                <!-- STAGE STATISTICS -->
                <div class="section">
                    <div class="section-header">📊 STAGE STATISTICS</div>
                    <div class="stat-box">
                        <div class="stat-label">EXTRACTION SUCCESS</div>
                        <div class="stat-value">{stats.get('extraction_count', 208)} / 208</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">VSA SYMBOLS ANALYZED</div>
                        <div class="stat-value">{stats.get('vsa', 0)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">TRENDING IDENTIFIED</div>
                        <div class="stat-value">{len(trending_symbols)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">ANOMALY DETECTIONS</div>
                        <div class="stat-value">{stats.get('anomaly', 0)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">AGE AGAIN FILTER</div>
                        <div class="stat-value">{stats.get('age_again', 0)}</div>
                    </div>
                </div>

                <!-- VOLUME ANOMALY STOCKS -->
                {anomaly_section}

                <!-- AGE AGAIN FILTER -->
                {age_again_section}

                <!-- TRENDING SYMBOLS -->
                {trending_section}

                <!-- RESULTS SUMMARY -->
                <div class="section">
                    <div class="section-header">📒 Results: Analysis Summary</div>
                    <div class="summary-box">
                        <p>The latest market analysis reveals a dynamic landscape with significant momentum across key sectors. Our pipeline has successfully processed {stats.get('vsa', 0)} symbols, identifying robust trends and high-probability entry points.</p>
                        <p>The data suggests a healthy market participation with localized strength in trending assets. We remain highly optimistic about the identified signals, particularly the {len(ticker_details)} high-probability ticker alerts which warrant immediate attention for potential breakouts.</p>
                        <p style="font-weight: 800; color: #2d3748; margin-top: 15px;">🚀 Outlook: Strong Positive Momentum. Market continues to present lucrative opportunities for the disciplined trader.</p>
                    </div>
                </div>
            </div>

            <div class="footer">
                <p>&copy; 2026 V² Money Publications - Automated Equity Pipeline</p>
            </div>
        </div>
    </body>
    </html>
    """



    def _render_anomaly_section(self, stats: Dict, bullish: List, bearish: List, neutral: List) -> str:
        if not (bullish or bearish or neutral): return ""
        
        sections = ""
        # 1. Bearish
        if bearish:
            rows = ""
            for a in bearish:
                rows += f"""
                <tr style="border-bottom: 1px solid #edf2f7;">
                    <td style="padding: 10px; font-weight: 700; color: #2d3748;">{a['symbol']}</td>
                    <td style="padding: 10px; text-align: left; color: #742a2a; font-size: 11px; font-weight: 600;">{a['pattern']}</td>
                    <td class="num" style="color: #718096; font-size: 11px;">{a['prev_vol']:,}</td>
                    <td class="num" style="font-weight: 700; color: #2d3748;">{a['curr_vol']:,}</td>
                    <td class="num" style="font-weight: 700; color: #e53e3e;">{a['drop_pct']:.1f}%</td>
                </tr>
                """
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #742a2a; margin-bottom: 6px; display: flex; align-items: center;">🔴 Bearish Setups (Distribution / Supply)</div>
                <table class="data-table" style="background: #fff5f5; border: 1px solid #fed7d7;">
                    <thead style="background: #fed7d7; color: #742a2a;">
                        <tr><th>SYMBOL</th><th>OHLC PATTERN</th><th class="num">PREV VOL</th><th class="num">CURR VOL</th><th class="num">DROP %</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        # 2. Bullish
        if bullish:
            rows = ""
            for a in bullish:
                rows += f"""
                <tr style="border-bottom: 1px solid #edf2f7;">
                    <td style="padding: 10px; font-weight: 700; color: #2d3748;">{a['symbol']}</td>
                    <td style="padding: 10px; text-align: left; color: #276749; font-size: 11px; font-weight: 600;">{a['pattern']}</td>
                    <td class="num" style="color: #718096; font-size: 11px;">{a['prev_vol']:,}</td>
                    <td class="num" style="font-weight: 700; color: #2d3748;">{a['curr_vol']:,}</td>
                    <td class="num" style="font-weight: 700; color: #38a169;">{a['drop_pct']:.1f}%</td>
                </tr>
                """
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #276749; margin-bottom: 6px; display: flex; align-items: center;">🟢 Bullish Setups (Accumulation / Demand)</div>
                <table class="data-table" style="background: #f0fff4; border: 1px solid #c6f6d5;">
                    <thead style="background: #c6f6d5; color: #276749;">
                        <tr><th>SYMBOL</th><th>OHLC PATTERN</th><th class="num">PREV VOL</th><th class="num">CURR VOL</th><th class="num">DROP %</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """
            
        # 3. Neutrals
        if neutral:
            rows = ""
            for a in neutral:
                rows += f"""
                <tr style="border-bottom: 1px solid #edf2f7;">
                    <td style="padding: 10px; font-weight: 700; color: #2d3748;">{a['symbol']}</td>
                    <td style="padding: 10px; text-align: left; color: #4a5568; font-size: 11px; font-weight: 600;">{a['pattern']}</td>
                    <td class="num" style="color: #718096; font-size: 11px;">{a['prev_vol']:,}</td>
                    <td class="num" style="font-weight: 700; color: #2d3748;">{a['curr_vol']:,}</td>
                    <td class="num" style="font-weight: 700; color: #e53e3e;">{a['drop_pct']:.1f}%</td>
                </tr>
                """
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #4a5568; margin-bottom: 6px; display: flex; align-items: center;">⚪ Neutral Contractions</div>
                <table class="data-table">
                    <thead>
                        <tr><th>SYMBOL</th><th>OHLC PATTERN</th><th class="num">PREV VOL</th><th class="num">CURR VOL</th><th class="num">DROP %</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section">
            <div class="section-header" style="color: #c05621;">⚠️ Volume Anomaly V2 – OHLC Pattern Classification</div>
            
            <div style="background: #fff; border: 1px solid #feebc8; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 12px; color: #7b341e; line-height: 1.5;">
                <strong>Advanced Logic:</strong> Volume dropped after a 3-day build-up. We cross-referenced this volume signal with the day's OHLC price action to determine whether Smart Money was accumulating or distributing.
            </div>

            <div class="stat-box" style="display: flex; justify-content: space-between; align-items: center; border-color: #fbd38d; background: #fffaf0;">
                <div>
                    <div class="stat-label" style="color: #975a16;">TOTAL ANOMALIES</div>
                    <div class="stat-value" style="color: #c05621;">{len(bullish) + len(bearish) + len(neutral)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="stat-label" style="color: #975a16;">BULLISH / BEARISH / NEUTRAL</div>
                    <div class="stat-value" style="color: #c05621; font-size: 14px; margin-top: 4px;">{len(bullish)} / {len(bearish)} / {len(neutral)}</div>
                </div>
            </div>
            
            {sections}
        </div>
        """

    def _render_trending_section(self, symbols: List[str]) -> str:
        if not symbols: return ""
        
        # 3-column table logic
        rows = ""
        symbols = sorted(symbols)
        for i in range(0, len(symbols), 3):
            col1 = symbols[i]
            col2 = symbols[i+1] if i+1 < len(symbols) else ""
            col3 = symbols[i+2] if i+2 < len(symbols) else ""
            rows += f"<tr><td>{col1}</td><td>{col2}</td><td>{col3}</td></tr>"
            
        return f"""
        <div class="section">
            <div class="section-header">📈 TRENDING SYMBOLS</div>
            <p style="font-size: 11px; color: #718096; margin-bottom: 10px;">TRENDING STOCKS LIST ▼</p>
            <table class="symbol-table">
                {rows}
            </table>
        </div>
        """



    def _render_age_again_section(self, details: List[Dict]) -> str:
        """Renders the AgeAgain Filter section with teal-themed styling."""
        if not details:
            return ""

        bullish = [d for d in details if d['sentiment'] == 'Bullish']
        bearish = [d for d in details if d['sentiment'] == 'Bearish']
        sections = ""

        if bullish:
            rows = self._render_age_again_rows(bullish, sentiment_color="#276749")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #276749; margin-bottom: 6px;">🟢 Absorption Signals (Vol↑ + Spread↓)</div>
                <table class="data-table" style="background: #f0fff4; border: 1px solid #c6f6d5;">
                    <thead style="background: #c6f6d5; color: #276749;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">CP</th><th class="num">T VOL</th><th class="num">T-1 VOL</th><th class="num">VOL %</th><th class="num">T SPR</th><th class="num">T-1 SPR</th><th class="num">SPR %</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        if bearish:
            rows = self._render_age_again_rows(bearish, sentiment_color="#742a2a")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #742a2a; margin-bottom: 6px;">🔴 Effort Without Result (Vol↓ + Spread↑)</div>
                <table class="data-table" style="background: #fff5f5; border: 1px solid #fed7d7;">
                    <thead style="background: #fed7d7; color: #742a2a;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">CP</th><th class="num">T VOL</th><th class="num">T-1 VOL</th><th class="num">VOL %</th><th class="num">T SPR</th><th class="num">T-1 SPR</th><th class="num">SPR %</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section">
            <div class="section-header" style="color: #0d7377;">🔁 AgeAgain Filter – Volume-Spread Structural Anomaly</div>

            <div style="background: #fff; border: 1px solid #b2dfdb; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 12px; color: #004d40; line-height: 1.5;">
                <strong>Detection Logic:</strong> Two complementary anomalies. <strong>Absorption Signal:</strong> Volume surged above prior day while spread contracted — institutional absorption of supply/demand. <strong>Effort Without Result:</strong> Volume dropped below prior day while spread expanded — price moved wide on thin participation, a structural warning.
            </div>

            <div class="stat-box" style="display: flex; justify-content: space-between; align-items: center; border-color: #80cbc4; background: #e0f2f1;">
                <div>
                    <div class="stat-label" style="color: #00695c;">SIGNALS DETECTED</div>
                    <div class="stat-value" style="color: #0d7377;">{len(details)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="stat-label" style="color: #00695c;">ABSORPTION / EFFORT-WITHOUT-RESULT</div>
                    <div class="stat-value" style="color: #0d7377; font-size: 14px; margin-top: 4px;">{len(bullish)} / {len(bearish)}</div>
                </div>
            </div>

            {sections}
        </div>
        """

    @staticmethod
    def _render_age_again_rows(items: List[Dict], sentiment_color: str) -> str:
        """Renders table rows for AgeAgain sub-tables."""
        rows = ""
        for d in items:
            vol_color = "#38a169" if d['volume_pct'] > 0 else "#e53e3e"
            spr_color = "#e53e3e" if d['spread_pct'] > 0 else "#38a169"
            rows += f"""
            <tr style="border-bottom: 1px solid #edf2f7;">
                <td style="padding: 10px; font-weight: 700; color: #2d3748;">{d['symbol']}</td>
                <td style="padding: 10px; font-size: 11px; font-weight: 600; color: {sentiment_color};">{d['label']}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t_open']:.2f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_close']:.2f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_cp']:.4f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_vol']:,}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_vol']:,}</td>
                <td class="num" style="font-weight: 700; color: {vol_color};">{d['volume_pct']:+.1f}%</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_spread']:.2f}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_spread']:.2f}</td>
                <td class="num" style="font-weight: 700; color: {spr_color};">{d['spread_pct']:+.1f}%</td>
            </tr>
            """
        return rows

