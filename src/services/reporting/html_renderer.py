"""
html_renderer.py
V² Money Publications Premium Layout.
Renders EigenFilter, AgeAgain, Trending, and Results Summary sections.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

class HTMLRenderer:
    """
    Renders professional, high-fidelity equity reports as per V² Money standards.
    Sections: Stage Stats, EigenFilter, AgeAgain Filter, Trending Symbols, Results Summary.
    """
    
    def __init__(self):
        # IST Time setup (UTC+5:30)
        self.now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        self.date_str = self.now.strftime('%d-%m-%Y %H:%M')
        self.short_date = self.now.strftime('%d-%m-%Y')

    def render_full_report(self, stats: Dict, eigen_details: List[Dict],
                           weekly_eigen_details: Optional[List[Dict]] = None,
                           monthly_eigen_details: Optional[List[Dict]] = None) -> str:

        # Render Sections
        eigen_section = self._render_eigen_section(eigen_details)
        weekly_eigen_section = self._render_weekly_eigen_section(weekly_eigen_details or [])
        monthly_eigen_section = self._render_monthly_eigen_section(monthly_eigen_details or [])
        
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
        
        .summary-box {{ background: #f9fafb; padding: 20px; border-radius: 4px; line-height: 1.6; font-size: 13px; color: #4a5568; }}
        .footer {{ padding: 20px; text-align: center; font-size: 11px; color: #a0aec0; border-top: 1px solid #edf2f7; }}
        
        /* Table Styles */
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
                        <div class="stat-label">EIGEN FILTER</div>
                        <div class="stat-value">{stats.get('eigen_filter', 0)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">WEEKLY EIGEN FILTER</div>
                        <div class="stat-value">{stats.get('weekly_eigen', 0)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">MONTHLY EIGEN FILTER</div>
                        <div class="stat-value">{stats.get('monthly_eigen', 0)}</div>
                    </div>
                </div>

                <!-- EIGEN FILTER -->
                {eigen_section}

                <!-- WEEKLY EIGEN FILTER -->
                {weekly_eigen_section}

                <!-- MONTHLY EIGEN FILTER -->
                {monthly_eigen_section}

            </div>

            <div class="footer">
                <p>&copy; 2026 V² Money Publications - Automated Equity Pipeline</p>
            </div>
        </div>
    </body>
    </html>
    """

    def _render_eigen_section(self, details: List[Dict]) -> str:
        """Renders the EigenFilter section with purple-themed styling."""
        if not details:
            return ""

        bullish = [d for d in details if d['sentiment'] == 'Bullish']
        bearish = [d for d in details if d['sentiment'] == 'Bearish']
        sections = ""

        if bullish:
            rows = self._render_eigen_rows(bullish, sentiment_color="#276749")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #276749; margin-bottom: 6px;">🟢 Bullish Classifications</div>
                <table class="data-table" style="background: #f0fff4; border: 1px solid #c6f6d5;">
                    <thead style="background: #c6f6d5; color: #276749;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">GAP</th><th class="num">CP</th><th class="num">T-1 CP</th><th class="num">ΔCP</th><th class="num">T VOL</th><th class="num">T-1 VOL</th><th class="num">VOL Δ%</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        if bearish:
            rows = self._render_eigen_rows(bearish, sentiment_color="#742a2a")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #742a2a; margin-bottom: 6px;">🔴 Bearish Classifications</div>
                <table class="data-table" style="background: #fff5f5; border: 1px solid #fed7d7;">
                    <thead style="background: #fed7d7; color: #742a2a;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">GAP</th><th class="num">CP</th><th class="num">T-1 CP</th><th class="num">ΔCP</th><th class="num">T VOL</th><th class="num">T-1 VOL</th><th class="num">VOL Δ%</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section">
            <div class="section-header" style="color: #6b21a8;">🔮 EigenFilter – Volume-Amplitude OHLC Divergence</div>

            <div style="background: #fff; border: 1px solid #e9d5ff; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 12px; color: #581c87; line-height: 1.5;">
                <strong>Detection Logic:</strong> Identifies institutional volume-price divergence via three-part filtering: volume surge detection, gap-direction analysis, and close-position drift validation. Stocks are classified into four high-conviction categories.
            </div>

            <div class="stat-box" style="display: flex; justify-content: space-between; align-items: center; border-color: #d8b4fe; background: #faf5ff;">
                <div>
                    <div class="stat-label" style="color: #7e22ce;">SIGNALS DETECTED</div>
                    <div class="stat-value" style="color: #6b21a8;">{len(details)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="stat-label" style="color: #7e22ce;">BULLISH / BEARISH</div>
                    <div class="stat-value" style="color: #6b21a8; font-size: 14px; margin-top: 4px;">{len(bullish)} / {len(bearish)}</div>
                </div>
            </div>

            {sections}
        </div>
        """

    @staticmethod
    def _render_eigen_rows(items: List[Dict], sentiment_color: str) -> str:
        """Renders table rows for EigenFilter sub-tables."""
        rows = ""
        for d in items:
            vol_color = "#38a169" if d['vol_delta_pct'] > 0 else "#e53e3e"
            delta_cp_color = "#38a169" if d['delta_cp'] > 0 else "#e53e3e"
            rows += f"""
            <tr style="border-bottom: 1px solid #edf2f7;">
                <td style="padding: 10px; font-weight: 700; color: #2d3748;">{d['symbol']}</td>
                <td style="padding: 10px; font-size: 11px; font-weight: 600; color: {sentiment_color};">{d['label']}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t_open']:.2f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_close']:.2f}</td>
                <td class="num" style="font-weight: 600; color: #6b21a8; font-size: 11px;">{d['gap_dir']}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_cp']:.4f}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_cp']:.4f}</td>
                <td class="num" style="font-weight: 700; color: {delta_cp_color};">{d['delta_cp']:+.4f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_vol']:,}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_vol']:,}</td>
                <td class="num" style="font-weight: 700; color: {vol_color};">{d['vol_delta_pct']:+.1f}%</td>
            </tr>
            """
        return rows

    def _render_weekly_eigen_section(self, details: List[Dict]) -> str:
        """Renders the Weekly EigenFilter section with sky-blue-themed styling."""
        if not details:
            return ""

        bullish = [d for d in details if d['sentiment'] == 'Bullish']
        bearish = [d for d in details if d['sentiment'] == 'Bearish']
        sections = ""

        if bullish:
            rows = self._render_weekly_eigen_rows(bullish, sentiment_color="#276749")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #276749; margin-bottom: 6px;">🟢 Bullish Weekly Classifications</div>
                <table class="data-table" style="background: #f0fff4; border: 1px solid #c6f6d5;">
                    <thead style="background: #c6f6d5; color: #276749;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">WEEK</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">GAP</th><th class="num">CP</th><th class="num">PREV CP</th><th class="num">ΔCP</th><th class="num">W VOL</th><th class="num">PREV W VOL</th><th class="num">VOL Δ%</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        if bearish:
            rows = self._render_weekly_eigen_rows(bearish, sentiment_color="#742a2a")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #742a2a; margin-bottom: 6px;">🔴 Bearish Weekly Classifications</div>
                <table class="data-table" style="background: #fff5f5; border: 1px solid #fed7d7;">
                    <thead style="background: #fed7d7; color: #742a2a;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">WEEK</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">GAP</th><th class="num">CP</th><th class="num">PREV CP</th><th class="num">ΔCP</th><th class="num">W VOL</th><th class="num">PREV W VOL</th><th class="num">VOL Δ%</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section">
            <div class="section-header" style="color: #0369a1;">🗓️ Weekly EigenFilter – Consolidated Volume-Amplitude Divergence</div>

            <div style="background: #fff; border: 1px solid #bae6fd; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 12px; color: #0c4a6e; line-height: 1.5;">
                <strong>Weekly Consolidation:</strong> Daily OHLCV data is aggregated into weekly candles (Open=first, High=max, Low=min, Close=last, Volume=sum). The same EigenFilter detection logic — volume surge, gap-direction, and close-position drift — is then applied to identify structural shifts on the weekly timeframe.
            </div>

            <div class="stat-box" style="display: flex; justify-content: space-between; align-items: center; border-color: #7dd3fc; background: #f0f9ff;">
                <div>
                    <div class="stat-label" style="color: #0284c7;">WEEKLY SIGNALS</div>
                    <div class="stat-value" style="color: #0369a1;">{len(details)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="stat-label" style="color: #0284c7;">BULLISH / BEARISH</div>
                    <div class="stat-value" style="color: #0369a1; font-size: 14px; margin-top: 4px;">{len(bullish)} / {len(bearish)}</div>
                </div>
            </div>

            {sections}
        </div>
        """

    @staticmethod
    def _render_weekly_eigen_rows(items: List[Dict], sentiment_color: str) -> str:
        """Renders table rows for Weekly EigenFilter sub-tables."""
        rows = ""
        for d in items:
            vol_color = "#38a169" if d['vol_delta_pct'] > 0 else "#e53e3e"
            delta_cp_color = "#38a169" if d['delta_cp'] > 0 else "#e53e3e"
            week_label = d.get('latest_week', 'N/A')
            rows += f"""
            <tr style="border-bottom: 1px solid #edf2f7;">
                <td style="padding: 10px; font-weight: 700; color: #2d3748;">{d['symbol']}</td>
                <td style="padding: 10px; font-size: 11px; font-weight: 600; color: {sentiment_color};">{d['label']}</td>
                <td class="num" style="font-weight: 600; color: #0369a1; font-size: 11px;">{week_label}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t_open']:.2f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_close']:.2f}</td>
                <td class="num" style="font-weight: 600; color: #0369a1; font-size: 11px;">{d['gap_dir']}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_cp']:.4f}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_cp']:.4f}</td>
                <td class="num" style="font-weight: 700; color: {delta_cp_color};">{d['delta_cp']:+.4f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_vol']:,}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_vol']:,}</td>
                <td class="num" style="font-weight: 700; color: {vol_color};">{d['vol_delta_pct']:+.1f}%</td>
            </tr>
            """
        return rows

    def _render_monthly_eigen_section(self, details: List[Dict]) -> str:
        """Renders the Monthly EigenFilter section with indigo-themed styling."""
        if not details:
            return ""

        bullish = [d for d in details if d['sentiment'] == 'Bullish']
        bearish = [d for d in details if d['sentiment'] == 'Bearish']
        sections = ""

        if bullish:
            rows = self._render_monthly_eigen_rows(bullish, sentiment_color="#276749")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #276749; margin-bottom: 6px;">🟢 Bullish Monthly Classifications</div>
                <table class="data-table" style="background: #f0fff4; border: 1px solid #c6f6d5;">
                    <thead style="background: #c6f6d5; color: #276749;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">MONTH</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">GAP</th><th class="num">CP</th><th class="num">PREV CP</th><th class="num">ΔCP</th><th class="num">M VOL</th><th class="num">PREV M VOL</th><th class="num">VOL Δ%</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        if bearish:
            rows = self._render_monthly_eigen_rows(bearish, sentiment_color="#742a2a")
            sections += f"""
            <div style="margin-top: 18px;">
                <div style="font-size: 13px; font-weight: 700; color: #742a2a; margin-bottom: 6px;">🔴 Bearish Monthly Classifications</div>
                <table class="data-table" style="background: #fff5f5; border: 1px solid #fed7d7;">
                    <thead style="background: #fed7d7; color: #742a2a;">
                        <tr><th>SYMBOL</th><th>LABEL</th><th class="num">MONTH</th><th class="num">OPEN</th><th class="num">CLOSE</th><th class="num">GAP</th><th class="num">CP</th><th class="num">PREV CP</th><th class="num">ΔCP</th><th class="num">M VOL</th><th class="num">PREV M VOL</th><th class="num">VOL Δ%</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section">
            <div class="section-header" style="color: #3730a3;">📅 Monthly EigenFilter – Consolidated Volume-Amplitude Divergence</div>

            <div style="background: #fff; border: 1px solid #c7d2fe; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 12px; color: #312e81; line-height: 1.5;">
                <strong>Monthly Consolidation:</strong> Daily OHLCV data is aggregated into monthly candles (Open=first, High=max, Low=min, Close=last, Volume=sum). The same EigenFilter detection logic — volume surge, gap-direction, and close-position drift — is then applied to identify structural shifts on the monthly timeframe.
            </div>

            <div class="stat-box" style="display: flex; justify-content: space-between; align-items: center; border-color: #a5b4fc; background: #eef2ff;">
                <div>
                    <div class="stat-label" style="color: #4338ca;">MONTHLY SIGNALS</div>
                    <div class="stat-value" style="color: #3730a3;">{len(details)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="stat-label" style="color: #4338ca;">BULLISH / BEARISH</div>
                    <div class="stat-value" style="color: #3730a3; font-size: 14px; margin-top: 4px;">{len(bullish)} / {len(bearish)}</div>
                </div>
            </div>

            {sections}
        </div>
        """

    @staticmethod
    def _render_monthly_eigen_rows(items: List[Dict], sentiment_color: str) -> str:
        """Renders table rows for Monthly EigenFilter sub-tables."""
        rows = ""
        for d in items:
            vol_color = "#38a169" if d['vol_delta_pct'] > 0 else "#e53e3e"
            delta_cp_color = "#38a169" if d['delta_cp'] > 0 else "#e53e3e"
            month_label = d.get('latest_month', 'N/A')
            rows += f"""
            <tr style="border-bottom: 1px solid #edf2f7;">
                <td style="padding: 10px; font-weight: 700; color: #2d3748;">{d['symbol']}</td>
                <td style="padding: 10px; font-size: 11px; font-weight: 600; color: {sentiment_color};">{d['label']}</td>
                <td class="num" style="font-weight: 600; color: #3730a3; font-size: 11px;">{month_label}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t_open']:.2f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_close']:.2f}</td>
                <td class="num" style="font-weight: 600; color: #3730a3; font-size: 11px;">{d['gap_dir']}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_cp']:.4f}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_cp']:.4f}</td>
                <td class="num" style="font-weight: 700; color: {delta_cp_color};">{d['delta_cp']:+.4f}</td>
                <td class="num" style="font-weight: 700; color: #2d3748;">{d['t_vol']:,}</td>
                <td class="num" style="color: #718096; font-size: 11px;">{d['t1_vol']:,}</td>
                <td class="num" style="font-weight: 700; color: {vol_color};">{d['vol_delta_pct']:+.1f}%</td>
            </tr>
            """
        return rows
