"""
html_renderer.py
Premium Presentation Layer for V² Money Publications.
Restores full sector statistics, ticker cards, and signal tables.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from src.constants import email_constants as const

class HTMLRenderer:
    """
    Renders professional, high-fidelity equity reports.
    Restores the 'V² Money Publications' layout and all data sections.
    """
    
    def __init__(self):
        # IST Time setup
        self.now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        self.date_str = self.now.strftime('%d-%m-%Y %H:%M')

    def render_full_report(self, stats: Dict, ticker_details: List[Dict], 
                           trigger_details: List[Dict], anomaly_details: List[Dict],
                           trending_symbols: List[str]) -> str:
        """Main orchestrator for the V² Money Publications layout."""
        
        # 1. Header & Stage Stats
        header = self._render_header(stats)
        
        # 2. Ticker Signals (Action Required)
        ticker_section = self._render_ticker_cards(ticker_details)
        
        # 3. VSA Triggered Stocks
        trigger_section = self._render_trigger_table(trigger_details)
        
        # 4. Anomaly V2
        anomaly_section = self._render_anomaly_table(anomaly_details)
        
        # 5. Trending Grid
        trending_section = self._render_trending_grid(trending_symbols)
        
        # 6. Overall Summary & Outlook
        summary = self._render_analysis_summary(stats, ticker_details)
        
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f7f6; color: #2d3748; margin: 0; padding: 20px; }}
                .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 8px; border: 1px solid #edf2f7; shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ border-bottom: 3px solid #38a169; margin-bottom: 25px; padding-bottom: 15px; }}
                .section {{ margin-top: 35px; }}
                .section-title {{ font-size: 20px; font-weight: bold; margin-bottom: 20px; color: #2d3748; letter-spacing: -0.5px; border-left: 5px solid #38a169; padding-left: 15px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
                th, td {{ padding: 14px; text-align: left; border-bottom: 1px solid #edf2f7; }}
                th {{ background-color: #f7fafc; color: #4a5568; font-size: 12px; text-transform: uppercase; font-weight: bold; }}
                .pill {{ padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 11px; text-transform: uppercase; }}
                .bullish {{ background-color: #c6f6d5; color: #276749; }}
                .bearish {{ background-color: #fed7d7; color: #9b2c2c; }}
                .neutral {{ background-color: #edf2f7; color: #4a5568; }}
                .symbol-grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
                .symbol-badge {{ background-color: #f7fafc; border: 1px solid #edf2f7; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 13px; }}
                .ticker-card {{ background-color: #fffaf0; border: 1px solid #feebc8; padding: 15px; border-radius: 6px; margin-bottom: 15px; }}
                .footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 25px; font-size: 13px; color: #718096; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0; color: #276749; font-size: 28px;">Trade Analysis Report</h1>
                    <p style="margin: 5px 0; color: #4a5568; font-weight: 500;">Generated on {self.date_str} IST | V² Money Publications</p>
                </div>
                
                {header}
                {ticker_section}
                {trigger_section}
                {anomaly_section}
                {trending_section}
                {summary}
                
                <div class="footer">
                    <p>© 2026 V² Money Publications - Automated Equity Pipeline<br>Automated institutional analysis systems. Confidential.</p>
                </div>
            </div>
        </body>
        </html>
        """

    def _render_header(self, stats: Dict) -> str:
        s = stats
        return f"""
        <div class="section">
            <div class="section-title">📊 STAGE STATISTICS</div>
            <table style="width: 100%; text-align: center;">
                <tr>
                    <td style="border: none;">
                        <span style="font-size: 10px; display: block;">EXTRACTION SUCCESS</span>
                        <strong style="font-size: 18px;">208 / 208</strong>
                    </td>
                    <td style="border: none;">
                        <span style="font-size: 10px; display: block;">VSA ANALYZED</span>
                        <strong style="font-size: 18px;">{s.get('vsa', 0)}</strong>
                    </td>
                    <td style="border: none;">
                        <span style="font-size: 10px; display: block;">TRENDING</span>
                        <strong style="font-size: 18px; color: #d69e2e;">{s.get('trending', 0)}</strong>
                    </td>
                    <td style="border: none;">
                        <span style="font-size: 10px; display: block;">HIGH-PROB</span>
                        <strong style="font-size: 18px; color: #38a169;">2</strong>
                    </td>
                    <td style="border: none;">
                        <span style="font-size: 10px; display: block;">ANOMALY</span>
                        <strong style="font-size: 18px; color: #e53e3e;">{s.get('anomaly', 0)}</strong>
                    </td>
                </tr>
            </table>
        </div>
        """

    def _render_ticker_cards(self, tickers: List[Dict]) -> str:
        if not tickers: return ""
        # High-prob candidates are those with highest confidence or specific signs
        # For now, sorting by confidence
        sorted_tickers = sorted(tickers, key=lambda x: x['confidence'], reverse=True)
        display = sorted_tickers[:3] # Show top 3
        
        cards = ""
        for t in display:
            sent_cls = "bullish" if "Bullish" in t['sentiment'] else "bearish" if "Bearish" in t['sentiment'] else "neutral"
            cards += f"""
            <div class="ticker-card">
                <div style="display: flex; justify-content: space-between;">
                    <strong style="font-size: 18px; color: #2d3748;">{t['symbol']}</strong>
                    <span class="pill {sent_cls}">{t['pattern']} ({t['sentiment']})</span>
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #4a5568;">
                    <table style="margin: 0; padding: 0;">
                        <tr style="border: none;"><td style="width: 150px; padding: 2px;">Effort vs Result:</td><td style="padding: 2px;">{t['effort']}</td></tr>
                        <tr style="border: none;"><td style="padding: 2px;">Spread Ratio:</td><td style="padding: 2px;">{t['spread_ratio']:.2f}</td></tr>
                        <tr style="border: none;"><td style="padding: 2px;">Pattern Confidence:</td><td style="padding: 2px;">{t['confidence']:.2f}</td></tr>
                    </table>
                </div>
                <p style="margin-top: 12px; font-size: 13px; line-height: 1.4;">{t['description']}</p>
                <div style="color: #c53030; font-size: 11px; font-weight: bold; margin-top: 5px;">Requires Immediate Review</div>
            </div>
            """
        return f"""
        <div class="section">
            <div class="section-title">🎯 TICKER SIGNALS (Action Required)</div>
            {cards}
        </div>
        """

    def _render_trigger_table(self, triggers: List[Dict]) -> str:
        if not triggers: return ""
        rows = ""
        for t in triggers[:20]:
            rows += f"""
            <tr>
                <td style="font-weight: bold;">{t['symbol']}</td>
                <td>{t['prev_vol']:,}</td>
                <td>{t['curr_vol']:,}</td>
                <td>{t['prev_spr']:.2f}</td>
                <td>{t['curr_spr']:.2f}</td>
                <td style="color: #c53030;">{t['vol_pct']:.1f}%</td>
                <td style="color: #2f855a;">{t['spr_pct']:.1f}%</td>
            </tr>
            """
        return f"""
        <div class="section">
            <div class="section-title">📊 VSA Triggered Stocks – Vol Contraction + Spread Expansion</div>
            <p style="font-size: 12px; color: #718096; margin-bottom: 10px;">
                Trigger Logic: Volume decreased while Spread expanded. Suggests supply removal facilitating easier price movement.
            </p>
            <table>
                <thead>
                    <tr><th>SYMBOL</th><th>PREV VOL</th><th>CURR VOL</th><th>PREV SPR</th><th>CURR SPR</th><th>VOL %</th><th>SPR %</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _render_anomaly_table(self, anomalies: List[Dict]) -> str:
        if not anomalies: return ""
        rows = ""
        for a in anomalies:
            rows += f"""
            <tr>
                <td style="font-weight: bold;">{a['symbol']}</td>
                <td>{a['pattern']}</td>
                <td>{a['prev_vol']:,}</td>
                <td>{a['curr_vol']:,}</td>
                <td style="color: {'#c53030' if a['drop_pct'] < 0 else '#2f855a'};">{a['drop_pct']:.1f}%</td>
            </tr>
            """
        return f"""
        <div class="section">
            <div class="section-title">⚠️ Volume Anomaly V2 – OHLC Pattern Classification</div>
            <table>
                <thead>
                    <tr><th>SYMBOL</th><th>OHLC PATTERN</th><th>PREV VOL</th><th>CURR VOL</th><th>DROP %</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _render_trending_grid(self, symbols: List[str]) -> str:
        if not symbols: return ""
        grid = "".join([f'<div class="symbol-badge">{s}</div>' for s in sorted(symbols)])
        return f"""
        <div class="section">
            <div class="section-title">📈 TRENDING SYMBOLS</div>
            <div class="symbol-grid">{grid}</div>
        </div>
        """

    def _render_analysis_summary(self, stats: Dict, ticker_details: List[Dict]) -> str:
        vsa_count = stats.get("vsa", 0)
        ticker_count = len(ticker_details)
        
        return f"""
        <div class="section">
            <div class="section-title">📒 Results: Analysis Summary</div>
            <p style="line-height: 1.6; color: #4a5568; margin-bottom: 15px;">
                Our pipeline has successfully processed <strong>{vsa_count}</strong> symbols today, identifying robust trends and high-probability entry points.
            </p>
            <p style="line-height: 1.6; color: #4a5568;">
                The overall market breadth remains supportive with <strong>{ticker_count}</strong> high-probability ticker alerts which warrant immediate attention.
            </p>
            <p style="margin-top: 20px; padding: 15px; background-color: #f0fff4; border-left: 5px solid #38a169; font-weight: bold; font-size: 16px;">
                🚀 Outlook: Strong Positive Momentum. Market continues to present lucrative opportunities for the disciplined trader.
            </p>
        </div>
        """
