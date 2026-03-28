"""
html_renderer.py
Presentation Layer for Automated Equity Reports.
Generates premium HTML email bodies with professional layouts, 
conditional coloring, and intelligent summaries.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from src.constants import email_constants as const

class HTMLRenderer:
    """
    Renders the final report into professional, premium HTML.
    Complies with system instructions for high-quality visuals and modern UI.
    """
    
    def __init__(self):
        # IST Time setup (as required by original logic)
        self.now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        self.date_str = self.now.strftime('%d-%m-%Y %H:%M')

    def render_full_report(self, stats: Dict, symbol_data: Dict[str, List[str]], 
                           anomaly_details: List[Dict]) -> str:
        """Main orchestrator for generating the full HTML report body."""
        summary_html = self._render_smart_summary(stats, symbol_data["ticker"])
        anomaly_section = self._render_anomaly_section(anomaly_details)
        
        # High-level template wrapping
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: {const.COLOR_BACKGROUND}; color: {const.COLOR_TEXT}; margin: 0; padding: 20px; }}
                .container {{ max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; border: 1px solid {const.COLOR_BORDER}; }}
                .header {{ border-bottom: 2px solid {const.COLOR_BULLISH}; margin-bottom: 20px; padding-bottom: 10px; }}
                .section {{ margin-top: 25px; }}
                .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #4a5568; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid {const.COLOR_BORDER}; }}
                th {{ background-color: #edf2f7; color: #4a5568; font-size: 13px; text-transform: uppercase; }}
                .pill {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
                .bullish {{ background-color: #c6f6d5; color: {const.COLOR_BULLISH}; }}
                .bearish {{ background-color: #fed7d7; color: {const.COLOR_BEARISH}; }}
                .badge {{ background-color: #edf2f7; color: #4a5568; padding: 2px 6px; border-radius: 12px; font-size: 11px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0; color: {const.COLOR_BULLISH};">Equity Analysis Report</h1>
                    <p style="margin: 5px 0; color: #718096; font-size: 14px;">Market Insight Date: {self.date_str}</p>
                </div>
                
                {summary_html}
                
                {anomaly_section}
                
                <div class="section" style="margin-top: 40px; border-top: 1px solid {const.COLOR_BORDER}; padding-top: 20px;">
                    <p style="font-size: 12px; color: #a0aec0; text-align: center;">
                        This is an automated system report. Please verify signals independently before trading.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

    def _render_smart_summary(self, stats: Dict, ticker_list: List[str]) -> str:
        """Generates the intelligent textual summary portion of the report."""
        vsa_count = stats.get("vsa", 0)
        trending_count = stats.get("trending", 0)
        ticker_count = len(ticker_list)
        
        # Branch logic based on thresholds (Centralized in email_constants)
        if trending_count > const.TRENDING_THRESHOLD_HIGH:
            landscape = f"The latest market analysis reveals a highly dynamic landscape with strong momentum across many sectors."
        else:
            landscape = f"The latest market analysis reveals a selective landscape with localized momentum."
            
        outlook = "🚀 Outlook: "
        if ticker_count >= const.TICKER_THRESHOLD_HIGH:
            outlook += "Strong Positive Momentum. High-conviction signals detected."
        else:
            outlook += "Consolidative/Selective. Exercise patience on new entries."
            
        return f"""
        <div class="section">
            <div class="section-title">📊 Executive Summary</div>
            <p style="line-height: 1.6; color: #4a5568;">
                Our pipeline processed <strong>{vsa_count}</strong> symbols today. {landscape} 
                We tracked <strong>{trending_count}</strong> trending assets and <strong>{ticker_count}</strong> 
                high-probability tickers for immediate observation.
            </p>
            <p style="padding: 10px; background-color: #f7fafc; border-left: 4px solid {const.COLOR_BULLISH}; font-weight: bold;">
                {outlook}
            </p>
        </div>
        """

    def _render_anomaly_section(self, anomalies: List[Dict]) -> str:
        """Renders the Anomaly V2 Table."""
        if not anomalies:
            return ""
            
        rows = ""
        for a in anomalies:
            sentiment_class = "bullish" if a['sentiment'] == "Bullish" else "bearish" if a['sentiment'] == "Bearish" else ""
            rows += f"""
            <tr>
                <td style="font-weight: bold;">{a['symbol']}</td>
                <td style="color: {const.COLOR_BEARISH if a['drop_pct'] < 0 else const.COLOR_BULLISH};">{a['drop_pct']:.1f}%</td>
                <td>{a['pattern']}</td>
                <td><span class="pill {sentiment_class}">{a['sentiment']}</span></td>
            </tr>
            """
            
        return f"""
        <div class="section">
            <div class="section-title">🚀 VSA Volume Anomalies (V2)</div>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Vol Change</th>
                        <th>Pattern Identified</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
