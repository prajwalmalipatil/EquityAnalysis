"""
nse_client.py
Infrastructure layer for interacting with NSE India's public API.
Handles session management, cookie acquisition via Selenium, and raw data fetching.
"""

import time
import random
import requests
from pathlib import Path
from typing import Optional, Tuple
from contextlib import contextmanager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from urllib.parse import quote

from src.constants import extraction_constants as const
from src.utils.http_client import with_retry
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("nse-client")

class NSEClient:
    """
    Client for extracting data from NSE India.
    Encapsulates session management and resilience patterns.
    """
    
    def __init__(self, use_selenium: bool = True, headless: bool = True):
        self.use_selenium = use_selenium
        self.headless = headless
        self.session: Optional[requests.Session] = None
        self._init_session()

    def _init_session(self):
        """Standard requests session with tuned adapters."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": const.DEFAULT_USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        # Standard warm up to get session cookies
        if self.use_selenium:
            self._warmup_cookies()

    def _warmup_cookies(self):
        """Uses Selenium to get initial session cookies from NSE home page."""
        logger.info("WARMING_UP_NSE_COOKIES", extra={"url": const.NSE_HOME_URL})
        driver = None
        try:
            options = webdriver.ChromeOptions()
            if self.headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.get(const.NSE_HOME_URL)
            
            # Transfer cookies to session
            for cookie in driver.get_cookies():
                self.session.cookies.set(cookie['name'], cookie['value'])
            
            logger.info("COOKIES_ACQUIRED_SUCCESSFULLY")
        except Exception as e:
            logger.error("COOKIE_WARMUP_FAILED", extra={"error": str(e)})
        finally:
            if driver:
                driver.quit()

    @with_retry(max_attempts=const.MAX_RETRIES, base_delay=const.BACKOFF_FACTOR)
    def fetch_historical_data(self, symbol: str, from_date: str, to_date: str) -> requests.Response:
        """
        Fetches historical CSV data for a symbol using the historicalOR endpoint.
        Note: from_date and to_date format: DD-MM-YYYY
        """
        url = (
            f"https://www.nseindia.com/api/historicalOR/generateSecurityWiseHistoricalData"
            f"?from={from_date}&to={to_date}&symbol={quote(symbol)}&type=priceVolumeDeliverable&series=ALL&csv=true"
        )
        
        logger.info("FETCHING_HISTORICAL_DATA", extra={"symbol": symbol, "url": url})
        
        headers = {
            "Accept": "text/csv,*/*;q=0.9",
            "Referer": "https://www.nseindia.com/report-detail/eq_security",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        
        resp = self.session.get(url, headers=headers, timeout=(10, 60))
        resp.raise_for_status()
        
        # Validate content type and size
        content_type = resp.headers.get("Content-Type", "")
        if "text/csv" not in content_type and "application/octet-stream" not in content_type:
            if "<html" in resp.text.lower():
                logger.error("RECEIVED_HTML_NOT_CSV", extra={"symbol": symbol})
                raise ValueError(f"Received HTML instead of CSV for {symbol}")
                
        # Length check: NSE sometimes returns just the header (approx 140-150 bytes)
        # if there's no data for that period/series.
        if len(resp.text.strip().split('\n')) <= 1:
            logger.warning("EMPTY_DATASET_FOR_SYMBOL", extra={
                "symbol": symbol, 
                "from": from_date, 
                "to": to_date,
                "msg": "Only headers received"
            })
            
        return resp

    def close(self):
        """Cleanup resources."""
        if self.session:
            self.session.close()

@contextmanager
def managed_nse_client(use_selenium: bool = True, headless: bool = True):
    """Context manager for NSEClient."""
    client = NSEClient(use_selenium=use_selenium, headless=headless)
    try:
        yield client
    finally:
        client.close()
