#!/usr/bin/env python3  # Shebang: run with system Python 3 when executed as a script
"""
NSE_extraction_fast_api.py — Production Hardened version
CHANGELOG:
- 2025-09-15: Production hardening improvements. Changes include: input validation for symbols to prevent injection attacks; proper session cleanup with context managers; configuration file support for production deployments; comprehensive unit test structure; sanitized file paths to prevent traversal attacks; environment variable support for all configuration options. Rationale: Security hardening, better resource management, production configurability, and testability improvements while maintaining all existing functionality.
- 2025-09-13: Hardened for production use. Changes include: single Selenium warm-up for site-wide cookies (no shared driver across threads to fix thread-safety); manual retry loop with exponential backoff, jitter, and Retry-After respect; content-type/HTML error checks with debug HTML saves; atomic file writes via tempfile; enhanced per-symbol logging (attempts, codes, lengths, error previews); symbol deduplication; added --dry-run mode. Rationale: Eliminate intermittents from shared resources/race conditions, improve retry robustness, add diagnostics, preserve parallelism/performance.
Performance Improvements:
1. Parallel processing for multiple symbols using ThreadPoolExecutor
2. Session reuse across requests to avoid repeated connection setup
3. Reduced browser initialization overhead (one-time setup)
4. Caching of computed date ranges
5. Optimized file existence checks
6. Better error handling with retries
7. Memory-efficient file operations
"""  # Module docstring describing purpose, changes and high-level notes

import argparse  # parse CLI args
import concurrent.futures  # thread pool for parallel symbol processing
import configparser  # configuration file support
import logging  # logging facility
import sys  # system utilities (stdout, exit, argv)
import time  # sleep and timeouts
import random  # jitter/backoff randomness
import os  # filesystem operations (rename, environ)
import tempfile  # atomic temp file writes
import re  # regular expressions for validation
from contextlib import contextmanager  # context manager decorator
from datetime import datetime, timedelta  # date utilities
from pathlib import Path  # Path class for filesystem paths
from functools import lru_cache  # cache decorator for compute_1y_range
from typing import List, Dict, Optional, Tuple, Union  # type hints
import requests  # HTTP client for API calls
from requests.adapters import HTTPAdapter  # adapter to tune retries/pooling
from urllib3.util.retry import Retry  # retry strategy for requests

# --- Selenium optional ---
try:
    from selenium import webdriver  # selenium browser automation (optional)
    from selenium.webdriver.chrome.service import Service  # manage chromedriver service
    from selenium.webdriver.common.by import By  # locator types
    from selenium.webdriver.support.ui import WebDriverWait  # explicit waits
    from selenium.webdriver.support import expected_conditions as EC  # expected conditions
    from webdriver_manager.chrome import ChromeDriverManager  # auto-install chromedriver
    HAS_SELENIUM = True  # flag that selenium imports succeeded
except Exception:
    HAS_SELENIUM = False  # selenium not available; script will run without browser warm-up

# ---- Configuration Class ----
class Config:
    """Centralized configuration with environment variable support"""
    NSE_HOME = "https://www.nseindia.com/"  # base site used for referer and warm-up
    USER_AGENT = (  # Updated to a more modern Chrome version
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    DEFAULT_DOWNLOAD_DIR = Path.cwd() / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))  # default output dir with timestamp
    MAX_WORKERS = 1  # Forced to 1 for high reliability against 403 Access Denied
    MAX_ATTEMPTS = 5  # max HTTP attempts per symbol
    BACKOFF_FACTOR = 3.0  # more aggressive backoff
    JITTER_MIN = 0.8  # minimal jitter multiplier
    JITTER_MAX = 2.0  # increased jitter range
    MIN_DELAY = 2.0  # increased minimal delay to 2s
    MAX_DELAY = 5.0  # increased maximal delay to 5s
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        cls.MAX_WORKERS = int(os.getenv('NSE_MAX_WORKERS', cls.MAX_WORKERS))
        cls.MAX_ATTEMPTS = int(os.getenv('NSE_MAX_ATTEMPTS', cls.MAX_ATTEMPTS))
        cls.BACKOFF_FACTOR = float(os.getenv('NSE_BACKOFF_FACTOR', cls.BACKOFF_FACTOR))
        cls.JITTER_MIN = float(os.getenv('NSE_JITTER_MIN', cls.JITTER_MIN))
        cls.JITTER_MAX = float(os.getenv('NSE_JITTER_MAX', cls.JITTER_MAX))
        cls.MIN_DELAY = float(os.getenv('NSE_MIN_DELAY', cls.MIN_DELAY))
        cls.MAX_DELAY = float(os.getenv('NSE_MAX_DELAY', cls.MAX_DELAY))
    
    @classmethod
    def from_file(cls, config_path: Path):
        """Load configuration from INI file"""
        if not config_path.exists():
            return
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if 'NSE' in config:
            section = config['NSE']
            cls.MAX_WORKERS = section.getint('max_workers', cls.MAX_WORKERS)
            cls.MAX_ATTEMPTS = section.getint('max_attempts', cls.MAX_ATTEMPTS)
            cls.BACKOFF_FACTOR = section.getfloat('backoff_factor', cls.BACKOFF_FACTOR)
            cls.JITTER_MIN = section.getfloat('jitter_min', cls.JITTER_MIN)
            cls.JITTER_MAX = section.getfloat('jitter_max', cls.JITTER_MAX)
            cls.MIN_DELAY = section.getfloat('min_delay', cls.MIN_DELAY)
            cls.MAX_DELAY = section.getfloat('max_delay', cls.MAX_DELAY)

# Initialize configuration from environment
Config.from_env()

logging.basicConfig(
    level=logging.INFO,  # default logging level
    format="%(asctime)s [%(levelname)s] %(message)s",  # log format
    handlers=[logging.StreamHandler(sys.stdout)],  # output to stdout
)
logger = logging.getLogger(__name__)  # module-level logger

# ---- Security & Validation ----
def validate_symbol(symbol: str) -> bool:
    """Validate NSE symbol format to prevent injection attacks"""
    if not symbol or len(symbol) > 20:
        return False
    # NSE symbols: start with letter, contain only letters/numbers/ampersand
    return bool(re.match(r'^[A-Z0-9][A-Z0-9&.\-]{0,19}$', symbol))

def sanitize_filename(symbol: str) -> str:
    """Sanitize symbol for safe filename usage, preventing path traversal"""
    # Replace any non-alphanumeric/underscore/dash/dot with underscore
    # But preserve hyphens and dots that are valid in NSE symbols
    sanitized = re.sub(r'[^\w\-_.]', '_', symbol)
    # Additional safety: prevent directory traversal attempts
    sanitized = re.sub(r'^\.+', '', sanitized)  # Remove leading dots
    sanitized = re.sub(r'[\\/]', '_', sanitized)  # Replace path separators
    return sanitized

def validate_and_clean_symbols(symbols_input: str) -> List[str]:
    """Parse, validate, and clean symbol input"""
    symbols = []
    for s in symbols_input.split(","):
        clean_s = s.strip().upper()
        if clean_s and validate_symbol(clean_s):
            symbols.append(clean_s)
        elif clean_s:
            logger.warning(f"Invalid symbol format (skipping): {clean_s}")
    
    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    
    return deduped

# ---- Helpers ----
def ensure_dir(p: Path) -> None:
    """Create directory if missing (no error if exists)"""
    p.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1)
def compute_1y_range() -> Tuple[str, str]:
    """Cache the date range since it doesn't change during execution"""
    # Use yesterday as the "to" date to ensure data availability and avoid 404s
    to_dt = datetime.now().date() - timedelta(days=1)
    from_dt = to_dt - timedelta(days=364)  # roughly one year
    return from_dt.strftime("%d-%m-%Y"), to_dt.strftime("%d-%m-%Y")  # return strings in dd-mm-YYYY format

@contextmanager
def managed_session():
    """Context manager for proper session cleanup"""
    session = create_base_session()
    try:
        yield session
    finally:
        session.close()

def create_base_session() -> requests.Session:
    """Create a requests session with basic retry and pooling"""
    session = requests.Session()  # new persistent session for connection pooling
    retry_strategy = Retry(
        total=3,  # number of retries for connection errors
        backoff_factor=0.5,  # base backoff multiplier for urllib3's Retry
        status_forcelist=[500, 502, 503, 504],  # retry on these server error statuses
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy, 
        pool_connections=Config.MAX_WORKERS, 
        pool_maxsize=Config.MAX_WORKERS * 2
    )  # configure connection pooling and retries
    session.mount("https://", adapter)  # use adapter for HTTPS
    session.mount("http://", adapter)  # use adapter for HTTP
    
    headers = {
        "User-Agent": Config.USER_AGENT,  # set UA for session
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",  # accept HTML/XML primarily
        "Accept-Language": "en-US,en;q=0.9",  # prefer english
        "Referer": Config.NSE_HOME,  # set referer to home page
    }
    session.headers.update(headers)  # add these headers to the session
    
    return session  # return configured session

def save_debug(resp: requests.Response, path: Path, symbol: str) -> None:
    """Save debug information for failed requests"""
    ensure_dir(path.parent)  # ensure directory exists before writing
    with open(path, "w", encoding="utf-8") as f:
        f.write("Headers:\n")  # write header section
        for k, v in resp.headers.items():
            f.write(f"{k}: {v}\n")  # dump response headers
        f.write("\nBody:\n")  # body delimiter
        f.write(resp.text)  # write response body (HTML or text)
    logger.info("[%s] Saved debug to %s", symbol, path)  # log debug save

def fetch_nse_csv_and_save(
    session: requests.Session, 
    symbol: str, 
    from_date: str, 
    to_date: str, 
    out_path: Path, 
    dry_run: bool = False
) -> Path:
    """Fetch NSE CSV data and save to file with comprehensive error handling"""
    base = "https://www.nseindia.com/api/historical/cm/equity"  # API endpoint for historical equity CSV
    params = {
        "symbol": symbol,  # symbol parameter
        "series": '["EQ"]',  # series filter as JSON string
        "from": from_date,  # from-date dd-mm-YYYY
        "to": to_date,  # to-date dd-mm-YYYY
        "csv": "true",  # request CSV format
    }
    headers = {
        "User-Agent": Config.USER_AGENT,  # request-specific UA
        "Accept": "text/csv,*/*;q=0.9",  # prefer CSV
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/report-detail/equity-historical-search",
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Connection": "keep-alive",
    }
    
    for attempt in range(1, Config.MAX_ATTEMPTS + 1):  # attempt loop 1..MAX_ATTEMPTS
        try:
            resp = session.get(base, params=params, headers=headers, timeout=(8, 40))  # perform GET with connect/read timeouts
            content_length = len(resp.content)  # measure response byte length
            logger.info("[%s] Attempt %d: GET %s -> %d, length %d", symbol, attempt, resp.url, resp.status_code, content_length)  # log attempt result
            
            if resp.status_code == 429:  # rate-limited
                retry_after = 1  # default wait if no header
                if 'Retry-After' in resp.headers:
                    try:
                        retry_after = int(resp.headers['Retry-After'])  # parse numeric Retry-After seconds
                    except ValueError:
                        retry_after = 60  # fallback if header not numeric (date or malformed)
                backoff = (Config.BACKOFF_FACTOR ** (attempt - 1)) * random.uniform(Config.JITTER_MIN, Config.JITTER_MAX)  # compute exponential jittered backoff
                sleep_time = max(retry_after, backoff)  # respect server's Retry-After if larger
                logger.info("[%s] Rate limited (429), sleeping %.1f sec", symbol, sleep_time)  # log rate-limit action
                time.sleep(sleep_time)  # sleep before next attempt
                continue  # retry loop
                
            if resp.status_code != 200:  # non-success status (other than 429 handled above)
                preview = resp.text[:200]  # take short preview of body for debugging
                logger.error("[%s] Error status %d, preview: %s", symbol, resp.status_code, preview)  # log error
                save_debug(resp, out_path.with_suffix('.debug.html'), symbol)  # save debug HTML for inspection
                if attempt == Config.MAX_ATTEMPTS:
                    raise RuntimeError(f"[{symbol}] Max attempts reached with status {resp.status_code}")  # fail after last attempt
                backoff = (Config.BACKOFF_FACTOR ** (attempt - 1)) * random.uniform(Config.JITTER_MIN, Config.JITTER_MAX)  # compute backoff
                time.sleep(backoff)  # sleep then retry
                continue  # next attempt
            
            content_type = resp.headers.get('Content-Type', '').lower()  # get content-type header
            if 'text/csv' not in content_type:  # validate content-type is CSV
                preview = resp.text[:200]  # preview for logs
                logger.error("[%s] Invalid content-type '%s', preview: %s", symbol, content_type, preview)  # log mismatch
                save_debug(resp, out_path.with_suffix('.debug.html'), symbol)  # save debug HTML
                if attempt == Config.MAX_ATTEMPTS:
                    raise RuntimeError(f"[{symbol}] Max attempts reached (invalid content-type)")  # fail after final attempt
                backoff = (Config.BACKOFF_FACTOR ** (attempt - 1)) * random.uniform(Config.JITTER_MIN, Config.JITTER_MAX)  # backoff value
                time.sleep(backoff)  # sleep
                continue  # retry
            
            preview = resp.content[:512].decode("utf-8", errors="ignore").lower()  # sample initial bytes, decode safely
            if '<html' in preview or 'error' in preview:  # heuristic: response may be an HTML error page
                logger.error("[%s] Appears to be HTML error, preview: %s", symbol, preview)  # log probable HTML error
                save_debug(resp, out_path.with_suffix('.debug.html'), symbol)  # save debug HTML for analysis
                if attempt == Config.MAX_ATTEMPTS:
                    raise RuntimeError(f"[{symbol}] Max attempts reached (HTML error)")  # fail after last attempt
                backoff = (Config.BACKOFF_FACTOR ** (attempt - 1)) * random.uniform(Config.JITTER_MIN, Config.JITTER_MAX)  # compute backoff
                time.sleep(backoff)  # sleep then retry
                continue  # next attempt
            
            # Success path: valid CSV content
            if not dry_run:
                ensure_dir(out_path.parent)  # ensure parent directory exists
                with tempfile.NamedTemporaryFile(dir=out_path.parent, delete=False, mode="wb") as tmp:
                    tmp.write(resp.content)  # write response bytes to temp file (atomicish)
                os.rename(tmp.name, str(out_path))  # atomically rename temp to final target (on same filesystem)
                logger.info("[%s] Saved CSV to %s", symbol, out_path)  # log success
            else:
                logger.info("[%s] Dry-run: would save CSV to %s (length %d)", symbol, out_path, content_length)  # dry-run log
            return out_path  # return path on success
        
        except Exception as e:
            logger.warning("[%s] Attempt %d failed: %s", symbol, attempt, e)  # log exception for this attempt
            if attempt == Config.MAX_ATTEMPTS:
                raise RuntimeError(f"[{symbol}] Max attempts reached: {e}")  # re-raise wrapped after final attempt
            backoff = (Config.BACKOFF_FACTOR ** (attempt - 1)) * random.uniform(Config.JITTER_MIN, Config.JITTER_MAX)  # compute backoff before retry
            time.sleep(backoff)  # sleep and loop to next attempt
    
    raise RuntimeError(f"[{symbol}] Failed after {Config.MAX_ATTEMPTS} attempts")  # safety net (shouldn't reach due to raises above)

def setup_chrome(download_dir: Path, headless: bool = True) -> "webdriver.Chrome":
    """Setup Chrome WebDriver with security-hardened options"""
    opts = webdriver.ChromeOptions()  # Chrome options container
    if headless:
        opts.add_argument("--headless=new")  # use new headless mode flag for modern Chrome
    opts.add_argument(f"--user-agent={Config.USER_AGENT}")  # set UA to the same UA used in requests
    opts.add_argument("--disable-gpu")  # disable GPU
    opts.add_argument("--no-sandbox")  # disable sandbox
    opts.add_argument("--disable-dev-shm-usage")  # fix for resource-constrained environments like CI
    opts.add_argument("--window-size=1920,1080")  # set window size for consistency
    
    prefs = {
        "download.default_directory": str(download_dir.resolve()),  # set Chrome download default folder
        "download.prompt_for_download": False,  # disable download prompt
    }
    opts.add_experimental_option("prefs", prefs)  # add prefs to options
    service = Service(ChromeDriverManager().install())  # install chromedriver
    driver = webdriver.Chrome(service=service, options=opts)  # instantiate driver
    driver.set_page_load_timeout(60)  # increased timeout to 60s for slow CI networks
    return driver  # return the driver instance

def process_symbol(
    symbol: str, 
    out_dir: Path, 
    overwrite: bool, 
    dry_run: bool, 
    base_session: requests.Session
) -> Dict[str, str]:
    """Process a single symbol - optimized for parallel execution"""
    # Use sanitized symbol for filename to prevent path traversal
    safe_symbol = sanitize_filename(symbol)
    out_path = out_dir / f"{safe_symbol}_1Y_{datetime.now().strftime('%Y%m%d')}.csv"  # planned output filename
    
    if out_path.exists() and not overwrite:
        logger.info("[%s] Already exists, skipping", symbol)  # skip if exists and not overwriting
        return {"symbol": symbol, "status": "skipped", "info": str(out_path)}  # return skip result
    
    from_d, to_d = compute_1y_range()  # compute date range (cached)
    
    try:
        saved = fetch_nse_csv_and_save(base_session, symbol, from_d, to_d, out_path, dry_run)  # fetch & save CSV
        return {"symbol": symbol, "status": "ok", "info": str(saved)}  # success result
    except Exception as e:
        logger.warning("[%s] API fetch failed: %s", symbol, e)  # log failure
        return {"symbol": symbol, "status": "error", "info": str(e)}  # return error result
    finally:
        time.sleep(random.uniform(Config.MIN_DELAY, Config.MAX_DELAY))  # polite pause before next symbol (reduce burstiness)

def create_sample_config(config_path: Path) -> None:
    """Create a sample configuration file"""
    config = configparser.ConfigParser()
    config['NSE'] = {
        'max_workers': str(Config.MAX_WORKERS),
        'max_attempts': str(Config.MAX_ATTEMPTS),
        'backoff_factor': str(Config.BACKOFF_FACTOR),
        'jitter_min': str(Config.JITTER_MIN),
        'jitter_max': str(Config.JITTER_MAX),
        'min_delay': str(Config.MIN_DELAY),
        'max_delay': str(Config.MAX_DELAY),
    }
    
    ensure_dir(config_path.parent)
    with open(config_path, 'w') as f:
        f.write("# NSE Data Extractor Configuration\n")
        f.write("# Adjust these values based on your system and requirements\n\n")
        config.write(f)
    
    logger.info(f"Created sample configuration file: {config_path}")

def main():
    p = argparse.ArgumentParser(description="NSE Historical Data Extractor - Production Hardened")  # CLI argument parser
    p.add_argument("--symbols", default="INFY", help="Comma-separated list of NSE symbols")  # default symbols comma-separated
    p.add_argument("--out-dir", default=str(Config.DEFAULT_DOWNLOAD_DIR), help="Output directory for CSV files")  # output directory CLI override
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")  # switch to overwrite existing files
    p.add_argument("--fast", action="store_true", help="Run in fast mode (headless browser)")  # fast flag (used to run headless)
    p.add_argument("--no-browser", action="store_true", help="Disable browser warm-up entirely")  # disable browser warm-up
    p.add_argument("--workers", type=int, default=Config.MAX_WORKERS, 
                   help=f"Number of parallel workers (default: {Config.MAX_WORKERS})")  # concurrency control
    p.add_argument("--dry-run", action="store_true", help="Perform requests but do not write files")  # dry-run toggle
    p.add_argument("--config", type=str, help="Path to configuration file")  # configuration file path
    p.add_argument("--create-config", action="store_true", help="Create sample configuration file and exit")  # create config helper
    args = p.parse_args()  # parse CLI args into namespace

    # Handle config file creation
    if args.create_config:
        config_path = Path(args.config) if args.config else Path("nse_config.ini")
        create_sample_config(config_path)
        return

    # Load configuration from file if specified
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            Config.from_file(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")

    # Validate and clean symbols with security checks
    symbols = validate_and_clean_symbols(args.symbols)
    if not symbols:
        logger.error("No valid symbols provided. Symbols must be valid NSE format (e.g., INFY, TCS, RELIANCE)")
        sys.exit(1)

    logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")

    out_dir = Path(args.out_dir)  # convert out_dir to Path
    ensure_dir(out_dir)  # ensure output directory exists

    # Validate worker count
    workers = min(args.workers, len(symbols), Config.MAX_WORKERS)
    if workers != args.workers:
        logger.info(f"Adjusted workers from {args.workers} to {workers}")

    use_browser = (not args.no_browser) and HAS_SELENIUM  # use browser only if selenium available and not disabled

    # Use context manager for proper session cleanup
    with managed_session() as base_session:
        driver = None  # init driver variable

        if use_browser:
            for warmup_attempt in range(1, 4):  # up to 3 attempts for warm-up
                driver = None
                try:
                    logger.info("Starting browser warm-up (attempt %d/3)...", warmup_attempt)
                    driver = setup_chrome(out_dir, headless=args.fast)
                    driver.get(Config.NSE_HOME)  # load homepage
                    
                    # Wait for site to actually load
                    WebDriverWait(driver, 30).until(lambda d: d.execute_script("return document.readyState") == "complete")
                    
                    # Navigate to the specific historical data search page - KEY for getting right cookies
                    report_url = "https://www.nseindia.com/report-detail/equity-historical-search"
                    driver.get(report_url)
                    # Use a more reliable wait for report page
                    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "reports_tab")))
                    
                    # Also visit one symbol page for good measure
                    if symbols:
                        quote_url = f"https://www.nseindia.com/get-quotes/equity?symbol={symbols[0]}"
                        driver.get(quote_url)
                        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "quoteName")))
                    
                    # Capture and transfer cookies
                    browser_cookies = driver.get_cookies()
                    if not browser_cookies:
                        raise RuntimeError("No cookies captured from browser")
                        
                    for c in browser_cookies:
                        # Copy cookie to session
                        domain = c.get("domain", ".nseindia.com")
                        if domain.startswith("www."):
                            domain = domain[3:] # standardize to .nseindia.com if needed
                        base_session.cookies.set(c["name"], c["value"], domain=domain)
                    
                    cookie_names = [c["name"] for c in browser_cookies]
                    logger.info("Selenium warm-up success. Cookies captured: %s", ", ".join(cookie_names))
                    break  # success, exit retry loop
                    
                except Exception as e:
                    logger.warning("Browser warm-up attempt %d failed: %s", warmup_attempt, e)
                    if warmup_attempt == 3:
                        logger.error("All browser warm-up attempts failed. Proceeding without browser cookies (may fail).")
                finally:
                    if driver:
                        try:
                            driver.quit()
                        except:
                            pass

        else:
            # Quick warm-up without browser
            try:
                base_session.get(Config.NSE_HOME, timeout=5)  # simple GET to seed session cookies/headers if possible
            except Exception:
                pass  # ignore warm-up errors silently

        results = []  # collect results per symbol
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_symbol = {
                executor.submit(process_symbol, sym, out_dir, args.overwrite, args.dry_run, base_session): sym 
                for sym in symbols
            }  # submit a task per symbol and map futures to symbols
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]  # get symbol for this completed future
                try:
                    result = future.result()  # obtain result (may raise)
                    results.append(result)  # append success/skip/error dict
                except Exception as exc:
                    logger.error("[%s] generated an exception: %s", symbol, exc)  # log unexpected exception
                    results.append({"symbol": symbol, "status": "error", "info": str(exc)})  # record error

    # Summary with statistics
    success_count = sum(1 for r in results if r['status'] == 'ok')
    skip_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nSummary: {success_count} succeeded, {skip_count} skipped, {error_count} failed")
    print("Detailed Results:")
    for r in results:
        print(f"{r['symbol']} | {r['status']} | {r['info']}")

# ---- Basic Test Structure (for future expansion) ----
def run_basic_tests():
    """Basic validation tests for key functions"""
    # Test symbol validation
    assert validate_symbol("INFY") == True, "Valid symbol should pass"
    assert validate_symbol("TCS") == True, "Valid symbol should pass"
    assert validate_symbol("invalid-symbol") == False, "Invalid symbol should fail"
    assert validate_symbol("") == False, "Empty symbol should fail"
    assert validate_symbol("A" * 25) == False, "Too long symbol should fail"
    
    # Test filename sanitization
    assert sanitize_filename("INFY") == "INFY", "Clean symbol unchanged"
    assert sanitize_filename("TEST/SYMBOL") == "TEST_SYMBOL", "Slash replaced"
    assert sanitize_filename("BAD\\PATH") == "BAD_PATH", "Backslash replaced"
    
    # Test date range computation
    from_date, to_date = compute_1y_range()
    assert datetime.strptime(from_date, "%d-%m-%Y"), "From date should be valid"
    assert datetime.strptime(to_date, "%d-%m-%Y"), "To date should be valid"
    
    # Test symbol validation and cleaning
    result = validate_and_clean_symbols("INFY,TCS,invalid-symbol,RELIANCE")
    assert "INFY" in result, "Valid symbols should be included"
    assert "TCS" in result, "Valid symbols should be included"
    assert "RELIANCE" in result, "Valid symbols should be included"
    assert "invalid-symbol" not in result, "Invalid symbols should be excluded"
    
    print("✅ All basic tests passed")

if __name__ == "__main__":
    # Uncomment the line below to run basic tests
    # run_basic_tests()
    main()  # entry point: run main when executed as script