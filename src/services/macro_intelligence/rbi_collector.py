import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import List
from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData, EventMetadata
from src.services.macro_intelligence.interfaces import OfficialSourceCollector
from src.utils.observability import get_tenant_logger
import hashlib
import re
import html

logger = get_tenant_logger("rbi-collector")

class RBICollector(OfficialSourceCollector):
    """
    Collects macroeconomic events specifically from the Reserve Bank of India (RBI).
    Parses RBI's unstable RSS feeds and guarantees robust extraction.
    """
    
    FEEDS = {
        "Press Releases": "https://rbi.org.in/pressreleases_rss.xml",
        "Notifications": "https://rbi.org.in/notifications_rss.xml",
        "Publications": "https://rbi.org.in/Publication_rss.xml",
    }
    
    @property
    def provider_name(self) -> str:
        return "RBI"

    def _get_retry_session(self) -> requests.Session:
        """Configures a requests session with exponential backoff."""
        session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=2,
            status_forcelist=(500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_since(self, last_ts: str) -> List[MacroEvent]:
        """
        Continuity Engine: Fetches all events published after last_ts.
        If last_ts is empty, fetches the last 30 days of data.
        """
        events = []
        
        if not last_ts:
            last_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
        else:
            try:
                # ISO 8601 parsing fallback
                last_dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
            except ValueError:
                last_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        gap_days = (now - last_dt).days

        logger.info("FETCHING_RBI_EVENTS", extra={"since": last_dt.isoformat(), "gap_days": gap_days})

        if gap_days > 365:
            logger.warning("YEARLY_ARCHIVE_SCAN_REQUIRED", extra={"gap_days": gap_days})
        elif gap_days > 30:
            logger.warning("MONTHLY_ARCHIVE_SCAN_REQUIRED", extra={"gap_days": gap_days})
        elif gap_days > 7:
            logger.warning("WEEKLY_ARCHIVE_SCAN_REQUIRED", extra={"gap_days": gap_days})
            
        session = self._get_retry_session()
        feed_success = False

        for category, url in self.FEEDS.items():
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                feed_success = True
            except requests.RequestException as e:
                logger.error("NETWORK_FAILURE", extra={"url": url, "error": str(e)})
                continue
                
            for entry in feed.entries:
                try:
                    pub_dt = parsedate_to_datetime(entry.published)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        
                    if pub_dt > last_dt:
                        raw_item = {
                            "entry": entry,
                            "category": category,
                            "pub_dt": pub_dt,
                            "session": session
                        }
                        events.append(self.normalize(raw_item))
                except (AttributeError, ValueError, KeyError, TypeError) as e:
                    logger.error("FAILED_TO_PARSE_ENTRY", extra={"entry_title": entry.get("title", ""), "error": str(e)})
                    continue

        # HTML archive fallback if feed failed or returned 0 items
        if not feed_success or len(events) == 0:
            logger.info("TRIGGERING_HTML_ARCHIVE_FALLBACK")
            fallback_events = self._fetch_from_html_archives(session, last_dt)
            events.extend(fallback_events)

        if not events:
            logger.info("NO_NEW_RBI_EVENTS_FOUND")
        else:
            logger.info("RBI_FETCH_COMPLETE", extra={"new_events_found": len(events)})
            
        return events

    def _extract_pdf_url(self, session: requests.Session, official_url: str) -> Optional[str]:
        """Fetches the official page HTML and extracts the main document PDF link."""
        try:
            response = session.get(official_url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            
            # Find all absolute/relative PDF links
            all_pdfs = re.findall(r'href=[\'"]?([^\'" >]+\.pdf)[\'"]?', html_content, re.IGNORECASE)
            
            candidate_pdfs = []
            for path in all_pdfs:
                path_lower = path.lower()
                # Skip layout/accessibility links
                if "accessibility" in path_lower or "utkarsh" in path_lower:
                    continue
                # Ensure absolute URL
                if path.startswith("/"):
                    resolved = f"https://www.rbi.org.in{path}"
                elif not path.startswith("http"):
                    resolved = f"https://www.rbi.org.in/scripts/{path}"
                else:
                    resolved = path
                candidate_pdfs.append(resolved)
                
            # Prefer links that contain /PDFs/ or /pdfs/
            for resolved in candidate_pdfs:
                if "/pdfs/" in resolved.lower():
                    return resolved
                    
            if candidate_pdfs:
                return candidate_pdfs[0]
                
        except Exception as e:
            logger.warning("FAILED_TO_EXTRACT_PDF_URL", extra={"url": official_url, "error": str(e)})
            
        return None

    def _fetch_from_html_archives(self, session: requests.Session, last_dt: datetime) -> List[MacroEvent]:
        """Scrapes RBI html archives directly as a fallback option."""
        events = []
        archives = [
            ("https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx", "Press Releases", r'href=[\'"]?BS_PressReleaseDisplay\.aspx\?prid=(\d+)[\'"]?'),
            ("https://www.rbi.org.in/scripts/NotificationUser.aspx", "Notifications", r'href=[\'"]?NotificationUser\.aspx\?Id=(\d+)(?:&amp;|&)Mode=0[\'"]?')
        ]
        
        for url, category, link_regex in archives:
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                html_content = response.text
                
                # Split by tableheader class date headers
                date_blocks = re.split(r'<td[^>]+class=["\']?tableheader["\']?[^>]*>\s*<b>\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4})\s*</?b>', html_content, flags=re.IGNORECASE)
                
                if len(date_blocks) > 1:
                    for i in range(1, len(date_blocks), 2):
                        date_str = date_blocks[i]
                        block_content = date_blocks[i+1]
                        
                        try:
                            dt = datetime.strptime(date_str, "%b %d, %Y")
                            pub_dt = dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
                            
                        if pub_dt <= last_dt:
                            continue
                            
                        # Find all matching rows
                        rows = re.findall(link_regex + r'[^>]*>(.*?)</a>', block_content, re.IGNORECASE | re.DOTALL)
                        
                        for prid, title_raw in rows:
                            title = re.sub(r'<[^>]+>', ' ', title_raw).strip()
                            if category == "Press Releases":
                                official_url = f"https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid={prid}"
                            else:
                                official_url = f"https://www.rbi.org.in/scripts/NotificationUser.aspx?Id={prid}&Mode=0"
                                
                            # Search for PDF link
                            pr_pos = block_content.find(prid)
                            pdf_url = None
                            if pr_pos != -1:
                                snippet = block_content[pr_pos:pr_pos+400]
                                pdf_match = re.search(r'href=[\'"]?(https://rbidocs\.rbi\.org\.in/[^\'">\s]+\.pdf)[\'"]?', snippet, re.IGNORECASE)
                                if pdf_match:
                                    pdf_url = pdf_match.group(1)
                                    
                            class MockEntry:
                                def __init__(self, t, l, s):
                                    self.title = t
                                    self.link = l
                                    self.summary = s
                                def get(self, key, default=None):
                                    return getattr(self, key, default)
                                    
                            entry = MockEntry(title, official_url, title)
                            
                            raw_item = {
                                "entry": entry,
                                "category": category,
                                "pub_dt": pub_dt
                            }
                            event = self.normalize(raw_item)
                            if pdf_url:
                                event.official_data.pdf_url = pdf_url
                            events.append(event)
            except Exception as e:
                logger.error("HTML_ARCHIVE_FALLBACK_FAILED", extra={"url": url, "error": str(e)})
                
        return events

    def normalize(self, raw_item: dict) -> MacroEvent:
        """Normalizes the RSS entry into a strictly formatted MacroEvent."""
        entry = raw_item["entry"]
        category = raw_item["category"]
        pub_dt = raw_item["pub_dt"]
        pub_date_iso = pub_dt.isoformat()
        official_url = entry.link
        session = raw_item.get("session")
        
        # Canonical event_id: sha256(pub_date + category + official_url)
        raw_key = f"{pub_date_iso}|{category}|{official_url}"
        event_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        
        # Clean title and summary (remove CDATA if present)
        title = entry.title.replace("<![CDATA[", "").replace("]]>", "").strip()
        raw_summary = entry.get("summary", title).replace("<![CDATA[", "").replace("]]>", "")
        clean_summary = html.unescape(re.sub(r'<[^>]+>', ' ', raw_summary)).strip()
        summary = re.sub(r'\s+', ' ', clean_summary)
        
        # Fetch PDF URL if session is available
        pdf_url = None
        if session:
            pdf_url = self._extract_pdf_url(session, official_url)
        
        return MacroEvent(
            event_id=event_id,
            official_data=OfficialData(
                title=title,
                publication_date=pub_date_iso,
                category=category,
                source="RBI",
                official_url=official_url,
                content=summary,
                pdf_url=pdf_url
            ),
            derived_data=DerivedData(),
            metadata=EventMetadata(
                processing_state="NEW",
                lifecycle_status="ACTIVE",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat()
            )
        )
