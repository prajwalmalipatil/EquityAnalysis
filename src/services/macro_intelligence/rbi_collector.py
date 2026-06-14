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
            # TODO: Implement yearly HTML archive scraping fallback
        elif gap_days > 30:
            logger.warning("MONTHLY_ARCHIVE_SCAN_REQUIRED", extra={"gap_days": gap_days})
            # TODO: Implement monthly HTML archive scraping fallback
        elif gap_days > 7:
            logger.warning("WEEKLY_ARCHIVE_SCAN_REQUIRED", extra={"gap_days": gap_days})
            
        session = self._get_retry_session()

        for category, url in self.FEEDS.items():
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
            except Exception as e:
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
                            "pub_dt": pub_dt
                        }
                        events.append(self.normalize(raw_item))
                except Exception as e:
                    logger.error("FAILED_TO_PARSE_ENTRY", extra={"entry_title": entry.get("title", ""), "error": str(e)})
                    continue
        if not events:
            logger.info("NO_NEW_RBI_EVENTS_FOUND_VIA_RSS")
        else:
            logger.info("RBI_FETCH_COMPLETE", extra={"new_events_found": len(events)})
            
        return events

    def normalize(self, raw_item: dict) -> MacroEvent:
        """Normalizes the RSS entry into a strictly formatted MacroEvent."""
        entry = raw_item["entry"]
        category = raw_item["category"]
        pub_dt = raw_item["pub_dt"]
        pub_date_iso = pub_dt.isoformat()
        official_url = entry.link
        
        # Canonical event_id: sha256(pub_date + category + official_url)
        raw_key = f"{pub_date_iso}|{category}|{official_url}"
        event_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        
        # Clean title and summary (remove CDATA if present)
        title = entry.title.replace("<![CDATA[", "").replace("]]>", "").strip()
        summary = entry.get("summary", title).replace("<![CDATA[", "").replace("]]>", "").strip()
        
        return MacroEvent(
            event_id=event_id,
            official_data=OfficialData(
                title=title,
                publication_date=pub_date_iso,
                category=category,
                source="RBI",
                official_url=official_url,
                content=summary
            ),
            derived_data=DerivedData(),
            metadata=EventMetadata(
                processing_state="NEW",
                lifecycle_status="ACTIVE",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat()
            )
        )
