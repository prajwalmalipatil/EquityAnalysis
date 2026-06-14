from src.services.macro_intelligence.read_models import DashboardBundle

class ReleaseValidator:
    """Validates the complete DashboardBundle before it is allowed to be published."""
    
    @staticmethod
    def validate(bundle: DashboardBundle) -> None:
        event_ids = set()
        title_dates = set()
        urls = set()
        
        for e in bundle.events:
            # 1. No mock IDs
            if "mock-" in e.event_id.lower():
                raise ValueError(f"Release Blocker: Mock ID found: {e.event_id}")
                
            # 2. Unique event_ids
            if e.event_id in event_ids:
                raise ValueError(f"Release Blocker: Duplicate event_id found: {e.event_id}")
            event_ids.add(e.event_id)
            
            # 3. No duplicate titles with identical dates
            title_date_key = f"{e.title}_{e.published[:10]}"
            if title_date_key in title_dates:
                raise ValueError(f"Release Blocker: Duplicate title+date found: {title_date_key}")
            title_dates.add(title_date_key)
            
            # 4. No duplicate URLs (if valid URL provided)
            url = e.url
            if url and url != "Unknown" and not url.startswith("http://mock"):
                if url in urls:
                    raise ValueError(f"Release Blocker: Duplicate URL found: {url}")
                urls.add(url)
                
            # 5. Required Fields
            if not e.title:
                raise ValueError(f"Release Blocker: Missing title for event: {e.event_id}")
                
            # 6. AI Validations
            conf = e.confidence
            if not (0 <= conf <= 100):
                raise ValueError(f"Release Blocker: Confidence out of bounds (0-100) for {e.event_id}: {conf}")
                
        # Validate Analytics
        analytics_total = bundle.analytics.analytics.get('total_events', 0)
        if analytics_total != len(bundle.events):
            raise ValueError(f"Release Blocker: Analytics total_events ({analytics_total}) "
                             f"does not match event count ({len(bundle.events)})")
                             
        # Validate Manifest
        if bundle.manifest.event_count != len(bundle.events):
            raise ValueError(f"Release Blocker: Manifest event_count ({bundle.manifest.event_count}) "
                             f"does not match event count ({len(bundle.events)})")
