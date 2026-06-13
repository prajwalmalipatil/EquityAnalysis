import unittest
import tempfile
import os
from pathlib import Path
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.event_repository import EventRepository, make_dedup_key

class TestEventRepository(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filepath = Path(self.temp_dir.name) / "events.jsonl"
        self.repo = EventRepository(self.filepath)

        self.event1 = MacroEvent(
            event_id="rbi_1",
            url="http://rbi.org/1",
            published_at="2026-06-10T10:00:00Z",
            title="Event 1",
            summary="Sum 1",
            category="Test",
            source="RBI",
            collected_at="2026-06-13T00:00:00Z"
        )
        self.event2 = MacroEvent(
            event_id="rbi_2",
            url="http://rbi.org/2",
            published_at="2026-06-11T10:00:00Z",
            title="Event 2",
            summary="Sum 2",
            category="Test",
            source="RBI",
            collected_at="2026-06-13T00:00:00Z"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_event(self):
        saved = self.repo.save_event(self.event1)
        self.assertTrue(saved)
        
        events = self.repo.get_all_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, "rbi_1")

    def test_deduplication(self):
        # Save event once
        self.assertTrue(self.repo.save_event(self.event1))
        
        # Save exact same event again
        self.assertFalse(self.repo.save_event(self.event1))
        
        # Should only be one event in repo
        events = self.repo.get_all_events()
        self.assertEqual(len(events), 1)

    def test_query_by_date_range(self):
        self.repo.save_event(self.event1) # 2026-06-10
        self.repo.save_event(self.event2) # 2026-06-11
        
        # Query exactly on 10th
        results = self.repo.get_events_by_date_range("2026-06-10T00:00:00Z", "2026-06-10T23:59:59Z")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].event_id, "rbi_1")
        
        # Query range covering both
        results2 = self.repo.get_events_by_date_range("2026-06-09", "2026-06-12")
        self.assertEqual(len(results2), 2)

    def test_cache_reloading(self):
        self.repo.save_event(self.event1)
        
        # Instantiate a new repo pointing to the same file
        repo2 = EventRepository(self.filepath)
        
        # Try saving the same event, it should be caught by cache loading
        self.assertFalse(repo2.save_event(self.event1))

if __name__ == '__main__':
    unittest.main()
