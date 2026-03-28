"""
test_observability.py
Unit tests for the observability utility.
"""

import unittest
import json
import logging
from io import StringIO
from src.utils.observability import get_tenant_logger

class TestObservability(unittest.TestCase):
    def test_logger_format(self):
        """Verify that the logger outputs structured JSON."""
        logger = get_tenant_logger("test-logger", tenant_id="TEST-TENANT")
        
        # Capture output
        log_capture = StringIO()
        ch = logging.StreamHandler(log_capture)
        # Use the same formatter as the utility
        ch.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(ch)
        
        logger.info("Test message")
        
        output = log_capture.getvalue().strip()
        log_json = json.loads(output)
        
        self.assertEqual(log_json["message"], "Test message")
        self.assertEqual(log_json["level"], "INFO")
        self.assertEqual(log_json["tenant_id"], "TEST-TENANT")
        self.assertIn("timestamp", log_json)

if __name__ == "__main__":
    unittest.main()
