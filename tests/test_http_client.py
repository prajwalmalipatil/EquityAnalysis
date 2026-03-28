"""
test_http_client.py
Unit tests for the http_client.py decorators.
"""

import unittest
from src.utils.http_client import with_retry, with_fallback

class TestHTTPClient(unittest.TestCase):
    def test_with_retry_success(self):
        """Verify successful function call with retry decorator."""
        @with_retry(max_attempts=3)
        def my_func():
            return "SUCCESS"
        
        self.assertEqual(my_func(), "SUCCESS")
    
    def test_with_retry_failure(self):
        """Verify the retry decorator raises an error after max attempts."""
        self.attempts = 0
        @with_retry(max_attempts=3, base_delay=0.1)
        def my_failing_func():
            self.attempts += 1
            raise ValueError("FAIL")
        
        with self.assertRaises(ValueError):
            my_failing_func()
        
        self.assertEqual(self.attempts, 3)

    def test_with_fallback_trigger(self):
        """Verify that fallback value is returned on error."""
        @with_fallback(fallback_value="FALLBACK")
        def my_failing_func():
            raise ValueError("FAIL")
        
        self.assertEqual(my_failing_func(), "FALLBACK")

if __name__ == "__main__":
    unittest.main()
