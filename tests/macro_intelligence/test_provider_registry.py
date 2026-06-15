import unittest
from unittest.mock import MagicMock
from src.services.macro_intelligence.registry import ProviderRegistry

class TestProviderRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ProviderRegistry()

    def test_register_and_discover(self):
        provider = MagicMock()
        provider.provider_name = "MockProvider"
        
        self.registry.register(provider)
        discovered = self.registry.discover()
        
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].provider_name, "MockProvider")

    def test_health_check_ai_provider_success(self):
        provider = MagicMock()
        provider.provider_name = "MockAI"
        provider.summarize.return_value = {"summary": "OK"}
        del provider.fetch_since  # Ensure it doesn't have fetch_since
        
        self.registry.register(provider)
        health = self.registry.health_check()
        
        self.assertEqual(health["MockAI"], "OK")
        provider.summarize.assert_called_once_with("Health check ping")

    def test_health_check_ai_provider_degraded(self):
        provider = MagicMock()
        provider.provider_name = "MockAI"
        provider.summarize.return_value = None
        del provider.fetch_since
        
        self.registry.register(provider)
        health = self.registry.health_check()
        
        self.assertEqual(health["MockAI"], "DEGRADED")

    def test_health_check_ai_provider_error(self):
        provider = MagicMock()
        provider.provider_name = "MockAI"
        provider.summarize.side_effect = Exception("API Key Expired")
        del provider.fetch_since
        
        self.registry.register(provider)
        health = self.registry.health_check()
        
        self.assertEqual(health["MockAI"], "ERROR: API Key Expired")

    def test_health_check_collector_success(self):
        provider = MagicMock()
        provider.provider_name = "MockCollector"
        del provider.summarize  # Ensure it doesn't have summarize
        
        self.registry.register(provider)
        health = self.registry.health_check()
        
        self.assertEqual(health["MockCollector"], "OK")
        provider.fetch_since.assert_called_once()

    def test_health_check_collector_error(self):
        provider = MagicMock()
        provider.provider_name = "MockCollector"
        provider.fetch_since.side_effect = Exception("Connection Timeout")
        del provider.summarize
        
        self.registry.register(provider)
        health = self.registry.health_check()
        
        self.assertEqual(health["MockCollector"], "ERROR: Connection Timeout")

if __name__ == "__main__":
    unittest.main()
