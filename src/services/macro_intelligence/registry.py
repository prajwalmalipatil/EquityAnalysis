from typing import List, Dict

class ProviderRegistry:
    """
    Central registry for all Macro Intelligence data providers.
    Supports dynamic discovery and health monitoring of providers like RBI, SEBI, NSE.
    """
    def __init__(self):
        # We store providers by their provider_name property.
        self._providers: Dict[str, any] = {}
        
    def register(self, provider: any):
        """Register an instantiated provider."""
        self._providers[provider.provider_name] = provider
        
    def discover(self) -> List[any]:
        """Returns all registered providers."""
        return list(self._providers.values())
        
    def health_check(self) -> dict:
        """Returns the health status of all registered providers."""
        health = {}
        for name, provider in self._providers.items():
            try:
                if hasattr(provider, "summarize"):
                    result = provider.summarize("Health check ping")
                    health[name] = "OK" if result is not None else "DEGRADED"
                elif hasattr(provider, "fetch_since"):
                    # Validate connectivity by running a short fetch check
                    from datetime import datetime, timezone
                    now_str = datetime.now(timezone.utc).isoformat()
                    provider.fetch_since(now_str)
                    health[name] = "OK"
                else:
                    health[name] = "DEGRADED"
            except Exception as e:
                health[name] = f"ERROR: {str(e)}"
        return health
