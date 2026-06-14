from abc import ABC, abstractmethod
from typing import Optional, List
import google.generativeai as genai
import os
import json
from datetime import datetime, timezone

from src.services.macro_intelligence.models import MacroEvent, AISummarySnapshot
from src.services.macro_intelligence.interfaces import EnrichmentServiceInterface
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("enrichment-service")

class AIProvider(ABC):
    @abstractmethod
    def summarize(self, text: str) -> Optional[dict]:
        """Returns a dict with 'summary' and 'key_points'."""
        pass

class GeminiProvider(AIProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiProvider")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def summarize(self, text: str) -> Optional[dict]:
        prompt = f"""
        Summarize the following macroeconomic event for quantitative trading.
        Extract the core impact, and list 3-5 key bullet points.
        Output MUST be valid JSON with keys "summary" (string) and "key_points" (list of strings).
        Do not include markdown blocks like ```json. Just raw JSON.
        
        Event text:
        {text}
        """
        try:
            resp = self.model.generate_content(prompt)
            # Clean backticks if any
            clean_text = resp.text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error("GEMINI_API_ERROR", extra={"error": str(e)})
            return None

class DummyProvider(AIProvider):
    """Fallback provider when no API key is available."""
    def summarize(self, text: str) -> Optional[dict]:
        logger.warning("USING_DUMMY_AI_PROVIDER")
        return {
            "summary": "AI summary not available. Please configure an API key.",
            "key_points": ["No key points available."]
        }

class EnrichmentService(EnrichmentServiceInterface):
    """
    Layer 4: AI Enrichment.
    Uses an injected AIProvider to generate semantic summaries of raw events.
    """
    
    def __init__(self, provider: AIProvider):
        self.provider = provider
        self.prompt_version = "v1.0.0"
        
    def process(self, event: MacroEvent) -> MacroEvent:
        if not event.official_data.content:
            return event
            
        result = self.provider.summarize(event.official_data.title + "\n\n" + event.official_data.content)
        if result:
            new_version = 1
            if event.derived_data.ai_snapshots:
                new_version = event.derived_data.ai_snapshots[-1].version + 1
                
            snapshot = AISummarySnapshot(
                version=new_version,
                provider="gemini",
                model="gemini-1.5-flash",
                generated_time=datetime.now(timezone.utc).isoformat(),
                confidence=result.get("confidence", 90),
                prompt_version=self.prompt_version,
                response_version="v1",
                summary=result.get("summary", "Summary unavailable"),
                key_points=result.get("key_points", []),
                raw_ai_response=json.dumps(result)
            )
            event.derived_data.ai_snapshots.append(snapshot)
            event.derived_data.ai_summary = snapshot.summary
            event.metadata.lifecycle_status = "ENRICHED"
            event.metadata.processing_state = "ENRICHED"
            
        return event
