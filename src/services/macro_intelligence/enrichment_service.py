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
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

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
        
    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return "gemini-1.5-flash"
        
    def summarize(self, text: str) -> Optional[dict]:
        prompt = f"""
        Summarize the following macroeconomic event for quantitative trading.
        Extract the core impact, and list 3-5 key bullet points.
        Output MUST be valid JSON with keys "summary" (string) and "key_points" (list of strings).
        Do not include markdown blocks like ```json. Just raw JSON.
        
        ---BEGIN EVENT TEXT---
        {text}
        ---END EVENT TEXT---
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
    @property
    def provider_name(self) -> str:
        return "dummy"

    @property
    def model_name(self) -> str:
        return "dummy"

    def summarize(self, text: str) -> Optional[dict]:
        logger.warning("USING_DUMMY_AI_PROVIDER")
        return {
            "summary": "AI summary not available. Please configure an API key.",
            "key_points": ["No key points available."],
            "confidence": 0
        }

class EnrichmentService(EnrichmentServiceInterface):
    """
    Layer 4: AI Enrichment.
    Uses an injected AIProvider to generate semantic summaries of raw events.
    """
    
    def __init__(self, provider: AIProvider):
        self.provider = provider
        self.prompt_version = "v1.0.0"
        
    @staticmethod
    def _sanitize_prompt_input(text: str) -> str:
        """Strips HTML tags, limits length, and adds boundary delimiters."""
        if not text:
            return ""
        import re
        # Strip HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        # Remove potential prompt injection delimiters
        clean = clean.replace('"""', '').replace("'''", '')
        # Truncate to prevent token overflow
        return clean[:8000]

    def process(self, event: MacroEvent) -> MacroEvent:
        if not event.official_data.content:
            return event
            
        sanitized_title = self._sanitize_prompt_input(event.official_data.title or "")
        sanitized_content = self._sanitize_prompt_input(event.official_data.content or "")
        result = self.provider.summarize(sanitized_title + "\n\n" + sanitized_content)
        if result:
            new_version = 1
            if event.derived_data.ai_snapshots:
                new_version = event.derived_data.ai_snapshots[-1].version + 1
                
            snapshot = AISummarySnapshot(
                version=new_version,
                provider=self.provider.provider_name,
                model=self.provider.model_name,
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
