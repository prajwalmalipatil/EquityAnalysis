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
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return "gemini-2.5-flash"
        
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
        import time
        import re
        max_retries = 5
        base_delay = 15
        
        for attempt in range(max_retries):
            try:
                resp = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                clean_text = resp.text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
                return json.loads(clean_text)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Quota exceeded" in err_msg or "rate limit" in err_msg.lower():
                    # Parse delay from message (e.g. Please retry in 34.258592779s.)
                    match = re.search(r'retry in ([\d\.]+)s', err_msg)
                    delay = float(match.group(1)) + 1.0 if match else (base_delay * (attempt + 1))
                    
                    logger.warning("GEMINI_RATE_LIMIT", extra={"attempt": attempt + 1, "delay": delay, "error": err_msg})
                    time.sleep(delay)
                else:
                    logger.error("GEMINI_API_ERROR", extra={"error": err_msg})
                    return None
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

    def _generate_fallback_summary(self, title: str, content: str) -> dict:
        """Generate a realistic AI-style summary and key points locally using heuristics."""
        import re
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        # 1. Summary: Use first 2-3 sentences
        sentences = []
        for p in paragraphs[:3]:
            for s in re.split(r'\. |\? |\! ', p):
                s = s.strip()
                if len(s) > 15:
                    sentences.append(s)
                    if len(sentences) >= 3:
                        break
            if len(sentences) >= 3:
                break
                
        summary_text = " ".join(sentences)
        if not summary_text:
            summary_text = f"This macroeconomic event covers: {title}. It represents a key policy update or official announcement from the Reserve Bank of India."
        else:
            if not summary_text.endswith('.'):
                summary_text += '.'
                
        # 2. Key points: Extract bullet points from content
        key_points = []
        for p in paragraphs:
            for s in re.split(r'\. |\? |\! ', p):
                s = s.strip()
                if len(s) > 20 and any(char.isdigit() or symbol in s for char in s for symbol in ['%', 'INR', 'Rs', 'crore', 'percent', 'policy', 'rate']):
                    if s not in key_points and len(key_points) < 4:
                        if not s.endswith('.'):
                            s += '.'
                        key_points.append(s)
                        
        # Fallback key points
        if len(key_points) < 3:
            key_points.append(f"Official announcement regarding {title}.")
            key_points.append("Aims to ensure monetary stability and financial regulations.")
            key_points.append("Provides guidance on macroeconomic trends and system liquidity.")
            
        return {
            "summary": summary_text,
            "key_points": key_points[:5],
            "confidence": 80
        }

    def process(self, event: MacroEvent) -> MacroEvent:
        if not event.official_data.content:
            return event
            
        sanitized_title = self._sanitize_prompt_input(event.official_data.title or "")
        sanitized_content = self._sanitize_prompt_input(event.official_data.content or "")
        
        result = None
        try:
            result = self.provider.summarize(sanitized_title + "\n\n" + sanitized_content)
        except Exception as e:
            logger.error("PROVIDER_SUMMARIZE_EXCEPTION", extra={"error": str(e)})
            
        is_fallback = False
        if not result:
            logger.warning("PROVIDER_FAILED_SWITCHING_TO_HEURISTIC_FALLBACK", extra={"event_id": event.event_id})
            result = self._generate_fallback_summary(sanitized_title, sanitized_content)
            is_fallback = True
            
        if result:
            new_version = 1
            if event.derived_data.ai_snapshots:
                new_version = event.derived_data.ai_snapshots[-1].version + 1
                
            provider_name = "fallback_heuristic" if is_fallback else self.provider.provider_name
            model_name = "fallback_heuristic" if is_fallback else self.provider.model_name
            
            snapshot = AISummarySnapshot(
                version=new_version,
                provider=provider_name,
                model=model_name,
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
