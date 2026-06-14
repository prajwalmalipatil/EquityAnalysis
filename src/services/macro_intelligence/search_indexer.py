import re
from typing import List, Dict, Set
from collections import defaultdict
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.read_models import SearchDocument

class SearchDocumentBuilder:
    """Extracts indexable tokens from a MacroEvent to decouple the domain from the indexer."""
    
    def __init__(self):
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", 
            "to", "for", "of", "with", "by", "from", "this", "that",
            "is", "are", "was", "were", "be", "been", "it", "as"
        }
        
    def _tokenize(self, text: str) -> Set[str]:
        if not text:
            return set()
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return {t for t in tokens if t not in self.stop_words and len(t) > 2}

    def build(self, event: MacroEvent) -> SearchDocument:
        text_blocks = [
            event.official_data.title,
            event.official_data.category,
            event.official_data.source,
            event.official_data.content,
        ]
        
        if event.derived_data:
            if event.derived_data.ai_theme:
                text_blocks.append(event.derived_data.ai_theme)
            if event.derived_data.ai_summary:
                text_blocks.append(event.derived_data.ai_summary)
            if event.derived_data.keywords:
                text_blocks.extend(event.derived_data.keywords)
                
        combined_text = " ".join(filter(None, text_blocks))
        tokens = self._tokenize(combined_text)
        return SearchDocument(doc_id=event.event_id, tokens=list(tokens))

class SearchIndexer:
    """Builds an inverted index from SearchDocuments for fast frontend search."""
    
    def build_index(self, documents: List[SearchDocument]) -> Dict[str, List[str]]:
        index = defaultdict(set)
        for doc in documents:
            for token in doc.tokens:
                index[token].add(doc.doc_id)
                
        # Sort keys and doc_ids to ensure byte-for-byte reproducibility
        return {token: sorted(list(doc_ids)) for token, doc_ids in index.items()}
