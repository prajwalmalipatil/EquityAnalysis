from typing import List, Dict, Optional
import re
import hashlib
from datetime import datetime
import uuid
from collections import defaultdict
from itertools import combinations

from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.relationship_models import (
    RelationshipType, RelationshipConfidence, RuleResult, RelationshipCandidate, Relationship
)

class RelationshipCandidateGenerator:
    """Generates candidates by finding commonalities (category, keywords) to avoid O(N^2)."""
    
    def generate(self, events: List[MacroEvent]) -> List[RelationshipCandidate]:
        candidates = []
        
        # We index events by category and by keyword to quickly find overlap
        category_index = defaultdict(list)
        keyword_index = defaultdict(list)
        
        for e in events:
            category_index[e.official_data.category].append(e)
            if e.derived_data and e.derived_data.keywords:
                for kw in e.derived_data.keywords:
                    keyword_index[kw].append(e)
        
        # A simple pair tracker to avoid duplicates
        seen_pairs = set()
        
        # 1. Same category candidates
        for cat, evs in category_index.items():
            if not cat: continue
            if len(evs) > 100:
                # If a category is too large, it loses its heuristic value, 
                # but we'll limit it or rely on chronological adjacency.
                # For simplicity, we just sort by date and take sliding windows.
                evs = sorted(evs, key=lambda x: x.official_data.publication_date, reverse=True)
                for i in range(len(evs) - 1):
                    # Compare only adjacent ones in the same category
                    pair = tuple(sorted([evs[i].event_id, evs[i+1].event_id]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        candidates.append(RelationshipCandidate(
                            source_event_id=evs[i].event_id,
                            target_event_id=evs[i+1].event_id,
                            generator_score=0.5,
                            generator_signals=["Same Category (Temporal Proximity)"]
                        ))
            else:
                for e1, e2 in combinations(evs, 2):
                    pair = tuple(sorted([e1.event_id, e2.event_id]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        candidates.append(RelationshipCandidate(
                            source_event_id=e1.event_id,
                            target_event_id=e2.event_id,
                            generator_score=0.5,
                            generator_signals=["Same Category"]
                        ))
        
        # 2. Shared keywords
        for kw, evs in keyword_index.items():
            if len(evs) > 50:
                continue # Keyword too generic
            for e1, e2 in combinations(evs, 2):
                pair = tuple(sorted([e1.event_id, e2.event_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidates.append(RelationshipCandidate(
                        source_event_id=e1.event_id,
                        target_event_id=e2.event_id,
                        generator_score=0.7,
                        generator_signals=["Shared Keyword: " + kw]
                    ))
                    
        return candidates

class RelationshipRule:
    def evaluate(self, source: MacroEvent, target: MacroEvent) -> Optional[RuleResult]:
        raise NotImplementedError

class SupersedesRule(RelationshipRule):
    def evaluate(self, source: MacroEvent, target: MacroEvent) -> Optional[RuleResult]:
        score = 0.0
        evidence = []
        
        s_date = source.official_data.publication_date
        t_date = target.official_data.publication_date
        
        # A cannot supersede B if A is older than B
        if s_date < t_date:
            return None
            
        content = (source.official_data.content or "").lower()
        title = (source.official_data.title or "").lower()
        
        if "supersedes" in content or "supersedes" in title:
            score += 0.85
            evidence.append("explicit keyword 'supersedes'")
            
        if "replaces" in content or "replaces" in title:
            score += 0.75
            evidence.append("explicit keyword 'replaces'")
            
        if score > 0:
            return RuleResult(
                relationship_type=RelationshipType.SUPERSEDES,
                score=min(score + 0.1, 1.0), # base bump for being a candidate
                rule_name="SupersedesRule",
                matched_terms=evidence,
                signals={"keyword_match": score}
            )
        return None

class AmendsRule(RelationshipRule):
    def evaluate(self, source: MacroEvent, target: MacroEvent) -> Optional[RuleResult]:
        score = 0.0
        evidence = []
        
        content = (source.official_data.content or "").lower()
        title = (source.official_data.title or "").lower()
        
        for term in ["amends", "modified", "substituted", "amendment"]:
            if term in content or term in title:
                score += 0.4
                evidence.append(f"matched '{term}'")
                
        if score > 0:
            return RuleResult(
                relationship_type=RelationshipType.AMENDS,
                score=min(score + 0.2, 1.0),
                rule_name="AmendsRule",
                matched_terms=evidence,
                signals={"keyword_match": score}
            )
        return None

class WithdrawsRule(RelationshipRule):
    def evaluate(self, source: MacroEvent, target: MacroEvent) -> Optional[RuleResult]:
        score = 0.0
        evidence = []
        
        content = (source.official_data.content or "").lower()
        title = (source.official_data.title or "").lower()
        
        for term in ["withdrawn", "rescinded", "stands cancelled"]:
            if term in content or term in title:
                score += 0.6
                evidence.append(f"matched '{term}'")
                
        if score > 0:
            return RuleResult(
                relationship_type=RelationshipType.WITHDRAWS,
                score=min(score + 0.2, 1.0),
                rule_name="WithdrawsRule",
                matched_terms=evidence,
                signals={"keyword_match": score}
            )
        return None

class RelationshipResolver:
    def __init__(self):
        self.rules: List[RelationshipRule] = [
            SupersedesRule(),
            AmendsRule(),
            WithdrawsRule()
        ]
        
    def _map_confidence(self, score: float) -> RelationshipConfidence:
        if score >= 0.90:
            return RelationshipConfidence.HIGH
        elif score >= 0.70:
            return RelationshipConfidence.MEDIUM
        return RelationshipConfidence.LOW
        
    def resolve(self, candidates: List[RelationshipCandidate], events: List[MacroEvent]) -> List[Relationship]:
        event_map = {e.event_id: e for e in events}
        relationships = []
        
        for candidate in candidates:
            source = event_map.get(candidate.source_event_id)
            target = event_map.get(candidate.target_event_id)
            
            if not source or not target:
                continue
                
            # Evaluate both directions
            for s, t in [(source, target), (target, source)]:
                best_result = None
                for rule in self.rules:
                    result = rule.evaluate(s, t)
                    if result:
                        if not best_result or result.score > best_result.score:
                            best_result = result
                            
                if best_result and best_result.score >= 0.50:
                    raw_id = f"{s.event_id}:{t.event_id}:{best_result.relationship_type.value}"
                    canon_id = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:16]
                    
                    provenance = {
                        "rule": best_result.rule_name,
                        "matched_terms": best_result.matched_terms,
                        "signals": best_result.signals,
                        "weights": {"generator": candidate.generator_score, "rule": best_result.score},
                        "generator_signals": candidate.generator_signals
                    }
                    
                    rel = Relationship(
                        id=canon_id,
                        source_event_id=s.event_id,
                        target_event_id=t.event_id,
                        type=best_result.relationship_type,
                        confidence=self._map_confidence(best_result.score),
                        provenance=provenance,
                        rule_version="1.0.0",
                        resolver_version="1.0.0",
                        created_at=datetime.utcnow().isoformat() + "Z"
                    )
                    relationships.append(rel)
                    
        return relationships
