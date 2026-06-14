import pytest
from src.services.macro_intelligence.search_indexer import SearchIndexer, SearchDocumentBuilder
from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData

def test_search_indexer():
    events = [
        MacroEvent(
            event_id="e1",
            official_data=OfficialData(
                title="Monetary Policy Report",
                publication_date="2026-06-14",
                category="monetary",
                source="RBI",
                official_url="",
                content="The central bank increased the repo rate to control liquidity.",
                attachments=[],
                pdf_url=""
            ),
            derived_data=DerivedData(
                ai_summary="Interest rates hiked.",
                ai_theme="Interest Rates",
                keywords=["repo rate", "liquidity"],
                market_relevance="High",
                quality_score=90,
                ai_snapshots=[]
            ),
            metadata=None
        ),
        MacroEvent(
            event_id="e2",
            official_data=OfficialData(
                title="Banking Regulation Updates",
                publication_date="2026-06-15",
                category="banking",
                source="RBI",
                official_url="",
                content="New compliance rules for liquidity coverage ratio.",
                attachments=[],
                pdf_url=""
            ),
            derived_data=DerivedData(
                ai_summary="Compliance rules updated.",
                ai_theme="Regulation",
                keywords=["compliance", "liquidity coverage"],
                market_relevance="Medium",
                quality_score=85,
                ai_snapshots=[]
            ),
            metadata=None
        )
    ]
    
    builder = SearchDocumentBuilder()
    search_docs = [builder.build(e) for e in events]
    
    indexer = SearchIndexer()
    index = indexer.build_index(search_docs)
    
    # Check stop words removed
    assert "the" not in index
    
    # Check simple term
    assert "repo" in index
    assert "e1" in index["repo"]
    assert "e2" not in index["repo"]
    
    # Check term in both
    assert "liquidity" in index
    assert "e1" in index["liquidity"]
    assert "e2" in index["liquidity"]
    
    # Check derived data included (keywords, themes)
    assert "compliance" in index
    assert "e2" in index["compliance"]
    
    # Check that lengths are > 2
    assert "to" not in index
