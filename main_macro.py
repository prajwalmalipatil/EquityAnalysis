from src.services.macro_intelligence.macro_pipeline import run_macro_pipeline
from src.services.orchestration.registry import platform_registry, ResearchModule

platform_registry.register(ResearchModule(
    name="MacroIntelligence",
    version="1.0.0",
    description="Rule-based Impact and Event Study Engine",
    inputs=["RBI Data"],
    outputs=["MacroEvents"],
    dependencies=[]
))

if __name__ == "__main__":
    import sys
    success = run_macro_pipeline()
    sys.exit(0 if success else 1)
