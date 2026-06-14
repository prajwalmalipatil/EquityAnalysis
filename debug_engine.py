from src.services.vsa.eigen_transition_engine_service import EigenTransitionEngineService
import pandas as pd

engine = EigenTransitionEngineService('daily', events_dir=None) # Memory only
df = pd.DataFrame({
    'Date': ['2026-06-11', '2026-06-12'],
    'Volume': [1000, 50079116.0],
    'High': [100, 134],
    'Low': [90, 122.96]
})
engine.update_active_sequences("IREDA", df)
engine.detect_triggers("IREDA", df, True, "Neutral")
sequences, _ = engine.reconstruct_state()
print("After detect:")
for s in sequences.values():
    print(s.model_dump_json(indent=2))

print("Running update again...")
engine.update_active_sequences("IREDA", df)
sequences, _ = engine.reconstruct_state()
for s in sequences.values():
    print(s.model_dump_json(indent=2))

