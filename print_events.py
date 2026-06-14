import json

with open("data/events/daily_events.jsonl") as f:
    events = [json.loads(l) for l in f]

ireda_events = [e for e in events if e.get("symbol") == "IREDA"]
for e in ireda_events[-5:]:
    print(e.get("action"), e.get("stage"), e.get("timestamp"))

