import pytest
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from src.services.vsa.eigen_transition_engine_service import EigenTransitionEngineService
from src.models.ete_models import ETEState

@pytest.fixture
def clean_engine(tmp_path):
    # Mock sequence config
    if not os.path.exists("src/constants"):
        os.makedirs("src/constants")
    config_path = "src/constants/ete_sequences_test.json"
    with open(config_path, "w") as f:
        json.dump({
            "AVE_1": {
                "sequence": ["HVHS", "LVLS"]
            }
        }, f)
        
    engine = EigenTransitionEngineService("daily", config_path=config_path, events_dir=tmp_path)
    yield engine
    if os.path.exists(config_path):
        os.remove(config_path)

def test_detect_triggers_idempotency(clean_engine):
    df = pd.DataFrame({
        "Date": [datetime.now()],
        "Open": [100], "High": [105], "Low": [95], "Close": [102], "Volume": [1000]
    })
    
    # First trigger
    clean_engine.detect_triggers("TEST", df, True)
    state, _ = clean_engine.reconstruct_state()
    assert len(state) == 1
    seq_id = list(state.keys())[0]
    assert state[seq_id].state == ETEState.TRIGGERED
    
    # Second trigger should not duplicate
    clean_engine.detect_triggers("TEST", df, True)
    state2, _ = clean_engine.reconstruct_state()
    assert len(state2) == 1

def test_advance_and_fail(clean_engine):
    df_trigger = pd.DataFrame({"Date": [datetime.now()], "Open": [10], "High": [15], "Low": [9], "Close": [12], "Volume": [100]})
    clean_engine.detect_triggers("TEST", df_trigger, True)
    
    # T+1: HVHS (Volume up, Spread up)
    df_t1 = pd.DataFrame({
        "Date": [datetime.now(), datetime.now()+timedelta(days=1)],
        "Open": [10, 10], "High": [15, 20], "Low": [9, 8], "Close": [12, 18], "Volume": [100, 200] # Spread 6 to 12, Vol 100 to 200
    })
    
    clean_engine.update_active_sequences("TEST", df_t1)
    state, _ = clean_engine.reconstruct_state()
    seq_id = list(state.keys())[0]
    assert state[seq_id].current_stage_index == 1
    assert state[seq_id].state == ETEState.WAITING
    
    # T+2: Expecting LVLS, but giving HVHS again -> should FAIL
    df_t2 = pd.DataFrame({
        "Date": [datetime.now()+timedelta(days=1), datetime.now()+timedelta(days=2)],
        "Open": [10, 10], "High": [20, 30], "Low": [8, 5], "Close": [18, 25], "Volume": [200, 300]
    })
    clean_engine.update_active_sequences("TEST", df_t2)
    state2, _ = clean_engine.reconstruct_state()
    assert state2[seq_id].state == ETEState.FAILED

def test_integrity_failure(clean_engine):
    df = pd.DataFrame({"Date": [datetime.now()], "Open": [10], "High": [15], "Low": [9], "Close": [12], "Volume": [100]})
    clean_engine.detect_triggers("TEST", df, True)
    
    # T+1
    df_t1 = pd.DataFrame({
        "Date": [datetime.now(), datetime.now()+timedelta(days=1)],
        "Open": [10, 10], "High": [15, 20], "Low": [9, 8], "Close": [12, 18], "Volume": [100, 200]
    })
    clean_engine.update_active_sequences("TEST", df_t1)
    
    # Manually corrupt the log (second event)
    with open(clean_engine.events_file, "r") as f:
        lines = f.readlines()
    
    event = json.loads(lines[1])
    event['matched'] = False # Tamper
    lines[1] = json.dumps(event) + "\n"
    with open(clean_engine.events_file, "w") as f:
        f.writelines(lines)
        
    state, _ = clean_engine.reconstruct_state()
    seq_id = list(state.keys())[0]
    assert state[seq_id].state == ETEState.INTEGRITY_FAILED

def test_hvls_and_lvhs(clean_engine):
    # Overwrite config to use HVLS and LVHS
    config_path = "src/constants/ete_sequences_test.json"
    with open(config_path, "w") as f:
        json.dump({
            "AVE_2": {
                "sequence": ["HVLS", "LVHS"]
            }
        }, f)
    # Re-instantiate engine to pick up new config
    engine = EigenTransitionEngineService("daily", config_path=config_path, events_dir=clean_engine.events_dir)

    df_trigger = pd.DataFrame({"Date": [datetime.now()], "Open": [10], "High": [20], "Low": [10], "Close": [15], "Volume": [100]}) # Spread = 10
    engine.detect_triggers("TEST", df_trigger, True)
    
    # T+1: HVLS (Volume up, Spread down)
    df_t1 = pd.DataFrame({
        "Date": [datetime.now(), datetime.now()+timedelta(days=1)],
        "Open": [10, 10], "High": [20, 15], "Low": [10, 10], "Close": [15, 12], "Volume": [100, 200] # Spread: 10 -> 5 (down), Vol: 100 -> 200 (up)
    })
    
    engine.update_active_sequences("TEST", df_t1)
    state, _ = engine.reconstruct_state()
    seq_id = list(state.keys())[0]
    assert state[seq_id].current_stage_index == 1
    assert state[seq_id].state == ETEState.WAITING
    
    # T+2: LVHS (Volume down, Spread up)
    df_t2 = pd.DataFrame({
        "Date": [datetime.now()+timedelta(days=1), datetime.now()+timedelta(days=2)],
        "Open": [10, 10], "High": [15, 27], "Low": [10, 10], "Close": [12, 20], "Volume": [200, 100] # Spread: 5 -> 17 (up), Vol: 200 -> 100 (down)
    })
    
    engine.update_active_sequences("TEST", df_t2)
    state2, _ = engine.reconstruct_state()
    assert state2[seq_id].state == ETEState.COMPLETED

def test_weekly_transition(tmp_path):
    config_path = "src/constants/ete_sequences_test_weekly.json"
    with open(config_path, "w") as f:
        json.dump({
            "AVE_1": {
                "sequence": ["HVHS", "LVLS"]
            }
        }, f)
        
    engine = EigenTransitionEngineService("weekly", config_path=config_path, events_dir=tmp_path)
    
    # T (trigger) on 2026-W23
    df_trigger = pd.DataFrame({
        "YearWeek": ["2026-W23"], "Open": [100], "High": [110], "Low": [90], "Close": [105], "Volume": [1000]
    })
    engine.detect_triggers("WEEKLY_TEST", df_trigger, True)
    
    # T+1 on 2026-W24 (HVHS: Vol up, Spread up)
    df_t1 = pd.DataFrame({
        "YearWeek": ["2026-W23", "2026-W24"],
        "Open": [100, 100], "High": [110, 130], "Low": [90, 80], "Close": [105, 120], "Volume": [1000, 1500]
    })
    
    engine.update_active_sequences("WEEKLY_TEST", df_t1)
    state, _ = engine.reconstruct_state()
    seq_id = list(state.keys())[0]
    assert state[seq_id].current_stage_index == 1
    assert state[seq_id].state == ETEState.WAITING
    
    # T+2 on 2026-W25 (LVLS: Vol down, Spread down)
    df_t2 = pd.DataFrame({
        "YearWeek": ["2026-W23", "2026-W24", "2026-W25"],
        "Open": [100, 100, 100], "High": [110, 130, 115], "Low": [90, 80, 95], "Close": [105, 120, 110], "Volume": [1000, 1500, 800]
    })
    
    engine.update_active_sequences("WEEKLY_TEST", df_t2)
    state2, _ = engine.reconstruct_state()
    assert state2[seq_id].state == ETEState.COMPLETED
    
    if os.path.exists(config_path):
        os.remove(config_path)

def test_monthly_transition(tmp_path):
    config_path = "src/constants/ete_sequences_test_monthly.json"
    with open(config_path, "w") as f:
        json.dump({
            "AVE_1": {
                "sequence": ["HVHS", "LVLS"]
            }
        }, f)
        
    engine = EigenTransitionEngineService("monthly", config_path=config_path, events_dir=tmp_path)
    
    # T (trigger) on 2026-05
    df_trigger = pd.DataFrame({
        "YearMonth": ["2026-05"], "Open": [100], "High": [110], "Low": [90], "Close": [105], "Volume": [1000]
    })
    engine.detect_triggers("MONTHLY_TEST", df_trigger, True)
    
    # T+1 on 2026-06 (HVHS: Vol up, Spread up)
    df_t1 = pd.DataFrame({
        "YearMonth": ["2026-05", "2026-06"],
        "Open": [100, 100], "High": [110, 130], "Low": [90, 80], "Close": [105, 120], "Volume": [1000, 1500]
    })
    
    engine.update_active_sequences("MONTHLY_TEST", df_t1)
    state, _ = engine.reconstruct_state()
    seq_id = list(state.keys())[0]
    assert state[seq_id].current_stage_index == 1
    assert state[seq_id].state == ETEState.WAITING
    
    if os.path.exists(config_path):
        os.remove(config_path)

