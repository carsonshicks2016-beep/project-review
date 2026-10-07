from unittest.mock import MagicMock
from pathlib import Path
from melee_lab import panel

def test_panel_payload_recurrent_and_slots(tmp_path):
    root = tmp_path
    runs = root / "runs"
    runs.mkdir()
    run_dir = runs / "20260915-000000-test1"
    run_dir.mkdir()
    
    (run_dir / "request.json").write_text('{"recurrent": true, "lstm_hidden_size": 128, "anchor": "dataset.npz"}')
    (run_dir / "status.json").write_text('{"status": "running", "frame": 450, "steps": 1200, "action": "n-air"}')
    (run_dir / "latest.json").write_text('{"steps": 1200}')
    
    manager = MagicMock()
    manager.runs = runs
    manager.root = root
    manager.list_runs.return_value = [
        {"id": "20260915-000000-test1", "status": "running", "steps": 1200, "slots": [
            {"index": 0, "frame": 450, "players": {"2": {"character": "MARIO"}}, "age_seconds": 0.5}
        ]}
    ]
    manager.matches.read.return_value = ([], 0)
    
    payload = panel.payload(manager)
    assert payload["run"]["id"] == "20260915-000000-test1"
    assert payload["run"]["recurrent"] is True
    assert payload["run"]["lstm_hidden_size"] == 128
    assert payload["run"]["live_frame"] == 450
    assert len(payload["slots"]) == 1
    assert payload["lineage"][-1]["kind"] == "Anchored Recurrent PPO"
    assert payload["lineage"][-1]["recurrent"] is True
