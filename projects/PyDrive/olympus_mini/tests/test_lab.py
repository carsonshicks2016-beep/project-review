import unittest
import json
from olympus_mini.lab.server import app, sim_manager, train_manager

class TestOlympusLab(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"OLYMPUS", response.data)
        self.assertIn(b"MINI", response.data)

    def test_telemetry_endpoint(self):
        response = self.client.get("/api/telemetry")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("task", data)
        self.assertIn("vx", data)
        self.assertIn("height", data)
        self.assertIn("foot_contacts", data)
        self.assertEqual(len(data["foot_contacts"]), 8)

    def test_control_commands(self):
        # Pause
        r = self.client.post("/api/control", json={"command": "pause"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(sim_manager.is_paused)
        
        # Play
        r = self.client.post("/api/control", json={"command": "play"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(sim_manager.is_paused)
        
        # Camera
        r = self.client.post("/api/control", json={"command": "set_camera", "camera": "orbit"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sim_manager.camera_preset, "orbit")

    def test_manual_action(self):
        r = self.client.post("/api/manual_action", json={"lean": 0.45, "cadence": 2.2, "thrust": 0.8})
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(sim_manager.manual_lean, 0.45)
        self.assertAlmostEqual(sim_manager.manual_cadence, 2.2)

    def test_train_api(self):
        # Start
        r = self.client.post("/api/train/start", json={"task": "sprint", "steps": 1024})
        self.assertEqual(r.status_code, 200)
        # Stop
        r = self.client.post("/api/train/stop")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(train_manager.is_training)

if __name__ == "__main__":
    unittest.main()
