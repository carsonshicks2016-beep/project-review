import json
import unittest

from smw_bowser_ai.protocol import (
    BridgeAction,
    BridgeObservation,
    extract_json_objects,
    make_length_prefixed,
    parse_length_prefixed,
)


class ProtocolTests(unittest.TestCase):
    def test_length_prefixed_roundtrip(self):
        action = BridgeAction(buttons=("Right", "Y"), macro="run_right")
        message, rest = parse_length_prefixed(make_length_prefixed(action.to_json()))
        self.assertFalse(rest)
        decoded = json.loads(message)
        self.assertEqual(decoded["buttons"], ["Right", "Y"])

    def test_extract_streamed_json_objects(self):
        objects, rest = extract_json_objects('{"frame":1}{"frame":2}{"partial"')
        self.assertEqual([obj["frame"] for obj in objects], [1, 2])
        self.assertEqual(rest, '{"partial"')

    def test_observation_from_dict(self):
        observation = BridgeObservation.from_dict(
            {"schema": "smw-bowser-ai-bridge-v1", "frame": 9, "ram": {"game_mode": 20}}
        )
        self.assertEqual(observation.frame, 9)
        self.assertEqual(observation.ram["game_mode"], 20)


if __name__ == "__main__":
    unittest.main()

