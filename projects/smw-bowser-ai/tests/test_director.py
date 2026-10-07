import unittest

from smw_bowser_ai.action_space import ActionSpace
from smw_bowser_ai.director import DirectorFSM
from smw_bowser_ai.memory import MemoryMap
from smw_bowser_ai.protocol import BridgeObservation
from smw_bowser_ai.route import RouteManifest


class DirectorTests(unittest.TestCase):
    def setUp(self):
        self.action_space = ActionSpace.from_file()
        self.memory_map = MemoryMap.from_file()
        self.route = RouteManifest.from_file()
        self.director = DirectorFSM(self.action_space, self.memory_map, self.route)

    def test_level_mode_delegates_to_player(self):
        observation = BridgeObservation(
            frame=1,
            mode="level",
            ram={"game_mode": 20, "lives": 5, "player_state": 0, "mario_x": 10, "screen_x": 0},
        )
        decision = self.director.decide(observation)
        self.assertTrue(decision.delegate_to_player)
        self.assertFalse(decision.abort)

    def test_starworld_aborts(self):
        observation = BridgeObservation(
            frame=1,
            mode="overworld",
            ram={"game_mode": 14, "lives": 5},
            raw={"level_name": "Star World 1"},
        )
        decision = self.director.decide(observation)
        self.assertTrue(decision.abort)

    def test_title_pulses_start(self):
        observation = BridgeObservation(frame=45, mode="title_or_menu", ram={"game_mode": 1})
        decision = self.director.decide(observation)
        self.assertIn("Start", decision.action.buttons)


if __name__ == "__main__":
    unittest.main()

