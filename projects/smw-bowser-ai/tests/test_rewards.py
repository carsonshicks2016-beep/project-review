import unittest

from smw_bowser_ai.protocol import BridgeObservation
from smw_bowser_ai.rewards import RewardModel
from smw_bowser_ai.route import RouteManifest


class RewardTests(unittest.TestCase):
    def test_progress_reward_positive(self):
        route = RouteManifest.from_file()
        rewards = RewardModel(route)
        previous = BridgeObservation(frame=1, ram={"screen_x": 100, "mario_x": 110, "lives": 5})
        current = BridgeObservation(frame=2, ram={"screen_x": 164, "mario_x": 170, "lives": 5})
        score = rewards.score(previous, current)
        self.assertGreater(score.progress, 0)

    def test_death_penalty(self):
        route = RouteManifest.from_file()
        rewards = RewardModel(route)
        previous = BridgeObservation(frame=1, ram={"game_mode": 20, "player_state": 0, "lives": 5})
        current = BridgeObservation(frame=2, ram={"game_mode": 20, "player_state": 9, "lives": 4})
        score = rewards.score(previous, current)
        self.assertLess(score.death, 0)
        self.assertLess(score.total, 0)

    def test_level_clear_reward(self):
        route = RouteManifest.from_file()
        rewards = RewardModel(route)
        current = BridgeObservation(frame=2, ram={"goal_state": 1, "lives": 5})
        score = rewards.score(None, current)
        self.assertGreaterEqual(score.level_clear, 25)


if __name__ == "__main__":
    unittest.main()

