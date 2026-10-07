import unittest

from smw_bowser_ai.action_space import ActionSpace


class ActionSpaceTests(unittest.TestCase):
    def test_loads_default_macros(self):
        space = ActionSpace.from_file()
        self.assertIn("idle", space.macros)
        self.assertIn("run_jump_right", space.macros)
        self.assertEqual(space.bridge_action("idle").buttons, ())

    def test_macro_expands_to_bridge_action(self):
        space = ActionSpace.from_file()
        action = space.bridge_action("run_jump_right")
        self.assertEqual(action.macro, "run_jump_right")
        self.assertGreater(action.frames, 1)
        self.assertIn("Right", action.buttons)
        self.assertIn("Y", action.buttons)
        self.assertIn("B", action.buttons)


if __name__ == "__main__":
    unittest.main()

