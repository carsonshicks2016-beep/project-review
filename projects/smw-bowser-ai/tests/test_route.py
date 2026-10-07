import unittest

from smw_bowser_ai.route import RouteManifest


class RouteTests(unittest.TestCase):
    def test_starworld_detection(self):
        route = RouteManifest.from_file()
        self.assertTrue(route.detect_starworld({"level_name": "Star World 4"}))
        self.assertFalse(route.detect_starworld({"level_name": "Yoshi's Island 2"}))

    def test_complete_route_trace_validates(self):
        route = RouteManifest.from_file()
        events = [
            {"frame": index, "route_step": step.id}
            for index, step in enumerate(route.steps)
            if step.required
        ]
        result = route.validate_trace(events)
        self.assertTrue(result.ok, [issue.message for issue in result.issues])

    def test_savestate_rejected_in_final_trace(self):
        route = RouteManifest.from_file()
        events = [
            {"frame": index, "route_step": step.id}
            for index, step in enumerate(route.steps)
            if step.required
        ]
        events.append({"frame": 999, "savestate_used": True})
        result = route.validate_trace(events)
        self.assertFalse(result.ok)
        self.assertIn("savestate_used", {issue.code for issue in result.issues})


if __name__ == "__main__":
    unittest.main()

