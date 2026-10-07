import unittest
from pydantic import ValidationError
from rallylab.api import CourseSpec


class LongCourseTests(unittest.TestCase):
    def test_existing_requests_keep_one_kilometre(self):
        self.assertEqual(CourseSpec(name="Old", seed=1, family="gentle").length, 1000)

    def test_three_kilometre_clear_course(self):
        spec = CourseSpec(name="Endurance", seed=43001, family="technical",
                          length=3000, rocks=0, scenery=False)
        self.assertEqual(spec.model_dump()["length"], 3000)
        self.assertFalse(spec.scenery)

    def test_rejects_unbounded_lengths(self):
        for length in (0, 999, 5001):
            with self.assertRaises(ValidationError):
                CourseSpec(name="Invalid", seed=1, family="gentle", length=length)
