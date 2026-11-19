import unittest

from recommend import PROFILES, STOCK_SCENARIOS, recommend


class RecommendationTests(unittest.TestCase):
    def test_results_respect_budget_and_stock_in_every_scenario(self):
        for profile, settings in PROFILES.items():
            for scenario in STOCK_SCENARIOS:
                with self.subTest(profile=profile, scenario=scenario):
                    result = recommend(profile, scenario)
                    self.assertLessEqual(len(result["items"]), 3)
                    for item in result["items"]:
                        self.assertLessEqual(item["price"], settings.budget)
                        self.assertGreater(item["stock"], 0)
                        self.assertNotIn(item["name"], result["trace"]["outOfStock"])

    def test_shortage_replaces_top_ranked_product(self):
        normal = recommend("desk", "normal")
        shortage = recommend("desk", "shortage")
        first = normal["items"][0]["name"]
        self.assertIn(first, shortage["trace"]["outOfStock"])
        self.assertNotEqual(first, shortage["items"][0]["name"])
        self.assertEqual(len(shortage["items"]), 3)
        self.assertIn(shortage["items"][0]["name"], shortage["copy"])

    def test_all_unavailable_returns_no_recommendations_or_copy(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                result = recommend(profile, "empty")
                self.assertEqual(result["items"], [])
                self.assertIsNone(result["copy"])

    def test_profiles_change_first_recommendation(self):
        first = {recommend(profile, "normal")["items"][0]["name"] for profile in PROFILES}
        self.assertEqual(len(first), len(PROFILES))


if __name__ == "__main__":
    unittest.main()
