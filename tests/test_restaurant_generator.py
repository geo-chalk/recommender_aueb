import unittest
from src.recommender_app.generators.restaurant_generator import generate_restaurants
from src.recommender_app.generators.restaurant_generator import Restaurant


class TestRestaurant(unittest.TestCase):
    def setUp(self):
        self.restaurant_args = {
            "cuisine": ["Italian"],
            "price_range": "$$",
            "has_self_del": True,
            "has_offer": False,
            "has_extra_del_cost": False,
            "min_cost": 6.0,
            "avg_rating": 3.7,
            "avg_del_time": 30,
            "payment_methods": ["CASH", "CARD"]
        }
        self.restaurant = Restaurant(**self.restaurant_args)

    def test_attributes(self):
        for key, value in self.restaurant_args.items():
            self.assertEqual(getattr(self.restaurant, key), value)


class TestGenerateRestaurants(unittest.TestCase):
    def setUp(self):
        self.rest_num = 10

    def test_generate_restaurants(self):
        restaurants = generate_restaurants(rest_num=self.rest_num)

        # Check if the correct number of restaurants are generate
        self.assertEqual(len(restaurants), self.rest_num)

        # Check if all generated objects are instances of Restaurant class
        for restaurant in restaurants:
            self.assertIsInstance(restaurant, Restaurant)


if __name__ == '__main__':
    unittest.main()
