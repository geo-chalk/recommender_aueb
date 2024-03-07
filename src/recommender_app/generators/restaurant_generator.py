"""
Module containing functions needed to generate restaurant objects
"""

import random
from typing import List, Tuple
import numpy as np

from recommender_app.generators import Restaurant


def generate_restaurants(cuisines: Tuple[str] = ('Pizza', 'Burger', 'Pasta', 'Souvlaki', 'Sushi', 'Chinese'),
                         max_cuisines: int = 3,
                         price_samples: Tuple[float] = (0.5, 0.3, 0.15, 0.05),
                         has_extra_del_cost_prob: float = 0.2,
                         min_cost: Tuple[int, int] = (6, 3),
                         avg_rating: Tuple[float, int] = (3.7, 1),
                         avg_del_time: Tuple[int, int] = (30, 7),
                         payment_methods: Tuple[Tuple[str], Tuple[str]] = (["CASH", "CARD"], ['COUPON']),
                         rest_num: int = 100
                         ) -> List[Restaurant]:
    """
    Generate a list of Restaurant objects with random attributes.

    Args:
        cuisines: A tuple of possible cuisine types.
        max_cuisines: The maximun allowed number of cuisines per restaurant.
        price_samples: A tuple representing the probability distribution of price ranges.
        has_extra_del_cost_prob: The probability that a restaurant has extra delivery cost.
        min_cost: A tuple representing the mean and standard deviation of the minimum cost.
        avg_rating: A tuple representing the mean and standard deviation of the average rating.
        avg_del_time: A tuple representing the mean and standard deviation of the average delivery time.
        payment_methods: A tuple containing two tuples of possible payment methods.
        rest_num: The number of restaurants to generate.

    Returns:
        List[Restaurant]: A list of Restaurant objects.
    """

    restaurants = []

    # Iterate over number of restaurants
    for i in range(rest_num):
        # define price range
        price_sample = np.random.multinomial(1, list(price_samples)).argmax()
        price_range = random.choice(["$" * (i + 1) for i in range(len(price_samples))][price_sample])

        # define restaurant dict
        restaurant_args: dict[str, bool | float | int] = dict(
            cuisine=random.sample(cuisines, random.randint(1, max_cuisines)),
            price_range=price_range,
            has_self_del=random.choice([True, False]),
            has_offer=random.choice([True, False]),
            has_extra_del_cost=True if random.random() < has_extra_del_cost_prob else False,
            min_cost=round(random.gauss(min_cost[0], min_cost[1]), 1),
            avg_rating=round(random.gauss(avg_rating[0], avg_rating[1]), 1),
            avg_del_time=round(random.gauss(avg_del_time[0], avg_del_time[1])),
            payment_methods=random.sample(payment_methods[0], 2) + random.randint(0, 1) * payment_methods[1]
        )

        restaurants.append(Restaurant(**restaurant_args))

    return restaurants
