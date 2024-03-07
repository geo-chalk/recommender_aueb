import random
from typing import List
from dataclasses import dataclass

import numpy as np


@dataclass
class Restaurant:
    cuisine: list
    price_range: str
    has_self_del: bool
    has_offer: bool
    has_extra_del_cost: bool
    min_cost: float
    avg_rating: float
    avg_del_time: int
    payment_methods: list


def generate_restaurants(rest_num: int = 100) -> List[Restaurant]:
    restaurants = []

    for i in range(rest_num):
        cuisine_num = random.randint(1, 3)

        cuisine = random.sample(['Pizza', 'Burger', 'Pasta',
                                 'Souvlaki', 'Sushi', 'Chinese'], cuisine_num)

        price_sample = np.random.multinomial(1, [0.5, 0.3, 0.15, 0.05]).argmax()
        price_range = random.choice(["$", "$$", "$$$", "$$$$"][price_sample])

        has_self_del = random.choice([True, False])

        has_offer = random.choice([True, False])

        has_extra_del_cost = True if random.random() < 0.2 else False

        min_cost = round(random.gauss(6, 3), 1)

        avg_rating = round(random.gauss(3.7, 1), 1)

        avg_del_time = round(random.gauss(30, 7))

        payment_methods = random.sample(["CASH", "CARD"], 2) + random.randint(0, 1) * ['COUPON']

        restaurants.append(Restaurant(cuisine, price_range, has_self_del,
                                      has_offer, has_extra_del_cost, min_cost,
                                      avg_rating, avg_del_time, payment_methods))

    return restaurants