import random
from typing import List
import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class User:
    gender: str
    age: int
    pcode: str
    favorite_cuisine: list


def generate_users_segment1(user_num: int = 1000,
                            postcode_num: int = 50
                            ) -> List[User]:
    """
    SEGMENT 1: One trick pony
    --------------------------------
    * Single cuisine lover
    * Age: Gaussian distro with a mean of 60
    * AVG_RATING >4
    * AVG_DEL_TIME <= 35 min
    * HAS_OFFER: 60% influence
    * 90% consistent
    """
    postcode_list = ['PC' + str(i) for i in range(postcode_num)]

    users = []

    for i in range(user_num):
        gender = random.choice(["M", "F"])  # pick a random gender

        pcode = random.choice(postcode_list)  # pick a random postcode

        age = int(np.random.normal(loc=60, scale=7, size=1)[0])  # random.randint(18,100) # pick a random age

        # pick random single favorite cuisine
        fave = random.sample(['Pizza', 'Burger', 'Pasta',
                              'Souvlaki', 'Sushi', 'Chinese'], 1)

        # append the new user
        users.append(User(gender, age, pcode, fave))

    return users


def generate_ratings_segment1(
        users: list,
        restaurants: list
):
    """
    SEGMENT 1: One trick pony
    --------------------------------
    * Single cuisine lover
    * Age: Gaussian distro with a mean of 60
    * AVG_RATING >4
    * AVG_DEL_TIME <= 35 min
    * HAS_OFFER: 60% influence
    * 90% consistent
    """

    fw = open('segment1.csv', 'w')
    writer = csv.writer(fw)

    writer.writerow(['age', 'gender', 'pcode', 'favorite_cuisine', 'restaurant_cuisine', 'price_range', 'has_self_del',
                     'has_offer', 'has_extra_del_cost', 'min_cost',
                     'avg_rating', 'avg_del_time', 'payment_methods', 'rating'])

    cnt = 0
    total = 0
    for usr in users:  # for each user

        rating_num = random.randint(1, len(restaurants))  # get the number of ratings
        total += rating_num
        my_restaurants = random.sample(restaurants, rating_num)  # sample restaurants to be rated

        for rst in my_restaurants:  # for each restaurant

            rating = 0  # initialize to negative rating

            # if the restaurant has my single favorite cuisine, and a rating over 4 and a delivery time<35 min
            if (usr.favorite_cuisine[0] in rst.cuisine) and (rst.avg_rating > 4) and (rst.avg_del_time < 35):

                # one random 50-50 chance, and one 60% chance if the rest has an offer
                if random.random() or (rst.has_offer and random.random() <= 0.6):
                    rating = 1
                    cnt += 1

            if random.random() < 0.1:  # consistency switch
                rating *= -1

            new_row = [usr.age, usr.gender, usr.pcode, usr.favorite_cuisine, rst.cuisine, rst.price_range,
                       rst.has_self_del, rst.has_offer, rst.has_extra_del_cost,
                       rst.min_cost, rst.avg_rating, rst.avg_del_time,
                       rst.payment_methods, rating]

            writer.writerow(new_row)

    fw.close()

    print('ones', cnt / total)


def generate_users_segment2(user_num: int = 1000,
                            postcode_num: int = 50
                            ):
    """
    SEGMENT 2: young+price-driven
    --------------------------------
    * Diverse set of favorite cuisines, but each person has their fave
    * Age: Gaussian distro with a mean of 20
    * AVG_RATING >4.2 if price in [$$$]
    * AVG_RATING >3.5 if price in [$$]
    * AVG_RATING >3 if price in [$]
    * no way if price in [$$$$]
    * no way if has_extra_del_cost
    * HAS_OFFER: 80% influence if price in [$$$], 60% influence if [$,$$]
    * 80% consistent
    """
    postcode_list = ['PC' + str(i) for i in range(postcode_num)]

    users = []

    for i in range(user_num):
        gender = random.choice(["M", "F"])  # pick a random gender

        pcode = random.choice(postcode_list)  # pick a random postcode

        age = int(np.random.normal(loc=20, scale=7, size=1)[0])  # random.randint(18,100) # pick a random age

        # pick random favorite cuisines
        fave = random.sample(['Pizza', 'Burger', 'Pasta',
                              'Souvlaki', 'Sushi', 'Chinese'], random.randint(1, 6))

        # append the new user
        users.append(User(gender, age, pcode, fave))

    return users


def generate_ratings_segment2(
        users: list,
        restaurants: list,
):
    """
    SEGMENT 2: young+price-driven
    --------------------------------
    * Diverse set of favorite cuisines, but each person has their fave
    * Age: Gaussian distro with a mean of 20
    * AVG_RATING >4.2 if price in [$$$]
    * AVG_RATING >3.5 if price in [$$]
    * AVG_RATING >3 if price in [$]
    * no way if price in [$$$$]
    * no way if has_extra_del_cost
    * HAS_OFFER: 80% influence if price in [$$$], 60% influence if [$,$$]
    * 80% consistent
    """

    fw = open('segment2.csv', 'w')
    writer = csv.writer(fw)

    writer.writerow(['age', 'gender', 'pcode', 'favorite_cuisine', 'restaurant_cuisine', 'price_range', 'has_self_del',
                     'has_offer', 'has_extra_del_cost', 'min_cost',
                     'avg_rating', 'avg_del_time', 'payment_methods', 'rating'])

    cnt = 0
    total = 0

    for usr in users:  # for each user

        rating_num = random.randint(1, len(restaurants))  # get the number of ratings
        total += rating_num
        my_restaurants = random.sample(restaurants, rating_num)  # sample restaurants to be rated

        for rst in my_restaurants:  # for each restaurant

            rating = 0  # initialize to negative rating

            # if the restaurant has my single favorite cuisine, and a rating over 4 and a delivery time<35 min
            if (any(element in rst.cuisine for element in usr.favorite_cuisine)) and \
                    (not rst.has_extra_del_cost):

                if (rst.price_range == '$' and rst.avg_rating > 3.5) or \
                        (rst.price_range == '$$' and rst.avg_rating > 4) or \
                        (rst.price_range == '$$$' and rst.avg_rating > 4.5):

                    # one random 50-50 chance, or offer boosts
                    if random.random() or \
                            (rst.has_offer and rst.price_range in ['$', '$$'] and random.random() <= 0.6) or \
                            (rst.has_offer and rst.price_range == '$$$' and random.random() <= 0.8):
                        rating = 1
                        cnt += 1

            if random.random() < 0.2:  # consistency switch
                rating *= -1

            new_row = [usr.age, usr.gender, usr.pcode, usr.favorite_cuisine, rst.cuisine, rst.price_range,
                       rst.has_self_del, rst.has_offer, rst.has_extra_del_cost,
                       rst.min_cost, rst.avg_rating, rst.avg_del_time,
                       rst.payment_methods, rating]

            writer.writerow(new_row)

    fw.close()

    print('ones', cnt / total, cnt, total)

