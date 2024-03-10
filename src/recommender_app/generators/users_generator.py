import random
from typing import List, Tuple
import csv
from pathlib import Path

import numpy as np

from recommender_app.generators import User, Restaurant
from recommender_app.utils.variables import CUISINES


class RatingsGenerator:
    """
        A class used to generate user ratings for restaurants.
    """

    def __init__(self,
                 output_dir: Path):
        self.output_dir = self._check_path(output_dir)

    @staticmethod
    def _check_path(path: Path) -> Path:
        """Checks and creates the directory if it does not exist."""
        # Create the parent directory
        path.mkdir(parents=True, exist_ok=True)
        return Path(path)

    @staticmethod
    def _generate_users_segment(user_num: int = 1000,
                                postcode_num: int = 50,
                                usr_age: Tuple[int, int] = (60, 7),
                                usr_cuisines: Tuple = CUISINES,
                                usr_cuisines_num: Tuple = (1, 1)
                                ) -> List[User]:
        """
        Generates a list of users with random attributes.

        Parameters:
            user_num (int): The number of users to generate. Defaults to 1000.
            postcode_num (int): The number of different postcodes. Defaults to 50.
            usr_age (Tuple[int, int]): A tuple containing the mean and standard deviation of the users' ages.
                Defaults to (60, 7).
            usr_cuisines (Tuple): A tuple containing possible favorite cuisines of the users.
                Defaults to ('Pizza', 'Burger', 'Pasta', 'Souvlaki', 'Sushi', 'Chinese').
            usr_cuisines_num (Tuple): A tuple containing the minimum and maximum number of favorite cuisines a user can have. Defaults to (1, 1).

        Returns:
            List[User]: A list of User objects with randomly generated attributes.
        """
        postcode_list: List[str] = ['PC' + str(i) for i in range(postcode_num)]

        users = []

        for i in range(user_num):
            usr = dict(
                gender=random.choice(["M", "F"]),
                pcode=random.choice(postcode_list),
                age=int(np.random.normal(loc=usr_age[0], scale=usr_age[1], size=1)[0]),
                favorite_cuisine=random.sample(usr_cuisines,
                                               random.randint(usr_cuisines_num[0],
                                                              usr_cuisines_num[1]))
            )

            # append the new user
            users.append(User(**usr))

        return users

    @staticmethod
    def segment_1(_user: User, _restaurant: Restaurant) -> Tuple[int, int]:
        """
        This function evaluates a restaurant based on a user's preferences and returns a rating and count.

        * The restaurant must offer the user's single favorite cuisine.
        * The restaurant should have an average rating over 4.
        * The restaurant's delivery time should be less than 35 minutes.
        * There is a 50% random chance of assigning a positive rating.
        * If the restaurant has an offer, there's an additional 60% random chance of assigning a positive rating.
        * A "consistency switch" exists with a 10% chance that can invert the rating.

        Args:
            _user (User): The user object containing user preferences.
            _restaurant (Restaurant): The restaurant object containing restaurant details.

        Returns:
            A tuple where the first element is the rating (1 or -1) and the second element is the count (0 or 1)
                which increments with each rating.
        """
        rating, cnt = 0, 0  # initialize to negative rating

        # if the restaurant has my single favorite cuisine, and a rating over 4 and a delivery time<35 min
        if ((_user.favorite_cuisine[0] in _restaurant.cuisine) and
                (_restaurant.avg_rating > 4) and (_restaurant.avg_del_time < 35)):

            # one random 50-50 chance, and one 60% chance if the rest has an offer
            if random.random() or (_restaurant.has_offer and random.random() <= 0.6):
                rating = 1
                cnt = 1

        if random.random() < 0.1:  # consistency switch
            rating *= -1

        return rating, cnt

    @staticmethod
    def segment_2(_user: User, _restaurant: Restaurant) -> Tuple[int, int]:
        """
            This function evaluates a restaurant based on a user's preferences and returns a rating and count.

            * The restaurant must offer any of the user's favorite cuisines.
            * The restaurant should not have extra delivery costs.
            * If the restaurant's price range is '$', it should have an average rating over 3.5.
            * If the restaurant's price range is '$$', it should have an average rating over 4.
            * If the restaurant's price range is '$$$', it should have an average rating over 4.5.
            * There is a 50% random chance of assigning a positive rating.
            * If the restaurant has an offer and its price range is either '$' or '$$', there's an additional
                60% random chance of assigning a positive rating.
            * If the restaurant has an offer and its price range is '$$$', there's an additional
                80% random chance of assigning a positive rating.
            * A "consistency switch" exists with a 20% chance that can invert the rating.

            Args:
                _user (User): The user object containing user preferences.
                _restaurant (Restaurant): The restaurant object containing restaurant details.

            Returns:
                A tuple where the first element is the rating (1 or -1) and the second element is the count (0 or 1)
                    which increments with each rating.
            """
        rating, cnt = 0, 0  # initialize to negative rating

        # if the restaurant has my single favorite cuisine, and a rating over 4 and a delivery time<35 min
        if (any(element in _restaurant.cuisine for element in _user.favorite_cuisine)) and \
                (not _restaurant.has_extra_del_cost):

            if (_restaurant.price_range == '$' and _restaurant.avg_rating > 3.5) or \
                    (_restaurant.price_range == '$$' and _restaurant.avg_rating > 4) or \
                    (_restaurant.price_range == '$$$' and _restaurant.avg_rating > 4.5):

                # one random 50-50 chance, or offer boosts
                if random.random() or \
                        (_restaurant.has_offer and _restaurant.price_range in ['$', '$$'] and random.random() <= 0.6) or \
                        (_restaurant.has_offer and _restaurant.price_range == '$$$' and random.random() <= 0.8):
                    rating = 1
                    cnt += 1

        if random.random() < 0.2:  # consistency switch
            rating *= -1

        return rating, cnt

    def generate_segment(self,
                         restaurants: List[Restaurant],
                         segment_id: str,
                         **kwargs
                         ) -> Path:
        """
        Generates user data for a specific segment, assigns ratings to restaurants and writes the data to a CSV file.
        Args:
            restaurants: A list of Restaurant objects.
            segment_id: The ID of the segment function to use for rating calculation.

        Returns:
            output_file: The path to the generated CSV file.
        """
        # generate users
        users = self._generate_users_segment(**kwargs)

        # open file path
        output_file = self.output_dir / f'{segment_id}.csv'
        fw = output_file.open(mode='w')
        writer = csv.writer(fw)

        writer.writerow(list(users[0].__dict__.keys())
                        + list(restaurants[0].__dict__.keys())
                        + ['rating'])

        cnt = 0
        total = 0
        for usr in users:  # for each user

            rating_num = random.randint(1, len(restaurants))  # get the number of ratings
            total += rating_num
            my_restaurants = random.sample(restaurants, rating_num)  # sample restaurants to be rated

            for rst in my_restaurants:  # for each restaurant

                # call the corresponding ratings function by name.
                rating, _cnt = getattr(self, segment_id)(_user=usr, _restaurant=rst)
                cnt += _cnt

                writer.writerow(list(usr.__dict__.values())
                                + list(rst.__dict__.values())
                                + [rating])

        fw.close()

        print('ones', cnt / total)
        return output_file
