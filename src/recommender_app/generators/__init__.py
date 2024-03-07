"""
Host the dataclasses which describe the users and restaurants
"""
from dataclasses import dataclass


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


@dataclass
class User:
    gender: str
    age: int
    pcode: str
    favorite_cuisine: list
