"""
Host the dataclasses which describe the users and restaurants
"""
from dataclasses import dataclass, fields


@dataclass
class Base:
    """Base project dataclass"""
    @classmethod
    def get_categorical_cols(cls):
        return [field.name for field in fields(cls) if field.type is list]


@dataclass
class Restaurant(Base):
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
class User(Base):
    gender: str
    age: int
    pcode: str
    favorite_cuisine: list
