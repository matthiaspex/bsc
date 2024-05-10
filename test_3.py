import jax
from jax import numpy as jnp


class Person():
    def __init__(self, gender):
        self.gender = gender
    
    def update_age(self, age):
        self.age = age

    def get_age(self):
        try:
            self.age
        except AttributeError:
            print("first call update_age method")

person = Person("man")
person.get_age()

person.update_age(22)
person.get_age()
