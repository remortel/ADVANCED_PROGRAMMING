# This code illustrates various forms of polymorfism
#
# Author: Nick van Remortel
# Last Revision: 08/03/2022

# Let's start with a simple example
# using the '+' operator

num1 = 1
num2 = 2
print(num1+num2)

str1 = "Hello "
str2 = "World!"
print(str1+str2)

# Now we show the polymorphic behavior of the
# len() function

print(len("program"))
print(len(["Pyhon", "Java", "C"]))
print(len({"name": "john", "address": "Nepal"}))

class Cat:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print(f"I am a cat. My name is {self.name}. I am {self.age} years old.")

    def make_sound(self):
        print("Meow")


class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print(f"I am a dog. My name is {self.name}. I am {self.age} years old.")

    def make_sound(self):
        print("Bark")


cat1 = Cat("Kitty", 2.5)
dog1 = Dog("Fluffy", 4)

for animal in (cat1, dog1):
    animal.make_sound()
    animal.info()
    animal.make_sound()



