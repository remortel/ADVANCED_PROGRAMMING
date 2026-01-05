# This code example demonstrates catch-all unpacking
# Author : Nick van Remortel
# revised on 02/03/2022

# First make a list
car_ages = [0, 9, 4, 8, 7, 20, 19, 1, 6, 15]
# now sort it from high to low
car_ages_descending = sorted(car_ages, reverse=True)
# this ststement below will result in an error
# oldest, second_oldest = car_ages_descending
# let's do it this way then (not optimal)
oldest = car_ages_descending[0]
second_oldest = car_ages_descending[1]
others =car_ages_descending[2:]
print(oldest, second_oldest, others)
# now use catch-all unpacking
oldest, second_oldest, *others = car_ages_descending
print(oldest, second_oldest, others)
# we can use starred expressions in any position
oldest, *others, youngest = car_ages_descending
print(oldest, youngest, others)

