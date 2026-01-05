# This code demonstrates the use of a code profiler
# we will profile code that does an inefficient way of sorting

from random import randint
from cProfile import Profile
from pstats import Stats
from bisect import bisect_left

def insertion_sort(data):
    result = []
    for value in data: 
        insert_value(result, value)
    return result
def insertion_sort2(data):
    result = []
    for value in data:
        insert_value2(result, value)
    return result

def insert_value(array, value):
    for i, existing in enumerate(array):
        if existing > value:
            array.insert(i, value)
            return
        array.append(value)

# let's try a more efficient sort
def insert_value2(array, value):
    i = bisect_left(array, value)
    array.insert(i, value)

max_size = 10**5
data = [randint(0, max_size) for _ in range(max_size)]
test = lambda: insertion_sort(data)

test2 = lambda: insertion_sort2(data)

# Set up the profiler
profiler = Profile()
profiler.runcall(test)
data = [randint(0, max_size) for _ in range(max_size)]
profiler.runcall(test2)

stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()

