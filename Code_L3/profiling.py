# This code demonstrates the use of a code profiler
# we will profile code that does an inefficient way of sorting

from random import randint
from cProfile import Profile
from pstats import Stats
from bisect import bisect_left
from tqdm import tqdm


def insertion_sort(data):
    result = []
    for value in tqdm(data, desc='Insertion sort'):
        insert_value(result, value)
    return result
def insertion_sort2(data):
    result = []
    for value in tqdm(data, desc='Insertion sort 2'):
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

# start profiling my code
#set up the profiler output file
filename = 'profile.prof'  # You can change this if needed
# Set up the profiler
profiler = Profile()
profiler.enable()
max_size = 10**5
#max_size = 10
data = [randint(0, max_size) for _ in range(max_size)]
sort1 = insertion_sort(data)
data = [randint(0, max_size) for _ in range(max_size)]
sort2 = insertion_sort2(data)


profiler.create_stats()
profiler.disable()
#stats.strip_dirs()
#stats.sort_stats('cumulative')
#stats.print_stats()
profiler.dump_stats(filename)

