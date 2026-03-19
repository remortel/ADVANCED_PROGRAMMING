# This code demonstrates the use of a profile function decorator
# it uses the functools wrap() function to make the wrapped
# functions transparent to various enquiries, such as help()

from random import randint
import cProfile
import pstats
from bisect import bisect_left
from tqdm import tqdm
from io import StringIO
from functools import wraps
import logging
from memory_profiler import profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
### here is define my function profiling decorator
## Note that decorators can take arguments as well
def myprofile(output_file=None, sort_by='cumulative', limit=20):
    """Decorator to profile a function

    Args:
        output_file: Optional file path to save binary profile data
        sort_by: Sort key ('cumulative', 'tottime', 'calls')
        limit: Number of functions to display in output
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create fresh profiler for each call
            profiler = cProfile.Profile()
            profiler.enable()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()

                # Format profile output to string
                stream = StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats(sort_by)
                stats.print_stats(limit)

                # Optionally save to file for external analysis tools
                if output_file:
                    profiler.dump_stats(output_file)

                # Log the profile results
                logger.info(f"Profile for {func.__name__}:\n{stream.getvalue()}")

        return wrapper
    return decorator
@profile
@myprofile(output_file='inser.prof', sort_by='cumtime', limit=10)
def insertion_sort(data):
    result = []
    for value in data:
        insert_value(result, value)
    return result
@profile
@myprofile(output_file='inser2.prof', sort_by='cumtime', limit=10)
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
#max_size = 10
data = [randint(0, max_size) for _ in range(max_size)]
sort1 = insertion_sort(data)
data = [randint(0, max_size) for _ in range(max_size)]
sort2 = insertion_sort2(data)


