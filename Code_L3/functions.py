# This piece of code demonstrates the advanced use of functions
# Author Nick van Remortel
# Revision date: 02/03/2022

import math
# a function can have multiple names
def succ(x):
    return x + 1
successor = succ
print(successor(10))
# we can even delete the original name
# and keep using the new one
del succ
print(successor(10))
# functions passed as arguments

def foo(func):
    print("The function " + func.__name__ + " was passed to foo")
    res = 0
    for x in [1, 2, 2.5]:
        res += func(x)
    return res
print(foo(math.sin))
print(foo(math.cos))
# A function that returns two values
def get_stats(numbers):
    minimum = min(numbers)
    maximum = max(numbers)
    return minimum, maximum
# A function that returns a list with unknown length
def get_avg_ratio(numbers):
    average = sum(numbers) / len(numbers)
    scaled = [x / average for x in numbers]
    scaled.sort(reverse=True)
    return scaled
# A function with variable positional arguments
def log(message, *values):
    if not values:
        print(message)
    else:
        values_str = ', '.join(str(x) for x in values)
        print(f'{message}: {values_str}')

# First define a list of numbers
lengths = [63, 73, 72, 60, 67, 66, 71, 61, 72, 70]

# unpacking the output tuple given by get_stats
minimum, maximum = get_stats(lengths)
print(f'Min:  {minimum}, Max: {maximum}')

# unpacking the output of get_avg_ratio by
# catch-all unpacking
longest, *middle, shortest = get_avg_ratio(lengths)
print(f'Longest:  {longest:4.0%}')
print(f'Shortest:  {shortest:4.0%}')

# using variable position argument function log()
log('My lengths are', lengths)
log('No lengths')

# The following piece of code illustrates
# the use of function definitions inside other
# functions and the problems with variable scope
# the function below uses a helper function to make
# a prioritized sort for a subgroup of elements in a list
numbers = [8, 3, 1, 2, 5, 4, 7, 6]
group = {2, 3, 5, 7} # we use a set for fast lookup

def sort_priority(numbers, group):
    found = False
    def helper(x):
        # nonlocal found # Investigate the effect of this statement
        if x in group:
            found = True # this variable assignment enforces local scope
            return (0,x)
        return (1, x)
    numbers.sort(key=helper)
    return found

# after calling sort_priority
# the numbers will be properly sorted but
# the found variale will always be False
# comment out the line that says nonlocal found
# and see the difference
found = sort_priority(numbers, group)
print('Found:', found)
print(numbers)

# Combining some aspects together
# a function that generates polynomials
def polynomial_creator(*coeffs):
    """ coefficients are in the form a_n, a_n_1, ... a_1, a_0
    """
    def polynomial(x):
        res = coeffs[0]
        for i in range(1, len(coeffs)):
            res = res * x + coeffs[i]
        return res
    return polynomial
p1 = polynomial_creator(4)
p2 = polynomial_creator(2, 4)
p3 = polynomial_creator(1, 8, -1, 3, 2)
p4 = polynomial_creator(-1, 2, 1)
print('Calculating polynomials')
for x in range(-2, 2, 1):
    print(x, p1(x), p2(x), p3(x), p4(x))


# Below we define a simple function that
# returns the remainder of an integer division
# This function takes two arguments that can
# be passed via their position, or via their keyword
def remainder(number, divisor):
    return number % divisor

# calling the function with positional arguments
print(f'computing the remainder of 20/7: {remainder(20,7)}')
# calling the function with keyword arguments
print(f'computing the remainder of 20/7: {remainder(number=20, divisor=7)}')
# when using keyword arguments, you can change the position and still
# get the right result
print(f'computing the remainder of 20/7: {remainder(divisor=7, number=20)}')
# you can also mix positional and keyword arguments
# but positional arguments must always come first
print(f'computing the remainder of 20/7: {remainder(20, divisor=7)}')
# print(f'computing the remainder of 20/7: {remainder(number=20, 7)}')
# we can provide keyword arguments by making a dictionary of the
# keyword-value pairs first
my_kwargs = {
    'number': 20,
    'divisor': 7,
}
# now pass the dictionary to the function as keyword arguments
# with the ** operator
print(f'computing the remainder of 20/7: {remainder(**my_kwargs)}')
# you can also define functions that take an arbitrary amount of
# keyword arguments using the catch-all mechanism
def print_parameters(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} = {value}')

# the three parameters below will be collected
# into a dictionary kwargs via the catch-all
# parameter mechanism and the ** operator
print_parameters(alpha=1.5, beta=9, gamma=4)

# defining a function that enforces the use of
# positional arguments before the /
# and enforces keyword arguments after the *
# anything in between the / and * can be
# either positional or keyword arguments
def safe_division(numerator, denominator, /,
                  ndigits=10, *,
                  ignore_overflow=False,
                  ignore_zero_division=False):
    try:
        fraction = numerator / denominator
        return round(fraction, ndigits)
    except OverflowError:
        if ignore_overflow:
            return 0
        else:
            raise
    except ZeroDivisionError:
        if ignore_zero_division:
            return float('inf')
        else:
            raise

# now use the safe_division function
# with the right sort of arguments
print(safe_division(22, 7)) # this works, last 3 arguments were optional
print(safe_division(22, 7, 5)) # also works
print(safe_division(22, 7, ndigits=4)) # also works
print(safe_division(22, 0, 6, ignore_zero_division=True)) # also works
#print(safe_division(22, 0, 6, True, True)) # does not work

# demonstration of the use of function decorators
# note that you can now re-use the
# argument_test_natural_number decorator
# with any function you like

def argument_test_natural_number(f):
    def helper(x):
        if type(x) == int and x > 0:
            return f(x)
        else:
            raise Exception("Argument is not an integer")
    return helper
@argument_test_natural_number
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)

for i in range(1,10):
	print(i, factorial(i))

# uncomment the line beow to generate an exception
# via the  argument_test_natural_number decorator
# print(factorial(-1))

# you can even use multiple decorators
# the outside decorator is applied first
def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper

@call_counter
@argument_test_natural_number
def factorial2(n):
    if n == 1:
        return 1
    else:
        return n * factorial2(n-1)

for i in range(1,10):
	print(i, factorial2(i), f'called {factorial2.calls:d} times')







