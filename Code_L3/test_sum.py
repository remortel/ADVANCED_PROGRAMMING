# Example of a testing script
#
# Author: nick van Remortel
# Last revised 08/03/2022
#
# Sum() is a built-in function that works with
# any iterable structure
# Let's test it with a list and a tuple

def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

def test_sum_tuple():
    assert sum((1, 2, 2)) == 6, "Should be 6"

if __name__ == "__main__":
    test_sum()
    test_sum_tuple()
    print("Everything passed")
