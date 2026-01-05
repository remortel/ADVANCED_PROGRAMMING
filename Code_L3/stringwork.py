# This program illustrates working with formatted strings
# give preference to the new Python interpolated format strings
# Author: Nick van Remortel
# last revision: (dd/mm/yy=01/03/2021)
#
"""This is my doctstring."""
# First we make a list of tuples that contain a food item and a quatity
pantry = [
    ('avocados', 1.25),
    ('bananas', 2.5),
    ('cherries', 15),
]
for i, (item, count) in enumerate(pantry):
    # This is the old c-style string formatting
    old_style = '#%d: %-10s = %d' % (
        i + 1,
        item.title(),
        round(count),
    )
    # this uses the string.format() method
    new_style = '#{}: {:<10s} = {}'.format(
        i + 1,
        item.title(),
        round(count),
    )
    # this is the best and shortest way using python f-strings
    f_string = f'#{i+1}: {item.title():<10s} = {round(count)}'

    assert old_style == new_style == f_string
    print(f_string)
