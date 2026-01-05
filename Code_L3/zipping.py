# This program illustrates the use of the zip() function
# to iterate in parallel over two  lists of the same length

# First we create a list of 3 names
names = ['Cecilia', 'Lise', 'Marie']
# let's count the number of characters in each name of this list
# this can be done in various ways
# first way: Loops
cts = []
for i in range(len(names)):
    cts.append(len(names[i]))

print(cts)
# second way, with maps
def countchar(name):
    return len(name)
ctts = map(countchar, names)
print(list(ctts))

# Third way (and best way):
# list comprehensions
counts = [len(n) for n in names]
print(counts)

# now we will iterate in parallel over the names and counts list
# to find the longest name
# this is again possible in 3 ways
# First way: loops
longest_name = None
max_count = 0
for i in range(len(names)):
    count = counts[i]
    if count > max_count:
        longest_name = names[i]
        max_count = count

print(longest_name)

# second way, using enumerate
longest_name = None
max_count = 0
for i, name in enumerate(names):
    count = counts[i]
    if count > max_count:
        longest_name = name
        max_count = count

print(longest_name)

# third way, using zip
longest_name = None
max_count = 0
for name, count in zip(names, counts):
    if count > max_count:
        longest_name = name
        max_count = count

print(longest_name)

