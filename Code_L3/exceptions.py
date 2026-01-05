# This code illustrates exception handling
# at various degrees of detail
# Author Nick van Remortel
# last revised 07/03/2022

# First make a function that divides two numbers
def divider(numerator, denominator):
    return int(numerator)/int(denominator)

# Execute this for the problematic cause
# division by zero

#print(divider(5, 0))
# try to do other decent things
# will this execute?
print(divider(5, 10))

# now put the statemnt in a try block

try:
    print(divider(5, 0))
except:
    print('something went wrong')
else:
    print('all going well')

print(divider(5, 10))

# you can be more specific and specify the
# possible exceptions

try:
    print(divider(5, 0))
except ZeroDivisionError:
    print('We know you are trying to divide by zero!')
else:
    print('all going well')

print(divider(5, 10))

# let's include some other weird possibilities
# now we also include a finally cluase
try:
    print(divider('hello', 5))
except ZeroDivisionError:
    print('We know you are trying to divide by zero!')
except ValueError:
    print('You are not using the right type')
else:
    print('all going well')
finally:
    print('this statement will always execute')

print(divider(5, 10))

# below we will raise our own exception
def yourname(name):
    if name != 'Nick':
        raise NameError('Not the right name')
    else:
        return name

try:
    print(yourname('Jack'))
except NameError:
    print('We know you are using the wrong name!')
else:
    print('all going well')
finally:
    print('this statement will always execute')
    


