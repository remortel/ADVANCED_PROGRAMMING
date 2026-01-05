# this piece of code demonstrates the Python type annotation feature
def greeting(name: str) -> str:
    return 'Hello ' + name

print(greeting(input('What is your name?\n')))
