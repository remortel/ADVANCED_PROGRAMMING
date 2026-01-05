# This is a test script that compares the use of
# instance methods
# class methods
# static methods
from Shapes import Circle
from Shapes import Point

# example calling a class method
print(Circle.unitcircle())
# example of calling a static method
radius=15.1234
print(f'the area of a circle with radius {radius:.2f} is: '
      f'{Circle.static_surf_circ(radius):.4f}')
# example of calling a static method
mypoint=Point(2.2, -1.5)
mycircle=Circle(mypoint, radius)
print(mycircle)

