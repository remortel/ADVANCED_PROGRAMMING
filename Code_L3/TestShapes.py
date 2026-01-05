# This is a test script that compares the use of
# instance methods
# class methods
# static methods
from Shapes import Circle
from Shapes import Zircle
from Shapes import Square
from Shapes import Point

# example calling a class method
print(Circle.unitcircle())
# example of calling a static method
radius=2.
print(f'the area of a circle with radius {radius:.2f} is: '
      f'{Circle.static_surf_circ(radius):.4f}')
# example of calling a static method
mypoint=Point(2.2, -1.5)
mycircle=Circle(mypoint, radius)
print(mycircle)

# Testing polymorphic class behavior
p1=Point(0.5, 0.7)
p2=Point(-0.2, 2.)
mycirc=Zircle(-0.2,3.0,2.0)
mysquare=Square(0.8, 2.7, 2.0)
print(mycirc)
for shape in (mycirc, mysquare):
      print(f'I made a {shape}, with surface '
            f'{shape.surface:10.4f} and circumference '
            f'{shape.circumference:10.4f}')