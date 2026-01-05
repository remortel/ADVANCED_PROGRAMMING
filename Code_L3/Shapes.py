# Demo of basic Module with two classes:
# Point: Euclidian 2D coordinates
# Circle: A class containing a Point member (center coordinates)
#         and a non-negative radius
# Author: Nick van Remortel
# last revision: (dd/mm/yy=08/03/2021)
# 
from __future__ import annotations
import math
from collections import namedtuple

# Define a decorator that can wrap
# Any class method
def argument_test_natural_number(f):
    def helper(self, x):
        if (type(x) == int or type(x) == float) and x > 0:
            return f(self, x)
        else:
            raise Exception("Argument is not an integer or float larger than 0")

    return helper

class Point(object):
    """An (x,y) coordinate in 2D Euclidian space"""

    def __init__(self, cx: float =0., cy: float =0.):
        """ Initialisation of Euclidian point attributes.
        :type cx: float
        :type cy: float
        """
        self.x = cx
        self.y = cy

    @property
    def x(self):
        """ Return the x coordinate """
        return self._x

    @x.setter
    def x(self, cx: float):
        """ Set the x coordinate."""
        self._x = cx

    @property
    def y(self) -> float:
        """ Return the x coordinate """
        return self._y

    @y.setter
    def y(self, cy: float):
        """ Set the y coordinate."""
        self._y = cy

    def __str__(self) -> str:
        """ Converts a point into a string. """
        return f'({self._x}, {self._y})'

    def __repr__(self) -> str:
        """ Represents a point as a string. """
        return f'({self_x}, {self._y})'

    def __add__(self, other) -> Point:
        """ Adds two points """
        return Point(self._x + other._x, self._y + other._y)

    def __sub__(self, other) -> Point:
        """ Adds two points """
        return Point(self._x - other._x, self._y - other._y)

    def polar(self) -> namedtuple:
        """Computes the polar coordinates out of cartesian coordinates"""
        pol=namedtuple('polar', 'r theta')
        cor=pol(math.hypot(self._x,self._y), math.atan2(self._y,self._x))
        return cor 


class Circle(object):
    """A simple 2d circle in the Eucledian plane"""

    def __init__(self, center: Point =Point(), radius: float =0.):
        """ Initialisation of circle attributes. """

        self.radius = radius
        self.center = center

    def __str__(self):
        """ Converts a point into a string. """
        return f'({self._center}, {self._radius})'

    @property
    def radius(self) -> float:
        """ Return the circle radius """
        return self._radius


    @radius.setter
    @argument_test_natural_number
    def radius(self, radius):

        self._radius = radius

    @property
    def center(self) -> Point:
        """ Return the circle center """
        return self._center

    @center.setter
    def center(self, center: Point =Point()):
        """ Set the center coordinates."""
        self._center = center

    def surface(self) -> float:
        """ return the surface area"""
        return (math.pi)*(self._radius)*(self._radius)

    def circumference(self) -> float:
        """return the circumference"""
        return 2.*(math.pi)*(self._radius)

    @classmethod
    def unitcircle(cls) -> Circle:
        return cls(Point(),1.)

    @staticmethod
    def static_surf_circ(r: float =0.) -> float:
        return (math.pi) * (r) * (r)

class Zircle(Point):
    """A simple 2d circle that inherits from Point"""


    def __init__(self, c1: float=0., c2: float=0., radius: float=0.):
        """ Initialisation of circle attributes. """
        super().__init__(c1,c2)
        self.radius = radius  #validate via property

    def __str__(self) ->str:
        """ Converts a point into a string. """
        return f'Circle ({super().__str__()}, {self._radius})'

    @property
    def radius(self) ->float:
        """ Return the circle radius """
        return self._radius

    @radius.setter
    @argument_test_natural_number
    def radius(self, radius: float):
       """ Set the radius."""
       self._radius = radius
    @property
    def surface(self) -> float:
        """ returns the surface area of the circle """
        return (math.pi)*(self.radius)*(self.radius)
    @property
    def circumference(self) -> float:
        """ returns the circumference of teh circle """
        return 2.*(math.pi)*(self.radius)

class Square(Point):
    """A simple 2d circle that inherits from Point"""

    def __init__(self, c1: float=0., c2: float=0., height: float=0.):
        """ Initialisation of circle attributes. """
        super().__init__(c1,c2)
        self.height = height  #validate via property

    def __str__(self) ->str:
        """ Converts a point into a string. """
        return f'Square ({super().__str__()}, {self._height})'

    @property
    def height(self) ->float:
        """ Return the circle radius """
        return self._height

    @height.setter
    @argument_test_natural_number
    def height(self, height: float=0.):
       """ Set the radius."""
       self._height = height
    @property
    def surface(self) -> float:
        """ returns the surface area of the circle """
        return self._height*self._height
    @property
    def circumference(self) -> float:
        """ returns the circumference of teh circle """
        return 4.*self._height
   
if __name__ == "__main__":
    import sys

    p1=Point(0.5, 0.7)
    p2=Point(-0.2, 2.)
    mycirc=Zircle(-2.2, 5.4,2.0)
    num = mycirc.surface
    print(num)
    mysquare=Square(4.2, -9.1, 2.0)
    for shape in (mycirc, mysquare):
        print(f'I made a {str(shape)}, with surface {shape.surface:2.2f} and circumference {shape.circumference:2.2f}')
    


