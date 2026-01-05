# Demo of basic Module with two classes:
# Point: Euclidian 2D coordinates
# Circle: A class containing a Point member (center coordinates)
#         and a non-negative radius
# Author: Nick van Remortel
# last revision: (dd/mm/yy=01/03/2021)
# 
from __future__ import annotations
import math
from collections import namedtuple




class Point(object):
    """An (x,y) coordinate in 2D Euclidian space"""

    def __init__(self, cx: float =0., cy: float =0.) -> None:
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
        return "({0._x}, {0._y})".format(self)

    def __repr__(self) -> str:
        """ Represents a point as a string. """
        return "({0._x}, {0._y})".format(self)

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
        return "({0._center}, {0._radius})".format(self)

    @property
    def radius(self) -> float:
        """ Return the circle radius """
        return self._radius

    @radius.setter
    def radius(self, radius):
        """ Set the radius."""
        if radius < 0:
            raise ValueError('radius of a circle must have positive value')
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
        return "(super().__str__, {0._radius})".format(self)

    @property
    def radius(self) ->float:
        """ Return the circle radius """
        return self._radius

    @radius.setter
    def radius(self, radius: float):
       """ Set the radius."""
       if radius < 0:
           raise ValueError('radius of a circle must have positive value')
       self._radius = radius

    def surface(self) -> float:
        """ returns the surface area of the circle """
        return (math.pi)*(self.radius)*(self.radius)

    def circumference(self) -> float:
        """ returns the circumference of teh circle """
        return 2.*(math.pi)*(self.radius)
   
if __name__ == "__main__":
    import sys

    p1=Point(0.5, 0.7)
    mycirc=Circle(p1,2.0)
    print("I made a circle:", mycirc)
    


