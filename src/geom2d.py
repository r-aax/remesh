"""
Geometry in 2D realization.
"""

import math


class Vect:
    """
    Vector/point.
    """

    def __init__(self, x, y):
        """
        Constructor.

        Parameters
        ----------
        x : float
            X coord.
        y : float
            Y coord.
        """

        self.x = x
        self.y = y

    def __repr__(self):
        """
        Representation.

        Returns
        -------
        str
            String.
        """

        return f'({self.x}, {self.y})'

    def __add__(self, v):
        """
        Addition.

        Parameters
        ----------
        v : Vect
            Second operand.

        Returns
        -------
        Vect
            Result vector.
        """

        return Vect(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        """
        Subtraction.

        Parameters
        ----------
        v : Vect
            Second operand.

        Returns
        -------
        Vect
            Result vector.
        """

        return Vect(self.x - v.x, self.y - v.y)

    def __mul__(self, k):
        """
        Multiplication on float.

        Parameters
        ----------
        k : float
            Second operand.

        Returns
        -------
        Vect
            Result vector.
        """

        assert type(k) is float
        return Vect(self.x * k, self.y * k)

    def __truediv__(self, k):
        """
        Division on float.

        Parameters
        ----------
        k : float
            Second operand.

        Returns
        -------
        Vect
            Result vector.
        """

        return self * (1.0 / k)

    def dot(self, v):
        """
        Dot product.

        Parameters
        ----------
        v : Vect
            Second operand.

        Returns
        -------
        float
            Result.
        """

        return self.x * v.x + self.y * v.y

    def cross(self, v):
        """
        Cross product.

        Parameters
        ----------
        v : Vect
            Second operand.

        Returns
        -------
        float
            Result.
        """

        return self.x * v.y - v.x * self.y

    def norm2(self):
        """
        Square of norm.

        Returns
        -------
        float
            Square of norm.
        """

        return self.dot(self)

    def norm(self):
        """
        Norm.

        Returns
        -------
        float
            Norm.
        """

        return math.sqrt(self.norm2())

    def dist(self, v):
        """
        Distance between points.

        Parameters
        ----------
        v : Vect
            Second point.

        Returns
        -------
        float
            Distance.
        """

        return (self - v).norm()


if __name__ == '__main__':
    pass
