"""
Geometry in 2D realization.
"""

import math
import mth
import numpy as np


class Vect:
    """
    Vector/point.
    """

    def __init__(self, x, y, id=0):
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
        self.id = id

    def __repr__(self):
        """
        Representation.

        Returns
        -------
        str
            String.
        """

        return f'p{self.id} ({self.x}, {self.y})'

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

    def angle_cos(self, v):
        """
        Cosine between vectors.

        Parameters
        ----------
        v : Vect
            Vector.

        Returns
        -------
        float
            Cosine between vectors.
        """

        # dot = ax * bx + ay * by = |a| * |b| * cos

        return self.dot(v) / (self.norm() * v.norm())

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


class Segm:
    """
    Edge.
    """

    def __init__(self, a, b):
        """
        Constructor.

        Parameters
        ----------
        a : Vect
            First point.
        b : Vect
            Second point.
        """

        # Prevent double segms.
        assert a.id < b.id

        self.a = a
        self.b = b

        # Equation coeffs.
        # a = p1, b = p2
        # (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1) = 0
        # (y2 - y1) * x + (x1 - x2) * y + (x1 * (y1 - y2) + y1 * (x2 - x1)) = 0
        x1, y1, x2, y2 = a.x, a.y, b.x, b.y
        self.ka = y2 - y1
        self.kb = x1 - x2
        self.kc = x1 * (y1 - y2) + y1 * (x2 - x1)
        # Then normalize it.
        m = self.ka**2 + self.kb**2
        self.ka /= m
        self.kb /= m
        self.kc /= m

    def __repr__(self):
        """
        Representation.

        Returns
        -------
        str
            String.
        """

        return f'({self.a.id} - {self.b.id})'

    def x_interval(self):
        """
        X interval.

        Returns
        -------
        (float, float)
            X interval.
        """

        return min(self.a.x, self.b.x), max(self.a.x, self.b.x)

    def y_interval(self):
        """
        Y interval.

        Returns
        -------
        (float, float)
            Y interval.
        """

        return min(self.a.y, self.b.y), max(self.a.y, self.b.y)

    def line_to_point_sdist(self, p):
        """
        Point position.

        Parameters
        ----------
        p : Point
            Point.

        Returns
        -------
        float
            Position - signed distance to line.
        """

        return self.ka * p.x + self.kb * p.y + self.kc

    def line_to_segm_sdist(self, s):
        """
        Position of s relative to self.
        If points are on opposite sides - it returns 0.0.
        In other case - position of the nearest.

        Parameters
        ----------
        s : Segm
            Other segm.

        Returns
        -------
        float
            Position.
        """

        ap = self.line_to_point_sdist(s.a)
        bp = self.line_to_point_sdist(s.b)

        if ap * bp <= 0.0:
            return 0.0
        elif (ap > 0.0) and (bp > 0.0):
            return min(ap, bp)
        elif (ap < 0.0) and (bp < 0.0):
            return max(ap, bp)
        else:
            raise Exception('internal error')

    def goodness(self, e):
        """
        Goodness with another segm.

        Parameters
        ----------
        e : Edge
            Another segm.

        Returns
        -------
        float
            Goodness coefficient.
        """

        # The same segment.
        if (self.a == e.a) and (self.b == e.b):
            return 0.0

        # Not intersect in boxes.
        if not mth.intervals_intersect(self.x_interval(), e.x_interval()):
            return 1.0
        if not mth.intervals_intersect(self.y_interval(), e.y_interval()):
            return 1.0

        # Not implemented.
        return -9.0


class GoodnessMatrix:
    """
    Goodness matrix.
    """

    def __init__(self, ss):
        """
        Constructor.

        Parameters
        ----------
        ss : [Segm]
            List of segments.
        """

        self.ss = ss
        self.l = len(ss)
        self.m = np.zeros((self.l, self.l))
        self.m -= 1.0

        # Init.
        for i in range(self.l):
            for j in range(self.l):
                if i <= j:
                    self.m[i][j] = self.ss[i].goodness(self.ss[j])
                else:
                    self.m[i][j] = self.m[j][i]

    def repr(self, i, j):
        """
        String representation of goodness matrix element.

        Parameters
        ----------
        i : int
            Row index.
        j : int
            Colunm index.

        Returns
        -------
        str
            Representation.
        """

        v = self.m[i][j]

        if v < 0.0:
            return '       -'
        else:
            c = '!' if i < j else ' '
            return f'{v:>8}{c}'

    def print(self):
        """
        Print.
        """

        print('Goodness matrix:')
        for i in range(self.l):
            s = ''.join([self.repr(i, j) for j in range(self.l)])
            print(f'  {s}')


def triangulation(ps):
    """
    Triangulation.

    Parameters
    ----------
    ps : [Point]
        List of points.

    Returns
    -------
    [(int, int, int)]
        List of indices for triangulation.
    """

    for i, p in enumerate(ps):
        p.id = i
    print(f'ps = {ps}')

    # Construct all segms.
    segms = [Segm(a, b) for a in ps for b in ps if a.id < b.id]
    print(f'segms = {segms}')

    # Goodness matrix.
    gm = GoodnessMatrix(segms)
    gm.print()

    # Check all relations between segments are defined.
    assert (gm.m >= 0.0).all()


if __name__ == '__main__':

    # Triangulation.
    print(triangulation([Vect(0.0, 0.0), Vect(2.0, 0.0), Vect(1.0, 1.0), Vect(1.0, 0.5)]))
