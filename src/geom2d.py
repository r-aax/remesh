"""
Geometry in 2D realization.
Vector/Point is a tuple (x, y).
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


def vect_direction(p0, p_from, p_to):
    """
    Direction of one vector relative another around p0.

    Parameters
    ----------
    p0 : Vect
        Zero vector (rotation point).
    p_from : Vect
        From vector.
    p_to : Vect
        To vector.

    Returns
    -------
    float
        Direction.
    """

    return (p_to - p0).cross(p_from - p0)


def is_ab_intersects_pq(a, b, p, q):
    """
    Check if ab segment intersects pq segment.

    Parameters
    ----------
    a : Vect
        First point of first segment.
    b : Vect
        Second point of first segment.
    p : Vect
        First poitn of second segment.
    q : Vect
        Second point of second segment.

    Returns
    -------
    bool
        True - if segments intersect,
        False - otherwise.
    """

    def on_segment(p1, p2, p_on):
        return (min(p1.x, p2.x) <= p_on.x <= max(p1.x, p2.x)) and (min(p1.y, p2.y) <= p_on.y <= max(p1.y, p2.y))

    d1 = vect_direction(p, q, a)
    d2 = vect_direction(p, q, b)
    d3 = vect_direction(a, b, p)
    d4 = vect_direction(a, b, q)

    if (d1 * d2 < 0.0) and (d3 * d4 < 0.0):
        return True
    elif (d1 == 0.0) and on_segment(p, q, a):
        return True
    elif (d2 == 0.0) and on_segment(p, q, b):
        return True
    elif (d3 == 0.0) and on_segment(a, b, p):
        return True
    elif (d4 == 0.0) and on_segment(a, b, q):
        return True
    else:
        return False


def is_ab_strong_intersects_pq(a, b, p, q):
    """
    Check if ab segment intersects pq segment (in strong sense).

    Parameters
    ----------
    a : Vect
        First point of first segment.
    b : Vect
        Second point of first segment.
    p : Vect
        First poitn of second segment.
    q : Vect
        Second point of second segment.

    Returns
    -------
    bool
        True - if segments strong intersect,
        False - otherwise.
    """

    return (a not in [p, q]) and (b not in [p, q]) and is_ab_intersects_pq(a, b, p, q)


def is_ab_strong_intersects_any_segment(a, b, ss):
    """
    Check if segment intersects any of ss segments.

    Parameters
    ----------
    a : Vect
        First segment point.
    b : Vect
        Second segment point.
    ss : [Vect]
        List of segments.

    Returns
    -------
    bool
        True - if segment strong intersect any,
        False - otherwise.
    """

    if not ss:
        return False

    return any(map(lambda s: is_ab_strong_intersects_pq(a, b, s[0], s[1]), ss))


def is_p_strong_in_abc(p, a, b, c):
    """
    Check if p in abc triangle.

    Parameters
    ----------
    p : Vect
        Point.
    a : Vect
        First triangle point.
    b : Vect
        Second triangle point.
    c : Vect
        Third triangle point.

    Returns
    -------
    bool
        True - if point in triangle,
        False - otherwise.
    """

    da = vect_direction(a, b, p)
    db = vect_direction(b, c, p)
    dc = vect_direction(c, a, p)

    if (da < 0.0) and (db < 0.0) and (dc < 0.0):
        return True
    elif (da > 0.0) and (db > 0.0) and (dc > 0.0):
        return True
    else:
        return False


def count_ps_in_abc(ps, a, b, c):
    """
    Count of points that lay in triangle.

    Parameters
    ----------
    ps : [Vect]
        List of points.
    a : Vect
        First triangle point.
    b : Vect
        Second triangle point.
    c : Vect
        Third triangle point.

    Returns
    -------
    int
        Count of points in triangle.
    """

    return sum(1 for p in ps if is_p_strong_in_abc(p, a, b, c))


def triangulation(ps, fixed_edges=[]):
    """
    Triangulation.

    Parameters
    ----------
    ps : [Vector]
        Points list.
    fixed_edges : [(int, int)]
        List of fixed edges.

    Returns
    -------
    [(ai, bi, ci)]
        List of indices of triangles.
    """

    # Count of points.
    n = len(ps)

    # All edges.
    all_edges = [(ps[ai].dist(ps[bi]), ai, bi) for ai in range(n) for bi in range(ai + 1, n)]
    all_edges.sort(key=lambda x: x[0])

    # All possible edges.
    pos_edges = []
    ss = []
    for fe in fixed_edges:
        ai = min(fe[0], fe[1])
        bi = max(fe[0], fe[1])
        pos_edges.append((ai, bi))
        ss.append((ps[ai], ps[bi]))
    for (_, ai, bi) in all_edges:
        if not is_ab_strong_intersects_any_segment(ps[ai], ps[bi], ss):
            pos_edges.append((ai, bi))
            ss.append((ps[ai], ps[bi]))

    # All triangles.
    all_tris = [(ai, bi, ci) for ai in range(n) for bi in range(ai + 1, n) for ci in range(bi + 1, n)
                if ((ai, bi) in pos_edges) and ((bi, ci) in pos_edges) and ((ai, ci) in pos_edges)]

    # Filter triangles.
    tris = [tri for tri in all_tris if count_ps_in_abc(ps, ps[tri[0]], ps[tri[1]], ps[tri[2]]) == 0]

    return tris


if __name__ == '__main__':

    # is_ab_intersects_pq
    assert is_ab_intersects_pq(Vect(0.0, 0.0), Vect(1.0, 1.0), Vect(1.0, 0.0), Vect(0.0, 1.0))
    assert not is_ab_intersects_pq(Vect(0.0, 0.0), Vect(1.0, 0.0), Vect(0.0, 1.0), Vect(1.0, 1.0))

    # is_p_strong_in_abc
    assert is_p_strong_in_abc(Vect(1.0, 0.5), Vect(0.0, 0.0), Vect(2.0, 0.0), Vect(1.0, 1.0))
    assert not is_p_strong_in_abc(Vect(1.0, -0.5), Vect(0.0, 0.0), Vect(2.0, 0.0), Vect(1.0, 1.0))

    # triangulation
    assert triangulation([Vect(0.0, 0.0), Vect(2.0, 0.0), Vect(1.0, 1.0)]) == [(0, 1, 2)]
    assert triangulation([Vect(0.0, 0.0), Vect(2.0, 0.0), Vect(1.0, 1.0), Vect(1.0, 0.5)]) \
           == [(0, 1, 3), (0, 2, 3), (1, 2, 3)]
    assert triangulation([Vect(0.0, 0.0), Vect(2.0, 0.0), Vect(1.0, 1.0),
                          Vect(1.0, 0.0), Vect(1.5, 0.5), Vect(0.5, 0.5)]) \
           == [(0, 3, 5), (1, 3, 4), (2, 3, 4), (2, 3, 5)]
    assert triangulation([Vect(0.0, 0.0), Vect(2.0, 0.0), Vect(1.0, 1.0),
                          Vect(1.0, 0.0), Vect(1.5, 0.5), Vect(0.5, 0.5)], fixed_edges=[(4, 5)]) \
           == [(0, 3, 5), (1, 3, 4), (2, 4, 5), (3, 4, 5)]
