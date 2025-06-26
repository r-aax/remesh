import numpy as np
import numpy.linalg as la
import itertools

from lxml.etree import ElementClassLookup

import mth
import random

# --------------------------------------------------------------------------------------------------

def points_dist(a, b):
    """
    Distance betweeen points.

    Parameters
    ----------
    a : Point
        First point.
    b : Point
        Second point.

    Returns
    -------
    float
        Distance.
    """

    return la.norm(a - b)

# --------------------------------------------------------------------------------------------------

def points_sarea(a, b, c):
    """
    Signed area based on points.

    Parameters
    ----------
    a : Point
        First point.
    b : Point
        Second point.
    c : Point
        Third point.

    Returns
    -------
    Vector
        Signed area.
    """

    return np.cross(b - a, c - a)

# --------------------------------------------------------------------------------------------------

def points_area(a, b, c):
    """
    Area between points.

    Parameters
    ----------
    a : Point
        Point A.
    b : Point
        Point B.
    c : Point
        Point C.

    Returns
    -------
    float
        Area between points.
    """

    return 0.5 * la.norm(points_sarea(a, b, c))

# --------------------------------------------------------------------------------------------------

def is_ab_intersects_pq(a, b, p, q):
    """
    Check is ab segments intersects pq.

    Parameters
    ----------
    a : Point
        First point of first segment.
    b : Point
        Second point of first segment.
    p : Point
        First point of second segment.
    q : Point
        Second point of second segment.

    Returns
    -------
    True - if segments intersect,
    False - otherwise.
    """

    # If segments intersect then:
    # 1) a and b lay in different sides of pq
    # 2) p and q lay in different sides of ab

    s1 = np.dot(points_sarea(p, q, a), points_sarea(p, q, b))
    s2 = np.dot(points_sarea(a, b, p), points_sarea(a, b, q))

    return (s1 < 0.0) and (s2 < 0.0)

# --------------------------------------------------------------------------------------------------

def if_ab_intersects_any_segment(a, b, ss):
    """
    Check if ab segment intersects any segment from the list.

    Parameters
    ----------
    a : Point
        First point.
    b : Point
        Second point.
    ss : [(Point, Point)]
        List of segments.

    Returns
    -------
    True - if there is intersection,
    False - otherwise.
    """

    if not ss:
        return False

    return any(map(lambda s: is_ab_intersects_pq(a, b, s[0], s[1]), ss))

# --------------------------------------------------------------------------------------------------

def is_points_near(a, b):
    """
    Check if points are near.

    Parameters
    ----------
    a : Point
        First point.
    b : Point
        Second point.

    Returns
    -------
    True - if points are near,
    False - if points are not near.
    """

    return points_dist(a, b) < mth.EPS

# --------------------------------------------------------------------------------------------------

def delete_near_points(ps):
    """
    Delete all near points.

    Parameters
    ----------
    ps : [Point]
        Points list.

    Returns
    -------
    [Point]
        List of points after delete near points.
    """

    l = len(ps)
    delete_flags = [False] * l

    for i in range(l):
        for j in range(i + 1, l):
            if is_points_near(ps[i], ps[j]):
                delete_flags[j] = True

    return [ps[i] for i in range(l) if not delete_flags[i]]

# --------------------------------------------------------------------------------------------------

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Get rotation matrix from vec1 to vec2.
    Source:
        https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

    Parameters
    ----------
    vec1 : Vector
        Vector from.
    vec2 : Vector
        Vector to.

    Returns
    -------
    Matrix
        Rotation matrix.
    """

    a, b = (vec1 / la.norm(vec1)).reshape(3), (vec2 / la.norm(vec2)).reshape(3)
    v = np.cross(a, b)

    if any(v):
        c = np.dot(a, b)
        s = la.norm(v)
        kmat = np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1.0 - c) / s**2)
    else:
        return np.eye(3)

# --------------------------------------------------------------------------------------------------

class Triangle:
    """
    Triangle - locus of points P:
               P = A + (B - A) * beta + (C - A) * gamma
               beta >= 0
               gamma >= 0
               beta + gamme <= 1
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, a, b, c, back_ref=None):
        """
        Constructor.

        Parameters
        ----------
        a : Point
            First point.
        b : Point
            Second point.
        c : Point
            Third point.
        back_ref : object
            Back reference.
        """

        self.points = [a, b, c]

        # Back reference for linking with face.
        self.back_ref = back_ref

    # ----------------------------------------------------------------------------------------------

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Triangle ({self.points[0]}, {self.points[1]}, {self.points[2]})'

    # ----------------------------------------------------------------------------------------------

    def center(self):
        """
        Get triangle center.

        Returns
        -------
        Point
            Center point.
        """

        return (self.points[0] + self.points[1] + self.points[2]) / 3.0

    # ----------------------------------------------------------------------------------------------

    def random_point(self):
        """
        Get random point in triangle.

        Returns
        -------
        Point
            Random point.
        """

        a, b, c = self.points[0], self.points[1], self.points[2]
        ra, rb, rc = random.random(), random.random(), random.random()

        return (ra * a + rb * b + rc * c) / (ra + rb + rc)

    # ----------------------------------------------------------------------------------------------

    def area(self):
        """
        Area.

        Returns
        -------
        float
            Area.
        """

        return points_area(self.points[0], self.points[1], self.points[2])

    # ----------------------------------------------------------------------------------------------

    def normal(self):
        """
        Get normal.

        Returns
        -------
        Vector
            Normal.
        """

        n = np.cross(self.points[1] - self.points[0], self.points[2] - self.points[0])
        n = n / la.norm(n)

        return n

    # ----------------------------------------------------------------------------------------------

    def min_height(self):
        """
        Min height.

        Returns
        -------
        float
            Min height.
        """

        a, b, c = self.points[0], self.points[1], self.points[2]
        bs = max([points_dist(a, b), points_dist(b, c), points_dist(a, c)])

        return self.area() / (0.5 * bs)

    # ----------------------------------------------------------------------------------------------

    def is_thin(self, local_eps=None):
        """
        Check if triangle thin.

        Returns
        -------
        True - if it is this,
        False - otherwise.
        """

        return self.min_height() < mth.EPS if local_eps is None else self.min_height() < local_eps

    # ----------------------------------------------------------------------------------------------

    def areas_difference(self, p):
        """
        Measure for p in triagle.
        |S(a, b, c) - S(a, b, p) - S(b, c, p) - S(a, c, p)|

        Parameters
        ----------
        p : Point
            Point for check.

        Returns
        -------
        float
            Areas difference.
        """

        a, b, c = self.points[0], self.points[1], self.points[2]

        return abs(self.area() - points_area(a, b, p) - points_area(b, c, p) - points_area(a, c, p))

    # ----------------------------------------------------------------------------------------------

    def has_common_points_with(self, t):
        """
        Check if triangle has common points with another triangle.

        Parameters
        ----------
        t : Triangle
            Another triangle.

        Returns
        -------
        True - if triangles have common points,
        False - if triangles have no common points.
        """

        for p1 in self.points:
            for p2 in t.points:
                if p1.data == p2.data:
                    # Check for equality as objects.
                    return True

        return False

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def sorting_by_the_selected_axis(triangles_for_sorting, axis):
        """
        Sorting triangles along given axis.

        Parameters
        ----------
        triangles_for_sorting : [Triangles]
            List of triangles.
        axis : int
            Axis.

        Returns
        -------
        [Triangle]
            Sorted list.
        """

        triangles_for_sorting.sort(key=lambda tri: tri.center()[axis])

        return triangles_for_sorting

    # ----------------------------------------------------------------------------------------------

    def is_intersect_with_segment(self, p, q):
        """
        Check if triangle intersects segment [p, q].
        Parameters
        ----------
        p : Point
            Segment begin.
        q : Point
            Segment end.

        Returns
        -------
        True - if there is intersection,
        False - if there is no intersection.
        """

        def is_in_tri(bet, gam):
            return (bet >= 0.0) and (gam >= 0.0) and (bet + gam <= 1.0)

        def is_in_seg(phi):
            return 0.0 <= phi <= 1.0

        a, b, c = self.points[0], self.points[1], self.points[2]

        #
        # Point (x, y, z) inside triangle can be represented as
        # x = x_a + (x_b - x_a) * bet + (x_c - x_a) * gam
        # y = y_a + (y_b - y_a) * bet + (y_c - y_a) * gam
        # z = z_a + (z_b - z_a) * bet + (z_c - z_a) * gam
        #    where (x_a, y_a, z_a) - coordinates of point a,
        #          (x_b, y_b, z_b) - coordinates of point b,
        #          (x_c, y_c, z_c) - coordinates of point c,
        #          bet >= 0,
        #          gam >= 0,
        #          bet + gam <= 1.
        # ...
        # x = x_a + x_ba * bet + x_ca * gam
        # y = y_a + y_ba * bet + y_ca * gam
        # z = z_a + z_ba * bet + z_ca * gam
        #
        # Point (x, y, z) on segment can be represented as
        # x = x_p + (x_q - x_p) * phi
        # y = y_p + (y_q - y_p) * phi
        # x = z_p + (z_q - z_p) * phi
        #   where (x_p, y_p, z_p) - coordinates of point p,
        #         (x_q, y_q, z_q) - coordinates of point q,
        #         0 <= phi <= 1.
        # ...
        # x = x_p + x_qp * phi
        # y = y_p + y_qp * phi
        # x = z_p + z_qp * phi
        #
        # So to find intersection we have to solve system
        # x_a + x_ba * bet + x_ca * gam = x_p + x_qp * phi
        # y_a + y_ba * bet + y_ca * gam = y_p + y_qp * phi
        # z_a + z_ba * bet + z_ca * gam = z_p + z_qp * phi
        # ...
        # x_ba * bet + x_ca * gam + (-x_qp) * phi = x_p - x_a
        # y_ba * bet + y_ca * gam + (-y_qp) * phi = y_p - y_a
        # z_ba * bet + z_ca * gam + (-z_qp) * phi = z_p - z_a
        # ...
        # x_ba * bet + x_ca * gam + x_pq * phi = x_pa
        # y_ba * bet + y_ca * gam + y_pq * phi = y_pa
        # z_ba * bet + z_ca * gam + z_pq * phi = z_pa
        #
        # Matrix view of this system can be written in the following view
        # [x_ba x_ca x_pq]     [bet]     [x_pa]
        # [y_ba y_ca y_pq]  X  [gam]  =  [y_pa]
        # [z_ba z_ca z_pq]     [phi]     [z_pa]
        #

        ba, ca, pq, pa = b - a, c - a, p - q, p - a

        m = np.array([ba, ca, pq])
        m = np.transpose(m)
        d = la.det(m)

        if d != 0.0:
            im = la.inv(m)
            [bet, gam, phi] = im.dot(pa)

            return is_in_tri(bet, gam) and (is_in_seg(phi))
        else:
            # TODO.
            # If det = 0.0 segment may lay in triangle plane.
            return False

    # ----------------------------------------------------------------------------------------------

    def find_intersection_with_segment(self, p, q):
        """
        Find intersection point with segment [p, q].
        Parameters
        ----------
        p : Point
            Segment begin.
        q : Point
            Segment end.

        Returns
        -------
        [Point]
            List of intersection points.
        """

        def is_in_tri(bet, gam):
            return (bet >= 0.0) and (gam >= 0.0) and (bet + gam <= 1.0)

        def is_in_seg(phi):
            return 0.0 <= phi <= 1.0

        a, b, c = self.points[0], self.points[1], self.points[2]

        #
        # Point (x, y, z) inside triangle can be represented as
        # x = x_a + (x_b - x_a) * bet + (x_c - x_a) * gam
        # y = y_a + (y_b - y_a) * bet + (y_c - y_a) * gam
        # z = z_a + (z_b - z_a) * bet + (z_c - z_a) * gam
        #    where (x_a, y_a, z_a) - coordinates of point a,
        #          (x_b, y_b, z_b) - coordinates of point b,
        #          (x_c, y_c, z_c) - coordinates of point c,
        #          bet >= 0,
        #          gam >= 0,
        #          bet + gam <= 1.
        # ...
        # x = x_a + x_ba * bet + x_ca * gam
        # y = y_a + y_ba * bet + y_ca * gam
        # z = z_a + z_ba * bet + z_ca * gam
        #
        # Point (x, y, z) on segment can be represented as
        # x = x_p + (x_q - x_p) * phi
        # y = y_p + (y_q - y_p) * phi
        # x = z_p + (z_q - z_p) * phi
        #   where (x_p, y_p, z_p) - coordinates of point p,
        #         (x_q, y_q, z_q) - coordinates of point q,
        #         0 <= phi <= 1.
        # ...
        # x = x_p + x_qp * phi
        # y = y_p + y_qp * phi
        # x = z_p + z_qp * phi
        #
        # So to find intersection we have to solve system
        # x_a + x_ba * bet + x_ca * gam = x_p + x_qp * phi
        # y_a + y_ba * bet + y_ca * gam = y_p + y_qp * phi
        # z_a + z_ba * bet + z_ca * gam = z_p + z_qp * phi
        # ...
        # x_ba * bet + x_ca * gam + (-x_qp) * phi = x_p - x_a
        # y_ba * bet + y_ca * gam + (-y_qp) * phi = y_p - y_a
        # z_ba * bet + z_ca * gam + (-z_qp) * phi = z_p - z_a
        # ...
        # x_ba * bet + x_ca * gam + x_pq * phi = x_pa
        # y_ba * bet + y_ca * gam + y_pq * phi = y_pa
        # z_ba * bet + z_ca * gam + z_pq * phi = z_pa
        #
        # Matrix view of this system can be written in the following view
        # [x_ba x_ca x_pq]     [bet]     [x_pa]
        # [y_ba y_ca y_pq]  X  [gam]  =  [y_pa]
        # [z_ba z_ca z_pq]     [phi]     [z_pa]
        #

        ba, ca, pq, pa = b - a, c - a, p - q, p - a

        m = np.array([ba, ca, pq])
        m = np.transpose(m)
        d = la.det(m)

        if d != 0.0:
            im = la.inv(m)
            [bet, gam, phi] = im.dot(pa)

            if is_in_tri(bet, gam) and (is_in_seg(phi)):
                return [p + phi * (q - p)]
            else:
                return []
        else:
            # TODO.
            # If det = 0.0 segment may lay in triangle plane.
            return []

    # ----------------------------------------------------------------------------------------------

    def is_intersect_with_triangle(self, t):
        """
        Check if triangle intersects another triangle.

        Parameters
        ----------
        t : Triangle
            Triangle for check intersection.

        Returns
        -------
        True - if there is intersection,
        False - if there is no intersection.
        """
        # There is intersection if any side of one triangle
        # intersects another triangle.
        return self.is_intersect_with_segment(t.points[0], t.points[1]) \
               or self.is_intersect_with_segment(t.points[1], t.points[2]) \
               or self.is_intersect_with_segment(t.points[2], t.points[0]) \
               or t.is_intersect_with_segment(self.points[0], self.points[1]) \
               or t.is_intersect_with_segment(self.points[1], self.points[2]) \
               or t.is_intersect_with_segment(self.points[2], self.points[0])

    # ----------------------------------------------------------------------------------------------

    def find_intersection_with_triangle(self, t):
        """
        Find intersection with another triangle.

        Parameters
        ----------
        t : Triangle
            Triangle for check intersection.

        Returns
        -------
        [Point]
            List of intersection points.
        """

        points = self.find_intersection_with_segment(t.points[0], t.points[1]) \
                 + self.find_intersection_with_segment(t.points[1], t.points[2]) \
                 + self.find_intersection_with_segment(t.points[2], t.points[0]) \
                 + t.find_intersection_with_segment(self.points[0], self.points[1]) \
                 + t.find_intersection_with_segment(self.points[1], self.points[2]) \
                 + t.find_intersection_with_segment(self.points[2], self.points[0])
        points = delete_near_points(points)

        assert len(points) <= 2

        return points

    # ----------------------------------------------------------------------------------------------

    def is_intersect_with_box(self, box):
        """
        Check intersection with box.

        Parameters
        ----------
        b : Box
            Box.

        Returns
        -------
        True - is there is intersection,
        False - otherwise.
        """

        def upgrade(lohi, f0, f1):
            if f0 > 0.0:
                lohi[1] = min(lohi[1], -f1 / f0)
            elif f0 < 0.0:
                lohi[0] = max(lohi[0], -f1 / f0)
            else:
                return f1 <= 0.0
            return lohi[0] <= lohi[1]

        [a, b, c] = self.points
        xa, ya, za = a[0], a[1], a[2]
        xb, yb, zb = b[0], b[1], b[2]
        xc, yc, zc = c[0], c[1], c[2]
        xl, yl, zl = box.lo[0], box.lo[1], box.lo[2]
        xh, yh, zh = box.hi[0], box.hi[1], box.hi[2]

        lohi = [0.0, 1.0]

        basic_eqns_count = 9
        b = [[  xb - xa ,   xc - xa , -(xh - xa)],
             [-(xb - xa), -(xc - xa),   xl - xa ],
             [  yb - ya ,   yc - ya , -(yh - ya)],
             [-(yb - ya), -(yc - ya),   yl - ya ],
             [  zb - za ,   zc - za , -(zh - za)],
             [-(zb - za), -(zc - za),   zl - za ],
             [      -1.0,       0.0,        0.0 ],
             [       0.0,      -1.0,        0.0 ],
             [       1.0,       1.0,       -1.0 ]]

        for i in range(basic_eqns_count):
            bi0 = b[i][0]
            if bi0 == 0.0:
                if not upgrade(lohi, b[i][1], b[i][2]):
                    return False
            else:
                for j in range(i + 1, basic_eqns_count):
                    if bi0 * b[j][0] < 0.0:
                        f0 = bi0 * b[j][1] - b[j][0] * b[i][1]
                        f1 = bi0 * b[j][2] - b[j][0] * b[i][2]
                        if bi0 < 0.0:
                            f0 = -f0
                            f1 = -f1
                        if not upgrade(lohi, f0, f1):
                            return False
        return True

# ==================================================================================================

class Box:
    """
    Box - locus of points (X, Y, Z):
          (MinX <= X <= MaxX) and
          (MinY <= Y <= MaxY) and
          (MinZ <= Z <= MaxZ).
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, lo, hi):
        """
        Constructor from points.

        Parameters
        ----------
        lo : [float]
            Array of lo bounds.
        hi : [float]
            Array of high bounds.
        """

        self.lo = np.array(lo)
        self.hi = np.array(hi)

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def from_intervals(xint, yint, zint):
        """
        Create from intervals.

        Parameters
        ----------
        xint : [float, float]
            X interval.
        yint : [float, float]
            Y interval.
        zint : [float, float]
            Z interval.

        Returns
        -------
        Box
            Box created from intervals.
        """

        return Box([xint[0], yint[0], zint[0]], [xint[1], yint[1], zint[1]])

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def from_points(ps):
        """
        Create box from points.

        Parameters
        ----------
        ps : [Point]
            Points list.

        Returns
        -------
        Box
            Box created from points.
        """

        xs = [p[0] for p in ps]
        ys = [p[1] for p in ps]
        zs = [p[2] for p in ps]

        # Lo point (MinX, MinY, MinZ).
        lo = np.array([min(xs), min(ys), min(zs)])

        # Hi point (MaxX, MaxY, MaxZ).
        hi = np.array([max(xs), max(ys), max(zs)])

        return Box(lo, hi)

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def from_triangles(ts):
        """
        Create box from triangles.

        Parameters
        ----------
        ts : [Triangle]
            Triangles list.

        Returns
        -------
        Box
            Box, created from triangles list.
        """

        # Extract points from all triangles (2-dimensional list).
        pss = [t.points for t in ts]

        # Merge lists and create box from merged points list.
        return Box.from_points([p for ps in pss for p in ps])

    # ----------------------------------------------------------------------------------------------

    def delta_box(self, ds):
        """
        Create delta box.

        Parameters
        ----------
        ds : [float]
            Array of deltas.
        """

        return Box(self.lo - ds, self.hi + ds)

    # ----------------------------------------------------------------------------------------------

    def cube(self):
        """
        Get containing cube.

        Returns
        -------
        Box
            Containing cube.
        """

        h = 0.5 * max(self.hi - self.lo)
        m = 0.5 * (self.lo + self.hi)
        return Box(m - h, m + h)

    # ----------------------------------------------------------------------------------------------

    def eights_subboxes(self):
        """
        Return array of 8 subboxes.

        Returns
        -------
        [Box, Box, Box, Box, Box, Box, Box, Box]
            Eight subboxes.
        """

        [xlo, ylo, zlo] = self.lo
        [xhi, yhi, zhi] = self.hi
        [xmd, ymd, zmd] = (self.lo + self.hi) / 2.0

        return [Box([xlo, ylo, zlo], [xmd, ymd, zmd]),
                Box([xmd, ylo, zlo], [xhi, ymd, zmd]),
                Box([xlo, ymd, zlo], [xmd, yhi, zmd]),
                Box([xmd, ymd, zlo], [xhi, yhi, zmd]),
                Box([xlo, ylo, zmd], [xmd, ymd, zhi]),
                Box([xmd, ylo, zmd], [xhi, ymd, zhi]),
                Box([xlo, ymd, zmd], [xmd, yhi, zhi]),
                Box([xmd, ymd, zmd], [xhi, yhi, zhi])]

    # ----------------------------------------------------------------------------------------------

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Box: X({self.lo[0]} - {self.hi[0]}), '\
               f'Y({self.lo[1]} - {self.hi[1]}), '\
               f'Z({self.lo[2]} - {self.hi[2]})'

    # ----------------------------------------------------------------------------------------------

    def volume(self):
        """
        Get volume.

        Returns
        -------
        float
            Volume.
        """

        return (self.hi - self.lo).prod()

    # ----------------------------------------------------------------------------------------------

    def is_point_inside(self, p):
        """
        Check if point inside.

        Parameters
        ----------
        p : Point
            Point.

        Returns
        -------
        True - if point is inside,
        Fals e- if point is not inside.
        """

        return (self.lo[0] <= p[0] <= self.hi[0]) \
               and (self.lo[1] <= p[1] <= self.hi[1]) \
               and (self.lo[2] <= p[2] <= self.hi[2])

    # ----------------------------------------------------------------------------------------------

    def is_intersect_with_box(self, other_box):
        """
        Check for intersection with another box.

        Parameters
        ----------
        other_box : Box
            Box for check intersection with.

        Returns
        -------
        True - if boxes intersect,
        False - if boxes don't intersect.
        """

        def is_no(i):
            return (other_box.lo[i] > self.hi[i]) or (other_box.hi[i] < self.lo[i])

        if is_no(0) or is_no(1) or is_no(2):
            return False
        else:
            return True

    # ----------------------------------------------------------------------------------------------

    def fstore_points_coordinates(self, f):
        """
        Store to file points coordinates.

        Parameters
        ----------
        f : file
            File.
        """

        f.write(f'{self.lo[0]} {self.lo[1]} {self.lo[2]}\n')
        f.write(f'{self.hi[0]} {self.lo[1]} {self.lo[2]}\n')
        f.write(f'{self.hi[0]} {self.hi[1]} {self.lo[2]}\n')
        f.write(f'{self.lo[0]} {self.hi[1]} {self.lo[2]}\n')
        f.write(f'{self.lo[0]} {self.lo[1]} {self.hi[2]}\n')
        f.write(f'{self.hi[0]} {self.lo[1]} {self.hi[2]}\n')
        f.write(f'{self.hi[0]} {self.hi[1]} {self.hi[2]}\n')
        f.write(f'{self.lo[0]} {self.hi[1]} {self.hi[2]}\n')

# ==================================================================================================

class TrianglesCloud:
    """
    Triangles cloud realization.
    Triangles cloud is container for triangles objects.
    It can contain triangles in self.Triangles field or subclouds in self.Subclouds list
    (BUT NOT IN BOTH).
    Each subcloud in self.Subclouds list is an instance of class TrianglesCloud.
    Example 1::
      One solid cloud of 4 triangles and no subclouds.
      TrianglesCloud:
        Triangles = [t1, t2, t3, t4]
        Subclouds = []
    Example 2::
      Cloud is separated into binary tree of subclouds.
                                          TrianglesCloud([])
                                                | |
                         *----------------------* *----------------------*
                         |                                               |
                         V                                               V
                  TrianglesCloud([])                              TrianglesCloud([])
                        | |                                             | |
             *----------* *----------*                       *----------* *----------*
             |                       |                       |                       |
             V                       V                       V                       V
      TrianglesCloud([t1])    TrianglesCloud([t2])    TrianglesCloud([t3])    TrianglesCloud([t4])
    """

    # ----------------------------------------------------------------------------------------------

    # Maximum count of triangles in list.
    max_list_triangles_count = 1

    # ----------------------------------------------------------------------------------------------

    def __init__(self, triangles_list):
        """
        Constructor by triangles list.

        Parameters
        ----------
        triangles_list : [Triangle]
            List of the triangles.
        """

        # List of triangles.
        self.triangles = triangles_list

        # List of children clouds.
        self.subclouds = []

        # Box.
        self.box = Box.from_triangles(self.triangles)

        # Build subclouds tree.
        self.build_subclouds_tree()

    # ----------------------------------------------------------------------------------------------

    def print(self, level=0):
        """
        Print on screen.

        Parameters
        ----------
        level : int
            Level (offset from left side).
        """

        off = '  ' * level

        if self.is_list():
            print(f'{off}TCloud (level = {level}, count = {len(self.triangles)} ) : {self.triangles}')
        else:
            print(f'{off}TCloud (level = {level}):')
            for sc in self.subclouds:
                sc.print(level + 1)

    # ----------------------------------------------------------------------------------------------

    def build_subclouds_tree(self):
        """
        Build subclouds tree.
        """

        if len(self.triangles) > TrianglesCloud.max_list_triangles_count:

            # Separate triangles list and build new subtrees.
            # self.Triangles must be cleaned up.
            new_tri_lists = self.separate_triangles_list(self.triangles)
            self.triangles = []
            self.subclouds = [TrianglesCloud(li) for li in new_tri_lists]

        else:

            # Do nothings.
            # Triangles stay triangles.
            pass

    # ----------------------------------------------------------------------------------------------

    def separate_triangles_list(self, triangles_list):
        """
        Separate triangle slist into pair of lists.

        Parameters
        ----------
        triangles_list : [Triangle]
            List of triangles.

        Returns
        -------
            A list of two lists of triangles.
        """

        assert len(triangles_list) > 1, 'internal error'

        box = Box.from_points([t.center() for t in triangles_list])

        # Edge points box.
        xmax, ymax, zmax = box.hi[0], box.hi[1], box.hi[2]
        xmin, ymin, zmin = box.lo[0], box.lo[1], box.lo[2]

        # Checking the long side.
        lenxyz = [xmax - xmin, ymax - ymin, zmax - zmin]
        indxyz = lenxyz.index(np.amax(lenxyz))

        # separation
        triangles_list = Triangle.sorting_by_the_selected_axis(triangles_list, indxyz)
        mid_of_list = len(triangles_list) // 2

        return [triangles_list[:mid_of_list],  triangles_list[mid_of_list:]]

    # ----------------------------------------------------------------------------------------------

    def is_list(self):
        """
        Check if cloud is list.

        Returns
        -------
        True - if cloud is alist,
        False - if cloud is not a list.
        """

        is_triangles = (self.triangles != [])
        is_subclouds = (self.subclouds != [])
        assert not (is_triangles and is_subclouds)

        return is_triangles

    # ----------------------------------------------------------------------------------------------

    def intersection_with_triangles_cloud(self, tc):
        """
        Find intersection with another triangles cloud.

        Parameters
        ----------
        tc : TrianglesCloud
            Another cloud of triangles.

        Returns
        -------
        [] - if there is no intersections,
        [[t1, t2]] - list of triangles pairs.
        """

        # Cold check.
        if not self.box.is_intersect_with_box(tc.box):
            return []

        if self.subclouds:
            return list(itertools.chain(*[a.intersection_with_triangles_cloud(tc)
                                          for a in self.subclouds]))

        elif tc.subclouds:
            return list(itertools.chain(*[self.intersection_with_triangles_cloud(b)
                                          for b in tc.subclouds]))

        else:
            # List triangles in intersected box potentially intersect.
            return [[t1, t2]
                    for t1 in self.triangles
                    for t2 in tc.triangles
                    if (not t1.has_common_points_with(t2)) and t1.is_intersect_with_triangle(t2)]

    # ----------------------------------------------------------------------------------------------

    def is_any_triangle_intersects_with_box(self, b):
        """
        If any triangle has intersection with box.

        Parameters
        ----------
        b : Box
            Box.

        Returns
        -------
        True - if any triangle has intersection with box,
        False - otherwise.
        """

        if self.is_list():

            # The list holds only one triangle.
            assert TrianglesCloud.max_list_triangles_count == 1
            return self.triangles[0].is_intersect_with_box(b)

        else:

            # Check all subclouds.
            for s in self.subclouds:
                if s.is_any_triangle_intersects_with_box(b):
                    return True
            return False

# --------------------------------------------------------------------------------------------------

    def paint_triangles_intersects_with_box(self, b, color):
        """
        """

        if self.is_list():

            # The list holds only one triangle.
            assert TrianglesCloud.max_list_triangles_count == 1
            if self.triangles[0].is_intersect_with_box(b):
                self.triangles[0].back_ref['M'] = color

        else:

            # Check all subclouds.
            for s in self.subclouds:
                s.paint_triangles_intersects_with_box(b, color)

# ==================================================================================================

class EnclosingParallelepipedsTree:
    """
    Tree of parallelepipeds, enclosing some figure.
    Each node in tree can contain children (up to 8).
    """

    # ----------------------------------------------------------------------------------------------

    # Small epsilon for boxes overlapping.
    epsilon = 1.0e-8

    # ----------------------------------------------------------------------------------------------

    def __init__(self, box):
        """
        Default constructor.

        Parameters
        ----------
        box : Box
            Boxx.
        """

        self.box = box
        self.active = True

        # Empty children.
        self.children = []

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def from_triangles_cloud(tc, min_box_volume):
        """
        Create from triangles cloud.

        Parameters
        ----------
        tc : TrianglesCloud
            Triangles cloud.
        min_box_volume : float
            Box volume limit.

        Returns
        -------
        EnclosingParallelepipedsTree
            Result tree.
        """

        # Take cube box overlapped the surface with delta 10%.
        c = tc.box.cube()
        b = c.delta_box((c.hi - c.lo)[0] / 10.0)

        return EnclosingParallelepipedsTree.from_triangles_cloud_inner(tc, b, min_box_volume)

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def from_triangles_cloud_inner(tc, box, min_box_volume):
        """
        Create parallelepipeds tree from triangles cloud and box.

        Parameters
        ----------
        tc : TrianglesCloud
            Triangles cloud.
        box : Box
              Box.
        min_box_volume : float
            Box volume limit.

        Returns
        -------
        EnclosingParallelepipedsTree
            Result tree.
        """

        # Too small box.
        if box.volume() < min_box_volume:
            return None

        # Create only if any triangle intersects the box.
        if not tc.is_any_triangle_intersects_with_box(box):
            return None

        # Create box for tree.
        tree = EnclosingParallelepipedsTree(box)

        # Process subtrees.
        sbs = box.eights_subboxes()
        tree.children = []
        for sb in sbs:
            child = EnclosingParallelepipedsTree.from_triangles_cloud_inner(tc, sb, min_box_volume)
            if not child is None:
                tree.children.append(child)

        return tree

    # ----------------------------------------------------------------------------------------------

    def is_leaf(self):
        """
        Check if leaf.

        Returns
        -------
        bool
            True - if it is leaf,
            False - otherwise.
        """

        return not self.children

    # ----------------------------------------------------------------------------------------------

    def depth(self):
        """
        Depth - count of levels.

        Returns
        -------
        int
            Depth.
        """

        if not self.children:
            return 1
        else:
            return 1 + max(np.array([ch.depth() for ch in self.children], dtype=int))

    # ----------------------------------------------------------------------------------------------

    def parallelepipeds_count(self):
        """
        Count of parallelepipeds.

        Returns
        -------
        int
            Count of parallelepipeds.
        """

        return 1 + np.array([ch.parallelepipeds_count() for ch in self.children], dtype=int).sum()

    # ----------------------------------------------------------------------------------------------

    def active_leaf_parallelepipeds_count(self):
        """
        Count of leaf parallelepipeds.

        Returns
        -------
        int
            Count of leaf parallelepipeds.
        """

        if self.is_leaf():
            if self.active:
                return 1
            else:
                return 0
        else:
            return np.array([ch.active_leaf_parallelepipeds_count()
                             for ch in self.children], dtype=int).sum()

    # ----------------------------------------------------------------------------------------------

    def print(self):
        """
        Print information.
        """

        print(f'EnclosingParallelepipedsTree : {self.box}, '
              f'{EnclosingParallelepipedsTree.epsilon}')

    # ----------------------------------------------------------------------------------------------

    def store(self, filename, is_store_only_leafs=True):
        """
        Store to file.

        Parameters
        ----------
        filename : str
            Name of file.
        is_store_only_leafs : bool
            Flag for storing only leafs.
        """

        pp_count = 0
        if is_store_only_leafs:
            pp_count = self.active_leaf_parallelepipeds_count()
        else:
            pp_count = self.parallelepipeds_count()
        points_count = pp_count * 8

        with open(filename, 'w', newline='\n') as f:
            f.write('TITLE="EnclosingParallelepipedsTree"\n')
            f.write('VARIABLES="X", "Y", "Z"\n')
            f.write(f'ZONE NODES={points_count}, ELEMENTS={pp_count}, '
                    f'DATAPACKING=POINT, ZONETYPE=FEBRICK\n')
            self.fstore_box_points_coordinates(f, is_store_only_leafs)
            for i in range(pp_count):
                s = i * 8
                f.write(f'{s + 1} {s + 2} {s + 3} {s + 4} {s + 5} {s + 6} {s + 7} {s + 8}\n')
            f.close()

    # ----------------------------------------------------------------------------------------------

    def fstore_box_points_coordinates(self, f, is_store_only_leafs):
        """
        Store box points coordinates to file.

        Parameters
        ----------
        f : file
            File.
        is_store_only_leafs : bool
            Flag for storing only leafs.
        """

        if not is_store_only_leafs or (self.is_leaf() and self.active):
            self.box.fstore_points_coordinates(f)

        for ch in self.children:
            ch.fstore_box_points_coordinates(f, is_store_only_leafs)

# ==================================================================================================

if __name__ == '__main__':

    # Distance between triangles.
    t1 = Triangle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    t2 = Triangle(np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.1]), np.array([2.0, 0.5, -0.1]))
    assert t1.is_intersect_with_triangle(t2)
    t1 = Triangle(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    t2 = Triangle(np.array([2.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    assert not t1.is_intersect_with_triangle(t2)

# ==================================================================================================
