import numpy as np
import numpy.linalg as la
import itertools
import mth


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

    return [ps[i] for i in range(l) if delete_flags[i]]


class Triangle:
    """
    Triangle - locus of points P:
               P = A + (B - A) * beta + (C - A) * gamma
               beta >= 0
               gamma >= 0
               beta + gamme <= 1
    """

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

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Triangle ({self.points[0]}, {self.points[1]}, {self.points[2]})'

    def center(self):
        """
        Get triangle center.

        Returns
        -------
        Point
            Center point.
        """

        return (self.points[0] + self.points[1] + self.points[2]) / 3.0

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

        return points


class Box:
    """
    Box - locus of points (X, Y, Z):
          (MinX <= X <= MaxX) and
          (MinY <= Y <= MaxY) and
          (MinZ <= Z <= MaxZ).
    """

    def __init__(self, ps):
        """
        Constructor from points.

        Parameters
        ----------
        ps : [Point]
            Box.
        """

        xs = [p[0] for p in ps]
        ys = [p[1] for p in ps]
        zs = [p[2] for p in ps]

        # Lo point (MinX, MinY, MinZ).
        self.lo = np.array([min(xs), min(ys), min(zs)])

        # Hi point (MaxX, MaxY, MaxZ).
        self.hi = np.array([max(xs), max(ys), max(zs)])

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

        return Box(ps)

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
        return Box([p for ps in pss for p in ps])

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Box: X({self.lo[0]} - {self.hi[0]}), Y({self.lo[1]} - {self.hi[1]}), Z({self.lo[2]} - {self.hi[2]})'

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

    # Maximum count of triangles in list.
    max_list_triangles_count = 1

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

    def build_subclouds_tree(self):
        """
        Build subclouds tree.
        """

        if len(self.triangles) > TrianglesCloud.max_list_triangles_count:

            # Separate triangles list and buld new subtrees.
            # self.Triangles must be cleaned up.
            new_tri_lists = self.separate_triangles_list(self.triangles)
            self.triangles = []
            self.subclouds = [TrianglesCloud(li) for li in new_tri_lists]

        else:

            # Do nothings.
            # Triangles stay triangles.
            pass

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

        if is_triangles and is_subclouds:
            raise Exception('internal error')

        return is_triangles

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

        if self.subclouds != []:
            return list(itertools.chain(*[a.intersection_with_triangles_cloud(tc) for a in self.subclouds]))

        elif tc.subclouds != []:
            return list(itertools.chain(*[self.intersection_with_triangles_cloud(b) for b in tc.subclouds]))

        else:
            # List triangles in intersected box potentially intersect.
            return [[t1, t2]
                    for t1 in self.triangles
                    for t2 in tc.triangles
                    if (not t1.has_common_points_with(t2)) and t1.is_intersect_with_triangle(t2)]


if __name__ == '__main__':

    # Distance between triangles.
    t1 = Triangle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    t2 = Triangle(np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.1]), np.array([2.0, 0.5, -0.1]))
    assert t1.is_intersect_with_triangle(t2)
    t1 = Triangle(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    t2 = Triangle(np.array([2.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    assert not t1.is_intersect_with_triangle(t2)
