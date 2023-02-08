import numpy as np
import itertools


class Triangle:
    """
    Triangle - locus of points P:
               P = A + (B - A) * beta + (C - A) * gamma
               beta >= 0
               gamma >= 0
               beta + gamme <= 1
    """

    def __init__(self, a, b, c):
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
        """

        self.points = [a, b, c]

        # Back reference for linking with face.
        self.back_ref = None

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

    def is_potential_intersect_with_box(self, other_box):
        """
        Check for potential intersection with another box.

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

        return is_no(0) or is_no(1) or is_no(2)


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
        if not self.Box.is_potential_intersect_with_box(tc.Box):
            return []

        if self.subclouds != []:
            return list(itertools.chain(*[a.intersection_with_triangles_cloud(tc) for a in self.subclouds]))

        elif tc.subclouds != []:
            return list(itertools.chain(*[self.intersection_with_triangles_cloud(b) for b in tc.subclouds]))

        else:
            return [[t1, t2]
                    for t1 in self.Triangles
                    for t2 in tc.Triangles
                    if t1.intersection_with_triangle(t2) != []]


if __name__ == '__main__':
    pass
