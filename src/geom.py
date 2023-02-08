import numpy as np


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
