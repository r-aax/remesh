import numpy as np
import numpy.linalg as la
import geom
import geom2d
import scipy


class Triangulator:
    """
    Class triangulator.
    """

    def __init__(self, ps):
        """
        Constructor.

        Parameters
        ----------
        ps : [Point]
            Points list (different points).
        """

        self.ps = ps

    def delaunay_triangulation(self, ps2):
        """
        Delaunay triangulation

        Parameters
        ----------
        ps2 : [Point]
            List of points for which we have to ignore 3rd coordinate.

        Returns
        -------
        [(int, int, int)]
            List of indices for triangulation.
        """

        points = [[p[0], p[1]] for p in ps2]
        tri = scipy.spatial.Delaunay(points)
        simp = tri.simplices

        return [(s[0], s[1], s[2]) for s in simp]

    def bruteforce_triangulation(self, ps2):
        """
        Bruteforce triangulation

        Parameters
        ----------
        ps2 : [Point]
            List of points for which we have to ignore 3rd coordinate.

        Returns
        -------
        [(int, int, int)]
            List of indices for triangulation.
        """

        points = [geom2d.Vect(p[0], p[1]) for p in ps2]

        return geom2d.triangulation(points)

    def find_triangulation_indices(self):
        """
        Find indices for triangulation.

        Returns
        -------
        [(int, int, int)]
            List of indices for triangulation.
        """

        # Parallel move all points to [0] point.
        v = self.ps[0].copy()
        ps2 = [p - v for p in self.ps]

        # Find normal of tri and target normal.
        n = np.cross(ps2[1], ps2[2])
        n = n / la.norm(n)
        nxy = np.array([0.0, 0.0, 1.0])

        # Rotate all points to 0XY.
        m = geom.rotation_matrix_from_vectors(n, nxy)
        ps2 = [m.dot(p) for p in ps2]

        # Triangulation.
        return self.delaunay_triangulation(ps2)
        # return self.bruteforce_triangulation(ps2)


if __name__ == '__main__':

    # Simple case: 1 point in triangle.
    tr = Triangulator([np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0.2, 0.2, 0])])
    idx = tr.find_triangulation_indices()
    assert idx == [(1, 3, 0), (3, 2, 0), (2, 3, 1)]

    # Point on side.
    tr = Triangulator([np.array([0, 0, 0]), np.array([2, 0, 0]), np.array([1, 1, 0]), np.array([1, 0, 0])])
    idx = tr.find_triangulation_indices()
    assert idx == [(3, 2, 0), (2, 3, 1)]

    # "False triangle". 2 points.
    tr = Triangulator([np.array([0, 0, 0]), np.array([2, 0, 0]), np.array([1, 1, 0]),
                       np.array([0.8, 0.2, 0]), np.array([1.2, 0.4, 0])])
    idx = tr.find_triangulation_indices()
    assert idx == [(1, 3, 0), (3, 2, 0), (2, 4, 1), (4, 3, 1), (3, 4, 2)]
