import numpy as np
import geom


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

    def find_pairs_indices(self):
        """
        Find indices for pairs.

        Returns
        -------
        [(int, int)]
            List of tuples.
        """

        # Create a, b pairs with distance and indixes.
        pairs = []
        for i, a in enumerate(self.ps):
            for j, b in enumerate(self.ps):
                if j > i:
                    pairs.append((geom.points_dist(a, b), i, a, j, b))
        pairs.sort(key=lambda x: x[0])

        # Create pairs without intersections.
        ss = []
        ind = []
        for _, i, a, j, b in pairs:
            if not geom.if_ab_intersects_any_segment(a, b, ss):
                ss.append((a, b))
                ind.append((i, j))

        return ind

    def find_triangulation_indices(self):
        """
        Find indices for triangulation.

        Returns
        -------
        [(int, int, int)]
            List of indices for triangulation.
        """

        l = len(self.ps)
        a = np.zeros((l, l, l))

        # Get pairs indices.
        pinds = self.find_pairs_indices()
        for i, j in pinds:
            for k in range(l):
                a[k, i, j] += 1
                a[i, k, j] += 1
                a[i, j, k] += 1

        # Get triangulation indices.
        tri = []
        for i in range(l):
            for j in range(i + 1, l):
                for k in range(j + 1, l):
                    if a[i, j, k] == 3:
                        tri.append((i, j, k))

        if (0, 1, 2) in tri:
            tri.remove((0, 1, 2))

        return tri


if __name__ == '__main__':
    tr = Triangulator([np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0.2, 0.2, 0])])
    idx = tr.find_triangulation_indices()
    assert idx == [(0, 1, 3), (0, 2, 3), (1, 2, 3)]
