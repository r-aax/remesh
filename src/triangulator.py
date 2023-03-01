import numpy


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

    def find_indices_for_triangulation(self):
        """
        Find indices for triangulation.

        Returns
        -------
        [(int, int, int)]
            List of tuples.
        """

        assert len(self.ps) == 4

        return [(0, 1, 3), (1, 2, 3), (2, 0, 3)]


if __name__ == '__main__':
    pass
