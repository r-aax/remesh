import time
from remesher import Remesher


class RemesherTong(Remesher):
    """
    Tong remesher.
    """

    def __init__(self):
        """
        Constant.
        """

        Remesher.__init__(self)
        self.name = 'tong'

    def inner_remesh(self, mesh):
        """
        Remesh.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        mesh.remesh()
