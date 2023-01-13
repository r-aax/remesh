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

    def remesh(self, mesh):
        """
        Remesh.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        self.remesh_time = time.time()
        mesh.remesh()
        self.remesh_time = time.time() - self.remesh_time
