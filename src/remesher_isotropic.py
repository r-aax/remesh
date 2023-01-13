import time
from remesher import Remesher


class RemesherIsotropic(Remesher):
    """
    Isotropic remesher.
    """

    def __init__(self):
        """
        Constructor.
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
        mesh.new_remesh()
        self.remesh_time = time.time() - self.remesh_time
