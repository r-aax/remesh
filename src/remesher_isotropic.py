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
        self.name = 'isotropic'

    def inner_remesh(self, mesh):
        """
        Remesh.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        mesh.new_remesh()
