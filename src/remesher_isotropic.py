from remesher import Remesher


class RemesherIsotropic(Remesher):
    """
    Isotropic remesher.
    """

    def __init__(self):
        """
        Constructor.
        """

        pass

    def remesh(self, mesh):
        """
        Remesh.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        mesh.new_remesh()
