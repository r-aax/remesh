from remesher import Remesher


class RemesherTong(Remesher):
    """
    Tong remesher.
    """

    def __init__(self):
        """
        Constant.
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

        mesh.remesh()
