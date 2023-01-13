import sys
import time
import msu
import logging
from logging import StreamHandler, Formatter

# Log.
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
log.addHandler(handler)


class Remesher:
    """
    Main remesher class.
    """

    def __init__(self):
        """
        Constructor.
        """

        # Time for remesh.
        self.remesh_time = 0.0

        # Log.
        self.log = log

    def remesh_prepare(self, mesh):
        """
        Prepare mesh for remeshing.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        for f in mesh.faces:
            f.target_ice = f.area * f['Hi']

        mesh.initial_target_ice = mesh.target_ice()

    def remesh(self, name_in, name_out):
        """
        Remesh.

        Parameters
        ----------
        name_in : str
            Name of in file.
        name_out : str
            Name of out file.
        """

        self.log.info(f'remesh_{self.name} start : {name_in} -> {name_out}')

        # Load mesh.
        mesh = msu.Mesh()
        mesh.load(name_in)

        # Remesh with time calculation.
        self.remesh_time = time.time()
        self.inner_remesh(mesh)
        self.remesh_time = time.time() - self.remesh_time

        # Calculate indicator.
        t_ice = mesh.target_ice()
        t_perc = 100.0 * (t_ice / mesh.initial_target_ice)

        # Store mesh.
        mesh.store(name_out)

        self.log.info(f'remesh_{self.name} end : time = {self.remesh_time:.5f} s, target_ice = {t_ice} ({t_perc}%)')
