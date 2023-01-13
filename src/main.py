import sys
import msu
import time
import logging
from logging import StreamHandler, Formatter

# Log.
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
log.addHandler(handler)


def lrs(name_in, name_out):
    """
    Load, remesh, store.

    Parameters
    ----------
    name_in : str
        Name of input mesh file.
    name_out : str
        Name of output mesh file.
    """

    log.info(f'remesh start : {name_in} -> {name_out}')
    g = msu.Mesh()
    g.load(name_in)
    t0 = time.time()
    g.new_remesh()
    t = time.time() - t0
    target_ice = g.target_ice()
    target_ice_perc = 100.0 * (target_ice / g.initial_target_ice)
    g.store(name_out)
    log.info(f'remesh end : time = {t:.5f} s, target_ice = {target_ice} ({target_ice_perc}%)')


if __name__ == '__main__':
    # lrs('../cases/naca/naca_t05.dat', '../res_naca_t05.dat')
    # lrs('../cases/naca/naca_t12.dat', '../res_naca_t12.dat')
    # lrs('../cases/naca/naca_t25.dat', '../res_naca_t25.dat')
    # lrs('../cases/blender_custom_meshes/holes.dat', '../res_holes.dat')
    # lrs('../cases/blender_custom_meshes/snowman.dat', '../res_snowman.dat')
    lrs('../cases/bunny.dat', '../res_bunny.dat')
