import sys
import msu
import time
from remesher_tong import RemesherTong
from remesher_isotropic import RemesherIsotropic


def lrs(name_in, name_out_tong, name_out_isotropic):
    """
    Load, remesh, store.

    Parameters
    ----------
    name_in : str
        Name of input mesh file.
    name_out : str
        Name of output mesh file.
    """

    remesher_tong = RemesherTong()
    remesher_isotropic = RemesherIsotropic()

    remesher_tong.log.info(f'remesh_tong start : {name_in} -> {name_out_tong}')
    g = msu.Mesh()
    g.load(name_in)
    t0 = time.time()
    remesher_tong.remesh(g)
    t = time.time() - t0
    target_ice = g.target_ice()
    target_ice_perc = 100.0 * (target_ice / g.initial_target_ice)
    g.store(name_out_tong)
    remesher_tong.log.info(f'remesh_tong end : time = {t:.5f} s, target_ice = {target_ice} ({target_ice_perc}%)')

    remesher_isotropic.log.info(f'remesh_isotropic start : {name_in} -> {name_out_isotropic}')
    g = msu.Mesh()
    g.load(name_in)
    t0 = time.time()
    remesher_isotropic.remesh(g)
    t = time.time() - t0
    target_ice = g.target_ice()
    target_ice_perc = 100.0 * (target_ice / g.initial_target_ice)
    g.store(name_out_isotropic)
    remesher_isotropic.log.info(f'remesh_isotropic end : time = {t:.5f} s, target_ice = {target_ice} ({target_ice_perc}%)')


if __name__ == '__main__':
    lrs('../cases/naca/naca_t12.dat', '../res_tong.dat', '../res_isotropic.dat')
