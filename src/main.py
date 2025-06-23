from remesher_tong import RemesherTong
from remesher_isotropic import RemesherIsotropic
import msu

def case_isotropic_remesh(in_file, out_file):
    """
    Isotropic remesh.

    Parameters
    ----------
    in_file : str
        In file.
    out_file : str
        Out file.
    """

    RemesherIsotropic().remesh(in_file, out_file, steps=1)


if __name__ == '__main__':
    #case = '../cases/naca/naca_t12.dat'
    #case = '../cases/blender_custom_meshes/one_hole_new.dat'
    #case = '../Tong_step_34.dat'
    #case = '../cases/blender_custom_meshes/plate.dat'
    #case = '../cases/bunny_fixed.dat'
    #case = '../cases/pseudogrids/ex1.dat'
    #RemesherTong(tracking_evolution=True).remesh(case, '../res_tong.dat', steps=150)
    #case_isotropic_remesh(case, '../res_isotropic.dat')
    m = msu.Mesh('../cases/wing_50.dat', False)
    m.print()
    m = msu.Mesh('../cases/air_inlet.dat', False)
    m.print()
