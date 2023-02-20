from remesher_tong import RemesherTong
from remesher_isotropic import RemesherIsotropic

if __name__ == '__main__':
    #case = '../cases/naca/naca_t12.dat'
    #case = '../cases/blender_custom_meshes/one_hole_new.dat'
    case = '../cases/blender_custom_meshes/plate.dat'
    #case = '../cases/bunny_2.dat'
    #case = '../cases/pseudogrids/ex1.dat'
    RemesherTong().remesh(case, '../res_tong.dat')
    RemesherIsotropic().remesh(case, '../res_isotropic.dat')
