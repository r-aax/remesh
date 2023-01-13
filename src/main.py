from remesher_tong import RemesherTong
from remesher_isotropic import RemesherIsotropic


if __name__ == '__main__':
    RemesherTong().remesh('../cases/naca/naca_t12.dat', '../res_tong.dat')
    RemesherIsotropic().remesh('../cases/naca/naca_t12.dat', '../res_isotropic.dat')
