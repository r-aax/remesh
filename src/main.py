from remesher_tong import RemesherTong
from remesher_isotropic import RemesherIsotropic


if __name__ == '__main__':
    #case = '../cases/naca/naca_t12.dat'
    case = '../cases/bunny_2.dat'
    RemesherTong().remesh(case, '../res_tong.dat')
    RemesherIsotropic().remesh(case, '../res_isotropic.dat')
