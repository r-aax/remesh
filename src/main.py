from remesher_tong import RemesherTong
from remesher_isotropic import RemesherIsotropic
import msu

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    m = msu.Mesh('../cases/spheres/small_sphere_double.dat')
    #m = msu.Mesh('../cases/bunny/bunny_double.dat')
    #m = msu.Mesh('../cases/cylinder/cyl_100.dat')
    m.self_intersections_elimination(is_debug=True)
    m.store('res.dat')
    print('DONE')

#---------------------------------------------------------------------------------------------------
