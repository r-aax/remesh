import msu
import geom
import numpy as np

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    m = msu.Mesh('../cases/cylinder/cyl_100.dat')
    tl = m.triangles_list()
    tc = geom.TrianglesCloud(tl)
    et = geom.EnclosingParallelepipedsTree.from_triangles_cloud(tc, 1.0e-8)
    et.store('tree.dat')
    et.print()
    print(f'enc par tree with {et.leaf_parallelepiped_count()} constructed')

#---------------------------------------------------------------------------------------------------
