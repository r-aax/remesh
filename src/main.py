import msu
import geom
import numpy as np

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    m = msu.Mesh('../cases/blender/small_sphere.dat')
    tl = m.triangles_list()
    tc = geom.TrianglesCloud(tl)
    et = geom.EnclosingParallelepipedsTree.from_triangles_cloud(tc, 0.0001)
    et.print()
    print(et.parallelepipeds_count())
    et.store('tree.dat')

#---------------------------------------------------------------------------------------------------
