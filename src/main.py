import msu
import geom

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    m = msu.Mesh('../cases/blender/small_sphere.dat')
    tl = m.triangles_list()
    tc = geom.TrianglesCloud(tl)
    tc.print()

#---------------------------------------------------------------------------------------------------
