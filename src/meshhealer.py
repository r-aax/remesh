import msu
import patcher
import geom


def store_and_say(mesh, f):
    """
    Store mesh and say.

    Parameters
    ----------
    mesh : msu.Mesh
        Mesh.
    f : str
        Filename.
    """

    mesh.store(f)
    print(f'{f} -- DONE')


def case_01_sphere_2():
    """
    Case 01.
    """

    c = 'case_01_sphere_2'
    f = '../cases/sphere_2.dat'

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    store_and_say(mesh, f'../{c}_phase_01_original.dat')

    # Delete intersections.
    mesh.delete_self_intersected_faces()
    c1 = mesh.ColorFree
    mesh.walk_until_border(mesh.lo_face(0), c1)
    c2 = mesh.ColorFree + 1
    mesh.walk_until_border(mesh.hi_face(0), c2)
    store_and_say(mesh, f'../{c}_phase_02_del_intersections.dat')

    # Del extra regions.
    mesh.delete_faces(msu.Mesh.ColorCommon)
    mesh.delete_faces(msu.Mesh.ColorBorder)
    mesh.delete_isolated_nodes()
    store_and_say(mesh, f'../{c}_phase_03_del_extra.dat')

    # Zip.
    zipper = patcher.Zipper(mesh)
    zipper.collect_border()
    zipper.zip(0, 1, is_flip_path_j=True)
    store_and_say(mesh, f'../{c}_phase_03_zipper.dat')

def case_02_sphere_2():
    """
    Case 02.
    """

    c = 'case_02_ex2'
    f = '../cases/pseudogrids/ex2.dat'

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    mesh.calculate_edges()
    store_and_say(mesh, f'../{c}_phase_01_original.dat')

    # Find intersections.
    tc = geom.TrianglesCloud(mesh.triangles_list())
    pairs = tc.intersection_with_triangles_cloud(tc)
    pairs = list(filter(lambda p: p[0].back_ref.glo_id < p[1].back_ref.glo_id, pairs))
    print('Pairs:')
    for pair in pairs:
        print(pair)
        [t1, t2] = pair
        ps = t1.find_intersection_with_triangle(t2)
        print(ps)
        ps = geom.delete_near_points(ps)
        print(ps)

def triangle_case():
    mesh = msu.Mesh()
    c = "triangle_case"
    p1 = [0.6, 0.3, 0.3]
    p2 = [0.6, -0.3, 0.3]
    p3 = [0.8, 0.7, 0.7]
    mesh.load('../cases/pseudogrids/ex1.dat')
    mesh.calculate_edges()
    mesh.split_face(mesh.faces[0], p1)
    mesh.split_edge(mesh.edges[0], p2)
    mesh.split_edge(mesh.edges[4], p3)
    store_and_say(mesh, f'../{c}.dat')


if __name__ == '__main__':
    #case_01_sphere_2()
    triangle_case()
