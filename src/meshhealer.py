import msu
import patcher
import geom
import numpy as np
import random


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
    f = '../cases/sphere_2.dat'

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    mesh.calculate_edges()
    store_and_say(mesh, f'../{c}_phase_01_original.dat')

    # Find intersections.
    for f in mesh.faces:
        f.split_points = []
    tc = geom.TrianglesCloud(mesh.triangles_list())
    pairs = tc.intersection_with_triangles_cloud(tc)
    pairs = list(filter(lambda p: p[0].back_ref.glo_id < p[1].back_ref.glo_id, pairs))
    print('Pairs:')
    for pair in pairs:
        print(pair)
        [t1, t2] = pair
        ps = t1.find_intersection_with_triangle(t2)
        ps = geom.delete_near_points(ps)
        t1.back_ref.split_points = t1.back_ref.split_points + ps
        t2.back_ref.split_points = t2.back_ref.split_points + ps
    for f in mesh.faces:
        f.split_points = geom.delete_near_points(f.split_points)
    ff = [f for f in mesh.faces]
    for f in ff:
        mesh.multisplit_face(f, f.split_points)
    store_and_say(mesh, f'../{c}_phase_02_cut.dat')


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


def case_04_triangle_multisplit():
    """
    Case 04.
    Split face with multiple points.
    """
    c = 'case_04_ex1'
    f = '../cases/pseudogrids/ex1.dat'

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    mesh.calculate_edges()
    store_and_say(mesh, f'../{c}_phase_01_original.dat')

    # Split.
    f = mesh.faces[0]
    t = f.triangle()
    random_points = []
    for _ in range(10):
        random_points.append(t.random_point())
    mesh.multisplit_face(mesh.faces[0], random_points)
    store_and_say(mesh, f'../{c}_phase_02_multisplit.dat')


if __name__ == '__main__':
    case_02_sphere_2()
