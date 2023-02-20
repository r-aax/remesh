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


def case_01_zip():
    """
    Case 01.
    Test zipper method for delete self-intersections for sphere_2 case.
    """

    c = 'case_01_zip'
    f = '../cases/sphere_2.dat'

    #
    # Without split faces.
    #

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    store_and_say(mesh, f'../{c}_ph_01_orig.dat')

    # Delete intersections.
    mesh.delete_self_intersected_faces()
    c1 = mesh.ColorFree
    mesh.walk_until_border(mesh.lo_face(0), c1)
    c2 = mesh.ColorFree + 1
    mesh.walk_until_border(mesh.hi_face(0), c2)
    store_and_say(mesh, f'../{c}_ph_02_del_intersections.dat')

    # Del extra regions.
    mesh.delete_faces(msu.Mesh.ColorCommon)
    mesh.delete_faces(msu.Mesh.ColorBorder)
    mesh.delete_isolated_nodes()
    store_and_say(mesh, f'../{c}_ph_03_del_extra.dat')

    # Zip.
    zipper = patcher.Zipper(mesh)
    zipper.collect_border()
    zipper.zip(0, 1, is_flip_path_j=True)
    store_and_say(mesh, f'../{c}_ph_04_zip.dat')

    #
    # With split faces.
    #

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    store_and_say(mesh, f'../{c}_ph_05_orig_2.dat')

    # Delete intersections.
    mesh.calculate_edges()
    mesh.split_self_intersected_faces()
    store_and_say(mesh, f'../{c}_ph_06_split_intersections_2.dat')
    mesh.delete_self_intersected_faces()
    c1 = mesh.ColorFree
    mesh.walk_until_border(mesh.lo_face(0), c1)
    c2 = mesh.ColorFree + 1
    mesh.walk_until_border(mesh.hi_face(0), c2)
    store_and_say(mesh, f'../{c}_ph_07_del_intersections_2.dat')

    # Del extra regions.
    mesh.delete_faces(msu.Mesh.ColorCommon)
    mesh.delete_faces(msu.Mesh.ColorBorder)
    mesh.delete_isolated_nodes()
    store_and_say(mesh, f'../{c}_ph_08_del_extra_2.dat')

    # Zip.
    zipper = patcher.Zipper(mesh)
    zipper.collect_border()
    zipper.zip(0, 1, is_flip_path_j=True)
    store_and_say(mesh, f'../{c}_ph_09_zip_2.dat')

    #
    # With double split faces.
    #

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    store_and_say(mesh, f'../{c}_ph_10_orig_3.dat')

    # Delete intersections.
    mesh.calculate_edges()
    mesh.split_self_intersected_faces()
    store_and_say(mesh, f'../{c}_ph_11_split_intersections_3.dat')
    mesh.calculate_edges()
    mesh.split_self_intersected_faces()
    store_and_say(mesh, f'../{c}_ph_12_more_split_intersections_3.dat')
    mesh.delete_self_intersected_faces()
    c1 = mesh.ColorFree
    mesh.walk_until_border(mesh.lo_face(0), c1)
    c2 = mesh.ColorFree + 1
    mesh.walk_until_border(mesh.hi_face(0), c2)
    store_and_say(mesh, f'../{c}_ph_13_del_intersections_3.dat')

    # Del extra regions.
    mesh.delete_faces(msu.Mesh.ColorCommon)
    mesh.delete_faces(msu.Mesh.ColorBorder)
    mesh.delete_isolated_nodes()
    store_and_say(mesh, f'../{c}_ph_14_del_extra_3.dat')

    # Zip.
    #zipper = patcher.Zipper(mesh)
    #zipper.collect_border()
    #zipper.zip(0, 1, is_flip_path_j=True)
    #store_and_say(mesh, f'../{c}_ph_15_zip_3.dat')


def case_02_self_intersections_elimination():
    """
    Case 02.
    Self-intersections elimination.
    """

    c = 'case_02_sie'
    f = '../cases/pseudogrids/ex2.dat'
    #f = '../cases/bunny_2.dat'

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    mesh.calculate_edges()
    store_and_say(mesh, f'../{c}_ph_01_orig.dat')

    # Find intersections.
    mesh.throw_intersection_points_to_faces()
    mesh.multisplit_by_intersection_points()
    store_and_say(mesh, f'../{c}_ph_02_cut.dat')

    for f in mesh.faces:
        print(f.triangle().area())


def case_03_triangle_split():
    """
    Case 03.
    Split triangle.
    """

    c = 'case_03_ts'
    f = '../cases/pseudogrids/ex1.dat'

    # Load.
    mesh = msu.Mesh()
    p1 = [0.6, 0.3, 0.3]
    p2 = [0.6, -0.3, 0.3]
    p3 = [0.8, 0.7, 0.7]
    mesh.load(f)
    mesh.calculate_edges()
    mesh.split_face(mesh.faces[0], p1)
    mesh.split_edge(mesh.edges[0], p2)
    mesh.split_edge(mesh.edges[4], p3)
    store_and_say(mesh, f'../{c}_ph_01_res.dat')


def case_04_triangle_multisplit():
    """
    Case 04.
    Split face with multiple points.
    """
    c = 'case_04_triangle_multisplit'
    f = '../cases/pseudogrids/ex1.dat'

    # Load.
    mesh = msu.Mesh()
    mesh.load(f)
    mesh.calculate_edges()
    store_and_say(mesh, f'../{c}_ph_01_orig.dat')

    # Split.
    f = mesh.faces[0]
    t = f.triangle()
    random_points = []
    for _ in range(10):
        random_points.append(t.random_point())
    mesh.multisplit_face(mesh.faces[0], random_points)
    store_and_say(mesh, f'../{c}_ph_02_multisplit.dat')


if __name__ == '__main__':
    case_01_zip()
    case_02_self_intersections_elimination()
    case_03_triangle_split()
    case_04_triangle_multisplit()
