import msu
import patcher


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


if __name__ == '__main__':
    case_01_sphere_2()
