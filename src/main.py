from venv import create

import msu
import geom
import numpy as np
import random
from collections import deque
from src.geom import EnclosingParallelepipedsTree

# ==================================================================================================

def make_cloud_structures(mesh, glob_i):
    """
    Make triangles cloud and parallelelipeds tree.

    Parameters
    ----------
    mesh : msu.Mesh
        Mesh.
    glob_i : int
        Global iteration.

    Returns
    -------
    (geom.TrianglesCloud, geom.EnclosingParallelepipedsTree)
        Structures.
    """

    mesh.store(f'glob_i_{glob_i}_ph_01_original.dat')
    min_box_volume = 1.0e-9 # constant
    tc = geom.TrianglesCloud(mesh.triangles_list())
    pt = geom.EnclosingParallelepipedsTree.from_triangles_cloud(tc, min_box_volume=min_box_volume)
    pt.store(f'glob_i_{glob_i}_ph_02_pt_all.dat', is_store_only_leafs=False)
    pt.store(f'glob_i_{glob_i}_ph_03_pt_active.dat', is_store_only_leafs=True)

    return tc, pt

# --------------------------------------------------------------------------------------------------

def create_map(d, glob_i):
    """
    Create map of parallelepipeds.

    Parameters
    ----------
    d : int
        Tree level.
    glob_i : int
        Global iteration.

    Returns
    -------
    ([[[object]]], int)
        Map and its side.
    """

    print(f'Glob I {glob_i} : create map')

    s = 2**(d - 1)
    m = []
    for i in range(s):
        m.append([])
        for j in range(s):
            m[i].append([])
            for k in range(s):
                m[i][j].append(None)

    return m, s

# --------------------------------------------------------------------------------------------------

def fill_map(m, pt, glob_i):
    """
    Fill map with leafs.

    Parameters
    ----------
    m : [[[any]]]
        Map.
    pt : geom.EnclosingParallelepipedsTree
        Tree.
    glob_i : int
        Global iteration.
    """

    print(f'Glob I {glob_i} : fill map')

    def reg(t, b):
        if t.is_leaf():
            r = 0.5 * (t.box.hi + t.box.lo) - b.lo
            r = r / ((b.hi - b.lo) / len(m))
            i, j, k = int(r[0]), int(r[1]), int(r[2])
            m[i][j][k] = t
        else:
            for ch in t.children:
                reg(ch, b)

    reg(pt, pt.box)

# --------------------------------------------------------------------------------------------------

def reduce_with_underlying_mesh_step(mesh, glob_i):

    # Make structures.
    tc, pt = make_cloud_structures(mesh, glob_i)
    d = pt.depth()
    print(f'Glob I {glob_i} : active parallelepipeds {pt.active_leaf_parallelepipeds_count()}')
    print(f'Glob I {glob_i} : depth {d}')

    # Create map.
    m, s = create_map(d, glob_i)
    fill_map(m, pt, glob_i)

    # Walk outer cells.
    for k in range(s):
        q = deque()
        for i in range(s):
            m[i][0][k] = 1
            q.append([i, 0])
            m[i][s - 1][k] = 1
            q.append([i, s - 1])
        for j in range(s):
            m[0][j][k] = 1
            q.append([0, j])
            m[s - 1][j][k] = 1
            q.append([s - 1, j])
        while len(q) > 0:
            [i, j] = q.pop()
            if i > 0:
                if m[i - 1][j][k] is None:
                    m[i - 1][j][k] = 1
                    q.append([i - 1, j])
            if i < s - 1:
                if m[i + 1][j][k] is None:
                    m[i + 1][j][k] = 1
                    q.append([i + 1, j])
            if j > 0:
                if m[i][j - 1][k] is None:
                    m[i][j - 1][k] = 1
                    q.append([i, j - 1])
            if j < s - 1:
                if m[i][j + 1][k] is None:
                    m[i][j + 1][k] = 1
                    q.append([i, j + 1])

    # Mark boxes.
    print(glob_i, 'mark boxes')
    for i in range(1, s - 1):
        for j in range(1, s - 1):
            for k in range(1, s - 1):
                if isinstance(m[i][j][k], geom.EnclosingParallelepipedsTree):
                    good = (m[i - 1][j][k] == 1) or (m[i + 1][j][k] == 1) \
                           or (m[i][j - 1][k] == 1) or (m[i][j + 1][k] == 1) \
                           or (m[i - 1][j - 1][k] == 1) or (m[i + 1][j - 1][k] == 1) \
                           or (m[i - 1][j + 1][k] == 1) or (m[i + 1][j + 1][k] == 1)
                    if not good:
                        m[i][j][k].active = False
    pt.store(f'{glob_i}_pt_filter.dat', is_store_only_leafs=True)

    # Classify cells.
    print(glob_i, 'classify cells')
    mesh.paint_faces(-1)
    for i in range(s):
        for j in range(s):
            for k in range(s):
                if isinstance(m[i][j][k], geom.EnclosingParallelepipedsTree):
                    if m[i][j][k].active:
                        tc.paint_triangles_intersects_with_box(m[i][j][k].box, 0)
    interse = mesh.paint_intersecting_faces(1)
    print(glob_i, f'interse {interse}')
    if interse == 0:
        return False
    mesh.store(f'{glob_i}_mesh_class.dat')

    # Reduce
    i = 100
    while True:
        es = []
        for e in mesh.edges:
            if e.is_border():
                if any((f['M'] == -1) or (f['M'] == 1) for f in e.faces):
                    es.append(e)
        es.sort(key=lambda e: e.length())
        print(f'{len(es)} painted border edges')
        if len(es) == 0:
            break
        for e in es:
            if e in mesh.edges:
                mesh.reduce_edge(e)
        mesh.store(f'{glob_i}_{i}_reduce.dat')
        i = i + 1

    return True
# --------------------------------------------------------------------------------------------------

def reduce_with_underlying_mesh(mesh):
    """
    Reduce with underlying mesh.

    Parameters
    ----------
    mesh : msu.Mesh
        Mesh.
    """

    glob_i = 100
    while reduce_with_underlying_mesh_step(mesh, glob_i):
        glob_i = glob_i + 1

# ==================================================================================================

if __name__ == '__main__':
    mesh = msu.Mesh('../cases/cylinder/cyl_100.dat')

    reduce_with_underlying_mesh(mesh)

    # Restore cavern.
    #i = 100
    #while True:
    #
    #    # paint intersection
    #    m.paint_faces(0)
    #    cnt = m.paint_intersecting_faces(1)
    #    print(f'{cnt} faces painted')
    #    m.store(f'{i}_01_painted_intersection.dat')
    #    if cnt == 0:
    #        break
    #
    #    # reduce
    #    es = []
    #    for e in m.edges:
    #        if e.is_border():
    #            if any(f['M'] == 1 for f in e.faces):
    #                es.append(e)
    #    es.sort(key=lambda e: e.length())
    #    print(f'{len(es)} painted border edges')
    #    for e in es:
    #        if e in m.edges:
    #            print(f'edge : {e.length()}')
    #            m.reduce_edge(e)
    #
    #    i = i + 1

    # Underlying mesh good cells detection.

# ==================================================================================================
