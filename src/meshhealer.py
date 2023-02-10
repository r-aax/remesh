import msu
import numpy as np

if __name__ == '__main__':
    case = '../cases/hex.dat'
    mesh = msu.Mesh()
    mesh.load(case)
    mesh.calculate_edges()
    print(mesh.faces)
    print(mesh.edges)
    #
    points = [np.array([3.0, -0.5, 1.0]), np.array([3.5, -0.95, 2.0])]
    for i in range(2):
        f = mesh.faces[10]
        print(f)
        e = mesh.edge_table[(f.nodes[0], f.nodes[1])]
    #mesh.split_face(f, point)
        mesh.split_edge(e, points[i])
        print(i)
        print(mesh.faces)
        print(mesh.edges)
    mesh.store('../res_meshhealer.dat')
