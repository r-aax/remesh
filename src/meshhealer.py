import msu
import numpy as np

if __name__ == '__main__':
    case = '../cases/hex.dat'
    mesh = msu.Mesh()
    mesh.load(case)
    mesh.calculate_edges()

    #
    f = mesh.faces[10]
    mesh.split_face(f, np.array([0.0, 0.0, 0.0]))
    mesh.calculate_edges()
    #

    mesh.store('../res_meshhealer.dat')
