import msu

if __name__ == '__main__':
    case = '../cases/pseudogrids/ex3.dat'
    mesh = msu.Mesh()
    mesh.load(case)
    mesh.store('../res_meshhealer.dat')
