import numpy
import trimesh
import argparse
import sys


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-n', '--stl_name')
    parser.add_argument('-d', '--dat_name', default="result.dat")
    parser.add_argument('-p', '--ice_percent', default=0.0)
    return parser

def print_row_in_file(f, x, cnt):
    s = ''
    for j in range(cnt):
        s += f'{x} '
    f.write(f'{s}\n')

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    stl_name = namespace.stl_name
    dat_name = namespace.dat_name
    ice_percent = numpy.float64(namespace.ice_percent)
    stl_mesh = trimesh.load_mesh(stl_name)
    with open(dat_name, 'w', newline='\n') as f:
        f.write('#\n')
        f.write('TITLE=""\n')
        f.write('VARIABLES="X", "Y", "Z", "T", "Hw", "Hi", "HTC", "Beta", "TauX", "TauY", "TauZ"\n')
        f.write('ZONE T="ZONE 1"\n')
        f.write(f'NODES={len(stl_mesh.vertices)}\n')
        f.write(f'ELEMENTS={len(stl_mesh.faces)}\n')
        f.write('DATAPACKING=BLOCK\n')
        f.write('ZONETYPE=FETRIANGLE\n')
        f.write('VARLOCATION=([4-11]=CELLCENTERED)\n')
        for i in range(3):
            s = ''
            for j in range(len(stl_mesh.vertices)):
                s += f'{stl_mesh.vertices[j][i]} '
            f.write(f'{s}\n')
        for i in range(2):
            print_row_in_file(f, 0.0, len(stl_mesh.faces))
        print_row_in_file(f, stl_mesh.volume*ice_percent, len(stl_mesh.faces))
        for i in range(5):
            print_row_in_file(f, 0.0, len(stl_mesh.faces))
        for face in stl_mesh.faces:
            s = ''
            for i in range(3):
                s += f'{face[i]+1} '
            f.write(f'{s}\n')
