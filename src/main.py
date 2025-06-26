import msu
import geom
import numpy as np
import random

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    m = msu.Mesh('../cases/bunny/bunny.dat')
    i = 10000
    m.store(f'res_{i}.dat')
    while m.edges:
        i = i + 1
        m.reduce_edge(m.edges[-1])
        print(f'ready to store {i} | {len(m.nodes)} {len(m.edges)} {len(m.faces)}')
        m.store(f'res_{i}.dat')

#---------------------------------------------------------------------------------------------------
