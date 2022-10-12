# Surface evolution.

import numpy as np

# Count of valuable digits (after dot) in node coordinates.
# If coordinates of nodes doesn't differ in valuable digits we consider them equal.
NODE_COORDINATES_VALUABLE_DIGITS_COUNT = 10

# String of export.
EXPORT_FORMAT_STRING = '{0:.18e}'


class Node:
    """
    Node - container for coordinates.
    """

    def __init__(self, p):
        """
        Initialization.
        Node may appear only as point holder.

        Parameters
        ----------
        p : np.array
            Point coordinates.
        """

        self.p = p
        self.faces = []

    def rounded_coordinates(self):
        """
        Tuple with rounded coordinates.

        Returns
        -------
        tuple
            Rounded coordinates.
        """

        return tuple(map(lambda x: round(x, NODE_COORDINATES_VALUABLE_DIGITS_COUNT), self.p))


class Face:
    """
    Face - container for physical data.
    """

    def __init__(self, variables, values):
        """
        Initialization.

        Parameters
        ----------
        variables : list(str)
            List of variables names.
        values : list
            List of values.
        """

        self.data = dict(zip(variables, values))
        self.nodes = []

    def __getitem__(self, item):
        """
        Get face data element.

        Parameters
        ----------
        item : str
            Name of data element.

        Returns
        -------
        value
            Value of data element.
        """

        return self.data.get(item, 0.0)

    def __setitem__(self, key, value):
        """
        Set data element.

        Parameters
        ----------
        key : str
            Name of data element.
        value
            Value of data element.
        """

        self.data[key] = value


class Zone:
    """
    Zone - set of faces.
    """

    def __init__(self, name):
        """
        Initialization.

        Parameters
        ----------
        name : str
            Name of zone.
        """

        self.name = name
        self.nodes = []
        self.faces = []

    @staticmethod
    def objects_slice_str(fun, obs):
        """
        String, that contains data slice for some objects.
        Formatting is defined here.

        Parameters
        ----------
        fun
            Function for data extracting.
        obs
            Objects list.

        Returns
        -------
        str
            String with data slice.
        """

        return ' '.join(map(lambda ob: EXPORT_FORMAT_STRING.format(fun(ob)), obs))

    def nodes_coordinate_slice_str(self, i):
        """
        String, that contains i-th coordinate of all nodes.

        Parameters
        ----------
        i : int
            Coordinate index.

        Returns
        -------
        str
            String with coordinate slice.

        """

        return Zone.objects_slice_str(lambda n: n.p[i], self.nodes)

    def faces_data_element_slice_str(self, e):
        """
        String, that contains data element for all faces.

        Parameters
        ----------
        e : str
            Name of data element.

        Returns
        -------
        str
            String with data element slice.
        """

        return Zone.objects_slice_str(lambda f: f[e], self.faces)


class Mesh:
    """
    Mesh - consists of surface triangle faces.
    """

    def __init__(self):
        """
        Initialization.
        """

        # Comment and title - save for store.
        self.comment = ''
        self.title = ''

        # Set empty sets of nodes, faces, zones.
        self.zones = []
        self.nodes = []
        self.faces = []

        # Rounded coordinates bag.
        self.rounded_coordinates_bag = set()

    def clear(self):
        """
        Clear all.
        """

        self.comment = ''
        self.title = ''
        self.nodes.clear()
        self.faces.clear()
        self.zones.clear()
        self.rounded_coordinates_bag.clear()

    def find_near_node(self, node):
        """
        Try to find node near to given node.

        Parameters
        ----------
        node : Node
            Given node.

        Returns
        -------
        Node or None
            If node is found, return it, otherwise return None.
        """

        rc = node.rounded_coordinates()

        # Try to find in bag.
        if rc not in self.rounded_coordinates_bag:
            return None

        # Node rounded coordinates is in bag, find it.
        for n in self.nodes:
            if rc == n.rounded_coordinates():
                return n

        raise Exception('Internal error')

    def add_node(self, node):
        """
        Add node to mesh.

        Parameters
        ----------
        node : Node
            Node to add.

        Returns
        -------
            If new node is added - return this node,
            otherwise - return existed node.
        """

        found_node = self.find_near_node(node)

        if found_node is None:
            self.nodes.append(node)
            self.rounded_coordinates_bag.add(node.rounded_coordinates())
            return node
        else:
            return found_node

    def load(self, filename):
        """
        Load mesh.

        Parameters
        ----------
        filename : str
            Name of file.
        """

        variables = []
        face_variables = []
        face_variables_count = 0

        # Clear all objects of the grid.
        self.clear()

        # Open file and try to load it line by line.
        with open(filename, 'r') as f:
            line = f.readline()
            while line:

                if line[0] == '#':

                    # Comment, save it.
                    self.comment = line[1:-1]

                elif 'TITLE=' in line:

                    # Title, save it.
                    self.title = line.split('=')[1][1:-2]

                elif 'VARIABLES=' in line:

                    # Variables.
                    variables_str = line.split('=')[1][:-1]
                    variables = variables_str.replace('"', '').replace(',', '').split()
                    face_variables = variables[3:]
                    face_variables_count = len(face_variables)

                elif 'ZONE T=' in line:

                    # New zone.
                    zone_name = line.split('=')[1][1:-2]
                    zone = Zone(zone_name)
                    self.zones.append(zone)

                    # Read count of nodes and faces to read.
                    nodes_line = f.readline()
                    faces_line = f.readline()
                    packing_line = f.readline()
                    zonetype_line = f.readline()
                    varlocation_line = f.readline()
                    if 'NODES=' not in nodes_line:
                        raise Exception('Wrong nodes line ({0}).'.format(nodes_line))
                    if 'ELEMENTS=' not in faces_line:
                        raise Exception('Wrong faces line ({0}).'.format(faces_line))
                    if 'DATAPACKING=BLOCK' != packing_line[:-1]:
                        raise Exception('Wrong packing line ({0}).'.format(packing_line))
                    if 'ZONETYPE=FETRIANGLE' != zonetype_line[:-1]:
                        raise Exception('Wrong zonetype line ({0}).'.format(zonetype_line))
                    right_varlocation_line = 'VARLOCATION=' \
                                             '([4-{0}]=CELLCENTERED)'.format(len(variables))
                    if right_varlocation_line != varlocation_line[:-1]:
                        raise Exception('Wrong varlocation line ({0}). '
                                        'Right value is {1}'.format(varlocation_line,
                                                                    right_varlocation_line))
                    nodes_to_read = int(nodes_line.split('=')[1][:-1])
                    faces_to_read = int(faces_line.split('=')[1][:-1])

                    # Read data for nodes.
                    c = []
                    for i in range(3):
                        line = f.readline()
                        c.append([float(xi) for xi in line.split()])
                    for i in range(nodes_to_read):
                        node = Node(np.array([c[0][i], c[1][i], c[2][i]]))
                        node = self.add_node(node)
                        zone.nodes.append(node)

                    # Read data for faces.
                    d = []
                    for i in range(face_variables_count):
                        line = f.readline()
                        d.append([float(xi) for xi in line.split()])
                    for i in range(faces_to_read):
                        face = Face(face_variables,
                                    [d[j][i] for j in range(face_variables_count)])
                        self.faces.append(face)
                        zone.faces.append(face)

                    # Read connectivity lists.
                    for i in range(faces_to_read):
                        line = f.readline()
                        face = zone.faces[i]
                        nodes = [zone.nodes[int(ss) - 1] for ss in line.split()]
                        if len(nodes) != 3:
                            raise Exception('Internal error')
                        face.nodes = nodes
                        for n in nodes:
                            n.faces.append(face)
                else:
                    raise Exception('Unexpected line : {0}.'.format(line))

                line = f.readline()
            f.close()

    def store(self, filename):
        """
        Store mesh.

        Parameters
        ----------
        filename : str
            Name of file.
        """

        variables = ['X', 'Y', 'Z'] + list(self.faces[0].data.keys())

        with open(filename, 'w', newline='\n') as f:

            # Store head.
            f.write(f'#{self.comment}\n')
            f.write(f'TITLE="{self.title}"\n')
            f.write('VARIABLES={0}\n'.format(', '.join(['"{0}"'.format(k) for k in variables])))

            # Store zones.
            for zone in self.zones:

                # Store zone head.
                f.write(f'ZONE T="{zone.name}"\n')
                f.write(f'NODES={len(zone.nodes)}\n')
                f.write(f'ELEMENTS={len(zone.faces)}\n')
                f.write('DATAPACKING=BLOCK\n')
                f.write('ZONETYPE=FETRIANGLE\n')
                f.write(f'VARLOCATION=([4-{len(variables)}]=CELLCENTERED)\n')

                # Write first 3 data items (X, Y, Z coordinates).
                for i in range(3):
                    f.write(zone.nodes_coordinate_slice_str(i) + ' \n')

                # Write rest faces data items.
                for e in variables[3:]:
                    f.write(zone.faces_data_element_slice_str(e) + ' \n')

                # Write connectivity lists.
                for face in zone.faces:
                    f.write(' '.join([str(zone.nodes.index(n) + 1) for n in face.nodes]) + '\n')

            f.close()

    def remesh(self):
        """
        Remesh.

        sources:
            [1] X. Tong, D. Thompson, Q. Arnoldus, E. Collins, E. Luke.
                Three-Dimensional Surface Evolution and Mesh Deformation for Aircraft Icing Applications. //
                Journal of Aircraft, DOI: 10.2514/1.C033949
        """

        pass


if __name__ == '__main__':
    print('surface-evolution')
    g = Mesh()
    g.load('../cases/naca/naca_mz.dat')
    g.remesh()
    g.store('../garbage.dat')
