# Surface evolution.

import numpy as np

# Count of valuable digits (after dot) in node coordinates.
# If coordinates of nodes doesn't differ in valuable digits we consider them equal.
NODE_COORDINATES_VALUABLE_DIGITS_COUNT = 10


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

    # ----------------------------------------------------------------------------------------------

    def find_near_node(self, n):
        """
        Find in grid nodes collection node that is near to a given node.

        :param n: node to check
        :return: near node from grid nodes collection
        """

        # First try to find  in bag.
        if n.rounded_coordinates() in self.rounded_coordinates_bag:
            for node in self.nodes:
                if n.rounded_coordinates() == node.rounded_coordinates():
                    return node
            raise Exception('We expect to find node ' \
                            'with coordinates {0} in the grid'.format(n.rounded_coordinates()))

        return None

    # ----------------------------------------------------------------------------------------------

    def add_node(self, n, is_merge_same_nodes=True, zone=None):
        """
        Add node.

        :param n: node
        :param is_merge_same_nodes: flag merge same nodes
        :param zone: zone
        :return: node registered in self.Nodes
        """

        found_node = self.find_near_node(n)

        if (found_node is None) or (not is_merge_same_nodes):
            # There is no such node in the grid.
            # We have to add it.
            self.nodes.append(n)
            self.rounded_coordinates_bag.add(n.rounded_coordinates())
            if zone:
                zone.add_node(n)
            return n
        else:
            # There is already such a node in the grid.
            # Just return it.
            return found_node

    # ----------------------------------------------------------------------------------------------

    def add_face(self, f):
        """
        Add face.

        :param f: face
        :param global_id: int - global id from face in grid
        :return: added face
        """


        self.faces.append(f)

        return f

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def link_node_face(node, face):
        """
        Link face with node.

        :param node: node
        :param face: face
        """

        assert (type(node) is Node)
        assert (type(face) is Face)

        node.faces.append(face)
        face.nodes.append(node)

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def link_edge_face(edge, face):
        """
        Link edge with face.

        :param edge: edge
        :param face: face
        """

        assert (type(edge) is Edge)
        assert (type(face) is Face)

        edge.faces.append(face)
        face.edges.append(edge)

    # ----------------------------------------------------------------------------------------------

    def load(self, filename,
             is_merge_same_nodes=True):
        """
        Load grid from file.

        :param filename: file name
        :param is_merge_same_nodes: merge same nodes
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
                    # Comment.
                    pass
                elif 'TITLE=' in line:
                    self.Name = line.split('=')[1][1:-2]
                elif 'VARIABLES=' in line:
                    variables_str = line.split('=')[-1][:-1]
                    variables = variables_str.replace('"', '').replace(',', '').split()
                    face_variables = variables[3:]
                    face_variables_count = len(face_variables)

                elif 'ZONE T=' in line:

                    # Create new zone.
                    zone_name = line.split('=')[-1][1:-2]
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
                    nodes_to_read = int(nodes_line.split('=')[-1][:-1])
                    # print('LOAD: zone {0}, nodes_to_read = {1}'.format(zone_name, nodes_to_read))
                    faces_to_read = int(faces_line.split('=')[-1][:-1])

                    # Read data for nodes.
                    c = []
                    for i in range(3):
                        line = f.readline()
                        c.append([float(xi) for xi in line.split()])
                    for i in range(nodes_to_read):
                        p = np.array([c[0][i], c[1][i], c[2][i]])
                        node = Node(p)
                        node = self.add_node(node, is_merge_same_nodes)
                        zone.add_node(node)

                    # Read data for faces.
                    d = []
                    for i in range(face_variables_count):
                        line = f.readline()
                        d.append([float(xi) for xi in line.split()])
                    for i in range(faces_to_read):
                        face = Face(face_variables,
                                    [d[j][i] for j in range(face_variables_count)])
                        self.add_face(face)
                        zone.add_face(face)

                    # Read connectivity lists.
                    for i in range(faces_to_read):
                        line = f.readline()
                        face = zone.Faces[i]
                        nodes = [zone.Nodes[int(ss) - 1] for ss in line.split()]
                        if len(nodes) != 3:
                            raise Exception('Wrong count of ' \
                                            'face linked nodes ({0}).'.format(len(nodes)))
                        Mesh.link_node_face(nodes[0], face)
                        Mesh.link_node_face(nodes[1], face)
                        Mesh.link_node_face(nodes[2], face)

                else:
                    raise Exception('Unexpected line : {0}.'.format(line))

                line = f.readline()
            f.close()

            # Now we need to fix rest objects links.
            for face in self.faces:
                node_a = face.nodes[0]
                node_b = face.nodes[1]
                node_c = face.nodes[2]

    # ----------------------------------------------------------------------------------------------

    def store(self, filename, face_variables=None):
        """
        Store grid to file.

        :param filename: file name
        """

        # Если не подан параметр с перечислением полей ячеек для экспорта, то экспортируем все поля.
        if face_variables is None:
            face_variables = list(self.faces[0].data.keys())
        variables = ['X', 'Y', 'Z'] + face_variables

        with open(filename, 'w', newline='\n') as f:

            # Store head.
            f.write('# crys-gsu\n')
            # if self.Name != '':
            # TITLE is not needed if empty, nut remesher craches if there is no title.
            f.write('TITLE="{0}"\n'.format(self.Name))
            f.write('VARIABLES={0}\n'.format(', '.join(['"{0}"'.format(k) for k in variables])))


            # Store zones.
            for zone in self.zones:

                # Store zone head.
                f.write('ZONE T="{0}"\n'.format(zone.Name))
                f.write('NODES={0}\n'.format(len(zone.Nodes)))
                f.write('ELEMENTS={0}\n'.format(len(zone.Faces)))
                f.write('DATAPACKING=BLOCK\n')
                f.write('ZONETYPE=FETRIANGLE\n')
                f.write('VARLOCATION=([4-{0}]=CELLCENTERED)\n'.format(len(variables)))

                # Write first 3 data items (X, Y, Z coordinates).
                for i in range(3):
                    f.write(zone.get_nodes_coord_slice_str(i) + ' \n')

                # Write rest faces data items.
                for e in variables[3:]:
                    f.write(zone.get_faces_data_slice_str(e) + ' \n')

                # Write connectivity lists.
                for face in zone.Faces:
                    f.write(' '.join([str(zone.Nodes.index(n) + 1) for n in face.nodes]) + '\n')

            f.close()



# ==================================================================================================

class Zone:
    """
    Zone of the grid.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, name):
        """
        Constructor.

        :param name: Name of zone.
        """

        self.Id = -1

        self.Name = name

        # No nodes or faces in the zone yet.
        self.Nodes = []
        self.Edges = []
        self.Faces = []

    # ----------------------------------------------------------------------------------------------

    def nodes_count(self):
        """
        Get count of nodes.

        :return: Nodes count.
        """

        return len(self.Nodes)

    # ----------------------------------------------------------------------------------------------

    def edges_count(self):
        """
        Get count of edges.

        :return: Edges count.
        """

        return len(self.Edges)

    # ----------------------------------------------------------------------------------------------

    def outer_edges_count(self):
        """
        Get count of outer edges.

        :return: Outer edges count.
        """

        return len([e for e in self.Edges if e.is_outer()])

    # ----------------------------------------------------------------------------------------------

    def faces_count(self):
        """
        Get count of faces.

        :return: Faces count.
        """

        return len(self.Faces)

    # ----------------------------------------------------------------------------------------------

    def get_nodes_coord_slice_str(self, i):
        """
        Get string composed from i-th coord of all nodes.

        :param i: Index of nodes coord.
        :return:  Composed string.
        """

        i_list = ['{0:.18e}'.format(node.p[i]) for node in self.Nodes]
        i_str = ' '.join(i_list)

        return i_str

    # ----------------------------------------------------------------------------------------------

    def get_faces_data_slice_str(self, e):
        """
        Get string composed from elements of data of all faces.

        :param e: Data element.
        :return:  Composed string.
        """

        e_list = ['{0:.18e}'.format(face[e]) for face in self.Faces]
        e_str = ' '.join(e_list)

        return e_str

    # ----------------------------------------------------------------------------------------------

    def add_node(self, n):
        """
        Add node to zone.

        :param n: Node.
        :return:  Added node.
        """

        # Just add node.
        self.Nodes.append(n)

        return n

    # ----------------------------------------------------------------------------------------------

    def add_edge(self, e):
        """
        Add edge to zone.

        :param e: Edge.
        :return:  Added edge.
        """

        # Just add egde.
        self.Edges.append(e)

        return e

    # ----------------------------------------------------------------------------------------------

    def add_face(self, f):
        """
        Add face to zone (with link).

        :param f: Face.
        :return:  Added face.
        """

        # Just add and set link to the zone.
        f.Zone = self
        self.Faces.append(f)

        return f


if __name__ == '__main__':
    print('surface-evolution')
    g = Mesh()
    g.load('../cases/naca/naca_mz.dat')
    g.store('../garbage.dat')
    pass
