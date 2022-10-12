# Surface evolution.

import numpy as np

# Count of valuable digits (after dot) in node coordinates.
# If coordinates of nodes doesn't differ in valuable digits we consider them equal.
NODE_COORDINATES_VALUABLE_DIGITS_COUNT = 10


class Node:
    """
    Node.
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
        self.edges = []
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


class Grid:
    """
    Grid (Surface Unstructured).
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Constructor.
        """

        # Empty name.
        self.Name = ''

        # Set empty sets of nodes, faces, zones.
        self.Nodes = []
        self.Edges = []
        self.Faces = []
        self.Zones = []

        # Rounded coordinates
        self.RoundedCoordsBag = set()

        self.number_of_border_nodes = 0

    # ----------------------------------------------------------------------------------------------

    def clear(self):
        """
        Clear grid.
        """

        self.Nodes.clear()
        self.Edges.clear()
        self.Faces.clear()
        self.Zones.clear()
        self.RoundedCoordsBag.clear()

    # ----------------------------------------------------------------------------------------------

    def zones_count(self):
        """
        Get zones count.

        :return: zones count
        """

        return len(self.Zones)

    # ----------------------------------------------------------------------------------------------

    def nodes_count(self):
        """
        Get count of nodes.

        :return: nodes count
        """

        return len(self.Nodes)

    # ----------------------------------------------------------------------------------------------

    def edges_count(self):
        """
        Get count of edges.

        :return: edges count
        """

        return len(self.Edges)

    # ----------------------------------------------------------------------------------------------

    def faces_count(self):
        """
        Get count of faces.

        :return: faces count.
        """

        return len(self.Faces)

    # ----------------------------------------------------------------------------------------------

    def faces_count_in_zones(self):
        """
        Count of faces that are placed in zones.

        :return: total faces count in zones
        """

        return sum([z.faces_count() for z in self.Zones])

    # ----------------------------------------------------------------------------------------------

    def find_near_node(self, n):
        """
        Find in grid nodes collection node that is near to a given node.

        :param n: node to check
        :return: near node from grid nodes collection
        """

        # First try to find  in bag.
        if n.rounded_coordinates() in self.RoundedCoordsBag:
            for node in self.Nodes:
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
            self.Nodes.append(n)
            self.RoundedCoordsBag.add(n.rounded_coordinates())
            if zone:
                zone.add_node(n)
            return n
        else:
            # There is already such a node in the grid.
            # Just return it.
            return found_node

    # ----------------------------------------------------------------------------------------------

    def add_edge(self, e):

        self.Edges.append(e)

        return e

    # ----------------------------------------------------------------------------------------------

    def add_face(self, f):
        """
        Add face.

        :param f: face
        :param global_id: int - global id from face in grid
        :return: added face
        """


        self.Faces.append(f)

        return f

    # ----------------------------------------------------------------------------------------------

    def set_zones_ids(self):
        """
        Set zones identifiers.
        """

        for (i, z) in enumerate(self.Zones):
            z.Id = i

    # ----------------------------------------------------------------------------------------------

    def reset_zones_ids(self):
        """
        Reset zones identifiers.
        """

        for z in self.Zones:
            z.Id = -1

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def link_node_edge(node, edge):
        """
        Link node with edge.

        :param node: node
        :param edge: edge
        """

        # проверка на правьность передаваемых типов объектов
        assert (type(node) is Node)
        assert (type(edge) is Edge)

        # проверка на добавление дублей
        assert not edge in node.edges
        assert not node in edge.Nodes

        node.edges.append(edge)
        edge.Nodes.append(node)

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
        face.Nodes.append(node)

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

        edge.Faces.append(face)
        face.Edges.append(edge)

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def find_edge(node_a, node_b):
        """
        Find edge with given nodes.

        :param node_a: the first node
        :param node_b: the second node
        :return: edge - if it is found, None - otherwise
        """

        for edge in node_a.edges:
            if node_b in edge.Nodes:
                return edge

        return None

    # ----------------------------------------------------------------------------------------------

    def complex_link_face_node_node_edge(self, face, node_a, node_b):
        """
        Complex link nodes with edge, and edge with face.

        :param face: face
        :param node_a: the first node
        :param node_b: th second node
        """

        # First we need to find edge.
        edge = Grid.find_edge(node_a, node_b)

        if edge is None:
            # New edge and link it.
            edge = Edge()
            self.add_edge(edge)
            Grid.link_node_edge(node_a, edge)
            Grid.link_node_edge(node_b, edge)
            Grid.link_edge_face(edge, face)
        else:
            # Edge is already linked with nodes.
            # Link only with the face.
            Grid.link_edge_face(edge, face)

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
                    self.Zones.append(zone)

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
                        Grid.link_node_face(nodes[0], face)
                        Grid.link_node_face(nodes[1], face)
                        Grid.link_node_face(nodes[2], face)

                else:
                    raise Exception('Unexpected line : {0}.'.format(line))

                line = f.readline()
            f.close()

            # Now we need to fix rest objects links.
            for face in self.Faces:
                node_a = face.Nodes[0]
                node_b = face.Nodes[1]
                node_c = face.Nodes[2]
                self.complex_link_face_node_node_edge(face, node_a, node_b)
                self.complex_link_face_node_node_edge(face, node_a, node_c)
                self.complex_link_face_node_node_edge(face, node_b, node_c)

            # Relink.
            self.link_nodes_and_edges_to_zones()

    # ----------------------------------------------------------------------------------------------

    def store(self, filename, face_variables=None):
        """
        Store grid to file.

        :param filename: file name
        """

        # Если не подан параметр с перечислением полей ячеек для экспорта, то экспортируем все поля.
        if face_variables is None:
            face_variables = list(self.Faces[0].Data.keys())
        variables = ['X', 'Y', 'Z'] + face_variables

        with open(filename, 'w', newline='\n') as f:

            # Store head.
            f.write('# crys-gsu\n')
            # if self.Name != '':
            # TITLE is not needed if empty, nut remesher craches if there is no title.
            f.write('TITLE="{0}"\n'.format(self.Name))
            f.write('VARIABLES={0}\n'.format(', '.join(['"{0}"'.format(k) for k in variables])))


            # Store zones.
            for zone in self.Zones:

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
                    f.write(' '.join([str(zone.Nodes.index(n) + 1) for n in face.Nodes]) + '\n')

            f.close()

    # ----------------------------------------------------------------------------------------------

    def link_nodes_and_edges_to_zones(self):
        """
        Link nodes and edges to zones.
        """

        # Clear old information.
        for z in self.Zones:
            z.Nodes.clear()
            z.Edges.clear()

        self.set_zones_ids()

        # Add nodes.
        for n in self.Nodes:
            zids = list(set([f.Zone.Id for f in n.faces]))
            for zid in zids:
                self.Zones[zid].add_node(n)

        # Add edges.
        for e in self.Edges:
            zids = list(set([f.Zone.Id for f in e.Faces]))
            for zid in zids:
                self.Zones[zid].add_edge(e)

        self.reset_zones_ids()

    # ----------------------------------------------------------------------------------------------

    def unlink_faces_from_zones(self):
        """
        Unlink faces from zones.
        """

        for face in self.Faces:
            if not face.Zone.IsFixed:
                face.Zone = None

    # ----------------------------------------------------------------------------------------------

    def check_faces_are_linked_to_zones(self):
        """
        Check if all faces are linked to zones.
        """

        for face in self.Faces:
            if face.Zone is None:
                raise Exception('Unlinked face detected.')

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

        # If face is already link to some zone,
        # we have to unlink it first.
        if f.Zone is not None:
            f.unlink_from_zone()

        # Just add and set link to the zone.
        f.Zone = self
        self.Faces.append(f)

        return f

    # ----------------------------------------------------------------------------------------------

    def get_real_face(self, e):
        """Get real faces connected to the edge.

        Parameters
        ----------
          e : Edge
            Edge.

        Returns
        -------
          f : [Face]
            incident face that belongs to the zone
        """
        assert len(e.Faces) == 2
        if e.Faces[0].Zone == self:
            return e.Faces[0]
        elif e.Faces[1].Zone == self:
            return e.Faces[1]
        else:
            raise RuntimeError()

    # ---------------------------------------------------------------------------------------------

    def get_ghost_face(self, e):
        """Get ghost faces connected to the edge.

        Parameters
        ----------
          e : Edge
            Edge.

        Returns
        -------
          f : [Face]
            incident faces that belongs to the zone
        """
        assert len(e.Faces) == 2
        if e.Faces[0].Zone != self:
            return e.Faces[0]
        elif e.Faces[1].Zone != self:
            return e.Faces[1]
        else:
            raise RuntimeError()


# ==================================================================================================

class Face:
    """
    Face of the grid.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, variables, values):
        """
        Constructor face.

        :param variables: Variables list.
        :param values:    Values list.
        """

        # Create face data as a dictionary.
        self.Data = dict(zip(variables, values))

        # Links with nodes and edges.
        self.Nodes = []
        self.Edges = []

        # Link to zone (each face belongs only to one single zone).
        self.Zone = None

    # ----------------------------------------------------------------------------------------------

    def __getitem__(self, item):
        """
        Get data field.

        :param item: Variable name.
        :return:     Value.
        """

        return self.Data.get(item, 0.0)

    # ----------------------------------------------------------------------------------------------

    def __setitem__(self, key, value):
        """
        Set data field.

        :param key:   Key value.
        :param value: Value.
        """

        self.Data[key] = value

    # ----------------------------------------------------------------------------------------------

    def get_neighbour(self, edge):
        """
        Get neighbour through edge.

        :param edge: Edge.
        :return:     Neighbour.
        """

        incident_faces = len(edge.Faces)

        if incident_faces == 1:
            if edge.Faces[0] != self:
                raise Exception('Error while getting face neighbour.')
            return None
        elif incident_faces == 2:
            if edge.Faces[0] == self:
                return edge.Faces[1]
            elif edge.Faces[1] == self:
                return edge.Faces[0]
            else:
                raise Exception('Error while getting face neighbour.')
        else:
            raise Exception('Wrong edge incident faces ({0}).'.format(incident_faces))

    # ----------------------------------------------------------------------------------------------

    def get_all_neighbours_by_nodes(self):
        """Получение списка всех соседей через узлы.

        Returns
        -------
        [Face]
            Список ячеек.
        """

        li = []
        for n in self.Nodes:
            for f in n.Faces:
                if f != self:
                    if not f in li:
                        li.append(f)

        return li

    # ----------------------------------------------------------------------------------------------

    def unlink_from_zone(self):
        """
        Unlink face from zone.
        """

        if self.Zone is None:
            return

        # Face is linked.
        # Unlink it.
        self.Zone.Faces.remove(self)
        self.Zone = None

    # ----------------------------------------------------------------------------------------------

    def link_face_with_nodes_and_edges(self, *args):
        """Link face with nodes and edges.

        Parameters
        ----------
            args: List of Edges or Nodes

        Returns
        -------
            Link this Face with Args

        """

        if args:
            for arg in args:
                if type(arg) is Edge:
                    arg.Faces.append(self)
                    self.Edges.append(arg)
                elif type(arg) is Node:
                    arg.Faces.append(self)
                    self.Nodes.append(arg)
                else:
                    raise Exception(f'Error: the type of arg must be Node or Edge, but not your type - {type(arg)}')
            return True
        return False

    # ----------------------------------------------------------------------------------------------

    def find_edge(self, p1, p2):
        """
        Find edge with given nodes.

        :param p1: the first node
        :param p2: the second node
        :return: edge - if it is found, None - otherwise
        """

        for edge in self.Edges:
            if p1 in edge.Nodes and p2 in edge.Nodes:
                return edge

        return None

# ==================================================================================================


class Edge:
    """
    Edge of the grid.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Constructor.
        """

        # Links to nodes and faces.
        self.Nodes = []
        self.Faces = []


if __name__ == '__main__':
    print('surface-evolution')
    g = Grid()
    g.load('../cases/naca/naca_mz.dat')
    g.store('../garbage.dat')
    pass
