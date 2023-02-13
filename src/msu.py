from typing import overload
import numpy as np
from numpy import linalg as LA
import mth
import geom

# Count of valuable digits (after dot) in node coordinates.
# If coordinates of nodes doesn't differ in valuable digits we consider them equal.
NODE_COORDINATES_VALUABLE_DIGITS_COUNT = 10

# String of export.
EXPORT_FORMAT_STRING = '{0:.18e}'


def find_common_nodes(face1, face2):
    """
    finds common nodes between faces and return them in sorted order

    Parameters
    ----------
    face1, face2: Face

    Returns
    -------
    Node, Node
    """
    nodes = []
    for n in face1.nodes:
        if n in face2.nodes:
            nodes.append(n)
    if nodes[0].glo_id < nodes[1].glo_id:
        return nodes[0], nodes[1]
    else:
        return nodes[1], nodes[0]


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

        # Global identifier.
        self.glo_id = -1

        self.p = p
        self.old_p = None
        self.faces = []
        self.A = None
        self.b = None
        # Direction for node moving (we call it normal).
        self.normal = None

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Node {self.glo_id}'

    def rounded_coordinates(self):
        """
        Tuple with rounded coordinates.

        Returns
        -------
        tuple
            Rounded coordinates.
        """

        return tuple(map(lambda x: round(x, NODE_COORDINATES_VALUABLE_DIGITS_COUNT), self.p))

    def is_isolated(self):
        """
        Check if node is isolated.

        Returns
        -------
        True - if node is isolated,
        False - if node is not isolated.
        """

        return len(self.faces) == 0


class Edge:
    """
    Edge - border between two faces
    """

    def __init__(self, face1, face2, node1=None, node2 = None):
        """
        Initialization.

        Parameters
        ----------
        face1: Face
            first face
        face2: Face
            second face
        """

        self.face1 = face1
        self.face2 = face2
        if face1 is not None and face2 is not None:
            self.node1, self.node2 = find_common_nodes(face1, face2)
        else:
            if face1 is None:
                self.face1, self.face2 = self.face2, self.face1
            if face1 is None:
                raise Exception('Edge(None, None)')
            self.node1, self.node2 = node1, node2

    def __eq__(self, other):
        return self.face1 == other.face1 and self.face2 == other.face2 \
                or self.face1 == other.face2 and self.face2 == other.face1

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Edge {self.node1.glo_id} - {self.node2.glo_id}'

    def points(self):
        return self.node1.p, self.node2.p

    def nodes(self):
        return self.node1, self.node2

    def old_points(self):
        return self.node1.old_p, self.node2.old_p

    def length(self):
        """
        Length of the edge.

        Returns
        -------
        float
            Length of the edge.
        """

        return LA.norm(self.node1.p - self.node2.p)

    def replace_face(self, f, new_f):
        if self.face1 == f:
            self.face1 = new_f
        elif self.face2 == f:
            self.face2 = new_f
        else:
            raise Exception('No such face')


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

        # Global identifier.
        self.glo_id = -1

        self.data = dict(zip(variables, values))
        self.nodes = []
        self.zone = None
        self.neighbour_faces = []
        # Area of the face.
        self.area = 0.0

        # Face normal and smoothed normal.
        self.normal = None
        self.smoothed_normal = None

        # Total ice volume to be accreted for this face.
        self.target_ice = 0.0

        # Ice chunk to be accreted on current iteration.
        self.ice_chunk = 0.0

        # H field.
        self.h = 0.0

        # Data about view of cubic function V(h) = ah + bh^2 + ch^3
        self.v_coef_a = 0.0
        self.v_coef_b = 0.0
        self.v_coef_c = 0.0

        # Jiao coeffs.
        self.jiao_coef_a = 0.0
        self.jiao_coef_b = 0.0
        self.jiao_coef_c = 0.0

        # Time step fraction.
        self.tsf = 0.0
        self.tsf_jiao = 0.0

        # Diverging or contracting face.
        self.is_contracting = False

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Face {self.glo_id} ({self.nodes[0].glo_id}, {self.nodes[1].glo_id}, {self.nodes[2].glo_id})'

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

    def copy(self):
        """
        Get copy of face.
        Copy with the same data.
        But copy doesn't contain links.

        Returns
        -------
        Face
            Copy.
        """

        return Face(self.data.keys(), self.data.values())

    def points(self):
        """
        Get points.

        Returns
        -------
        tuple
            Points.
        """

        return self.nodes[0].p, self.nodes[1].p, self.nodes[2].p

    def center(self):
        """
        Center point.

        Returns
        -------
        Point
            Point of center.
        """

        return (self.nodes[0].p + self.nodes[1].p + self.nodes[2].p) / 3.0

    def normals(self):
        """
        Get normals.

        Returns
        -------
        tuple
            Normals.
        """

        return self.nodes[0].normal, self.nodes[1].normal, self.nodes[2].normal

    def calculate_area(self):
        """
        Calculate area.
        """

        a, b, c = self.points()

        self.area = 0.5 * LA.norm(np.cross(b - a, c - b))

    def calculate_normal(self):
        """
        Calculate normal.
        """

        a, b, c = self.points()

        self.normal = np.cross(b - a, c - b)
        self.normal = self.normal / LA.norm(self.normal)
        self.smoothed_normal = self.normal.copy()

    def calculate_neighbour_faces(self):
        """
        calculate neighbours of the face
        """
        connected_faces = {}
        for n in self.nodes:
            for f in n.faces:
                if f not in connected_faces:
                    connected_faces[f] = 1
                else:
                    connected_faces[f] += 1
        self.neighbour_faces = [f for f in connected_faces if connected_faces[f] == 2]

    def inner_angle(self, n):
        """
        Get inner angle of the node.

        Parameters
        ----------
        n : Node
            Node.

        Returns
        -------
        float
            Angle in radians.
        """

        ns = list(filter(lambda ni: ni != n, self.nodes))
        v1, v2 = ns[0].p - n.p, ns[1].p - n.p

        # (a, b) = |a| * |b| * cos(alpha)
        return np.arccos(np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2)))


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

    ColorCommon = 0
    ColorToDelete = 1
    ColorBorder = 2
    ColorFree = 3

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
        self.edges = []
        self.edge_table = {}
        # Rounded coordinates bag.
        self.rounded_coordinates_bag = set()

        # Target ice in the beginning of remeshing.
        self.initial_target_ice = 0.0

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

    def max_node_glo_id(self):
        """
        Get maximum node global id
        (id of the last node).

        Returns
        -------
        int
            Maximum node global id,
            or -1, if there is no nodes.
        """

        if self.nodes:
            return self.nodes[-1].glo_id
        else:
            return -1

    def max_face_glo_id(self):
        """
        Get maximum face global id
        (id of the last face).

        Returns
        -------
        int
            Maximum face global if,
            or -1, if there is no faces.
        """

        if self.faces:
            return self.faces[-1].glo_id
        else:
            return -1

    def add_node(self, node, zone):
        """
        Add node to mesh.

        Parameters
        ----------
        node : Node
            Node to add.
        zone : Zone
            Zone to add node to.
        """

        found_node = self.find_near_node(node)

        if found_node is None:
            max_glo_id = self.max_node_glo_id()
            node.glo_id = max_glo_id + 1
            self.nodes.append(node)
            self.rounded_coordinates_bag.add(node.rounded_coordinates())
            node_to_zone = node
        else:
            node_to_zone = found_node

        zone.nodes.append(node_to_zone)

    def add_face(self, face, zone):
        """
        Add face to mesh.

        Parameters
        ----------
        face : Face
            Face to add.
        zone : Zone
            Zone to add to.
        """

        max_glo_id = self.max_face_glo_id()
        face.glo_id = max_glo_id + 1
        self.faces.append(face)
        zone.faces.append(face)
        face.zone = zone

    def add_edge(self, edge):
        """
        Add edge to mesh.

        Parameters
        ----------
        edge : Edge
            Edge to add.
        """

        self.edges.append(edge)
        self.edge_table[(edge.node1, edge.node2)] = edge


    def add_face_node_link(self, f, n):
        """
        Add face-node link.

        Parameters
        ----------
        f : Face
            Face.
        n : Node
            Node.
        """

        f.nodes.append(n)
        n.faces.append(f)

    def add_face_nodes_links(self, f, ns):
        """
        Add face-node links.

        Parameters
        ----------
        f : Face
            Face.
        ns : [Node]
            Nodes list.
        """

        for n in ns:
            self.add_face_node_link(f, n)

    def delete_face_node_link(self, f, n):
        """
        Delete face-node link.

        Parameters
        ----------
        f : Face
            Face.
        n : Node
            Node.
        """

        f.nodes.remove(n)
        n.faces.remove(f)

    def replace_face_node_link(self, f, n, new_n):
        """
        Replace node in face-node link.

        Parameters
        ----------
        f : Face
            Face.
        n : Node
            Old node.
        new_n : Node
            New node.
        """

        i = f.nodes.index(n)
        f.nodes[i] = new_n
        n.faces.remove(f)
        new_n.faces.append(f)

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
                        self.add_node(node, zone)

                    # Read data for faces.
                    d = []
                    for i in range(face_variables_count):
                        line = f.readline()
                        d.append([float(xi) for xi in line.split()])
                    for i in range(faces_to_read):
                        face = Face(face_variables,
                                    [d[j][i] for j in range(face_variables_count)])
                        self.add_face(face, zone)
                    # Read connectivity lists.
                    for i in range(faces_to_read):
                        line = f.readline()
                        face = zone.faces[i]
                        nodes = [zone.nodes[int(ss) - 1] for ss in line.split()]

                        if len(nodes) != 3:
                            raise Exception('Internal error')
                        for n in nodes:
                            self.add_face_node_link(face, n)
                else:
                    raise Exception('Unexpected line : {0}.'.format(line))

                line = f.readline()
            f.close()

        # Set identifiers.
        for i, f in enumerate(self.faces):
            f['Id'] = i
        for n in self.nodes:
            if len(n.faces) == 0:
                self.nodes.remove(n)

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

    def calculate_faces_areas(self):
        """
        Calculate faces areas.
        """

        for f in self.faces:
            f.calculate_area()

    def calculate_faces_normals(self):
        """
        Calculate faces normals.
        """

        for f in self.faces:
            f.calculate_normal()

    def calculate_edges(self):
        """
        calculate edges of the mesh
        """
        self.edges = []
        for f in self.faces:
            f.calculate_neighbour_faces()
            es = [Edge(f, neighbour) for neighbour in f.neighbour_faces if f.glo_id < neighbour.glo_id]
            for e in es:
                self.add_edge(e)
            if (len(f.neighbour_faces) < 3):
                remaining_nodes = sorted([n for n in f.nodes if len(set(n.faces) & set(f.neighbour_faces)) < 2], key=lambda n:n.glo_id)
                for i in range(3 - len(f.neighbour_faces)):
                    self.add_edge(Edge(f, None, remaining_nodes[i], remaining_nodes[i+1]))


    def target_ice(self):
        """
        Get sum targe ice.

        Returns
        -------
        float
            Target ice.
        """

        return sum(map(lambda f: f.target_ice, self.faces))

    def add_additional_data_for_analysis(self):
        """
        Add additional data for analysis.
        """

        # Additional data for analyzis.
        for f in self.faces:
            v = f.normal
            f['NX'] = v[0]
            f['NY'] = v[1]
            f['NZ'] = v[2]
            f['NMod'] = LA.norm(v)
            v = f.nodes[0].normal
            f['N1X'] = v[0]
            f['N1Y'] = v[1]
            f['N1Z'] = v[2]
            f['N1Mod'] = LA.norm(v)
            v = f.nodes[1].normal
            f['N2X'] = v[0]
            f['N2Y'] = v[1]
            f['N2Z'] = v[2]
            f['N2Mod'] = LA.norm(v)
            v = f.nodes[2].normal
            f['N3X'] = v[0]
            f['N3Y'] = v[1]
            f['N3Z'] = v[2]
            f['N3Mod'] = LA.norm(v)

    def calculate_nodes_normals(self):
        """
        Calculate nodes normals.
        """

        for n in self.nodes:
            n.normal = sum(map(lambda f: f.normal, n.faces)) / len(n.faces)

    def delete_face(self, f):
        """
        Delete face.

        Parameters
        ----------
        f : Face
            Face to delete.
        """

        # Remove from nodes.
        while f.nodes:
            self.delete_face_node_link(f, f.nodes[0])

        # Remove from zones.
        for z in self.zones:
            if f in z.faces:
                z.faces.remove(f)

        # Remove from mesh.
        self.faces.remove(f)

    def delete_node(self, n):
        """
        Delete node.

        Parameters
        ----------
        n : Node
            Node to be deleted.
        """

        # First we must delete all adjacent faces.
        while n.faces:
            self.delete_face(n.faces[0])

        # Remove node from zones.
        for z in self.zones:
            if n in z.nodes:
                z.nodes.remove(n)

        # Remove node from mesh.
        self.nodes.remove(n)

    def delete_isolated_nodes(self):
        """
        Delete isolated nodes.
        """

        nodes_to_delete = [n for n in self.nodes if n.is_isolated()]

        for n in nodes_to_delete:
            self.delete_node(n)

    def delete_edge(self, e):
        """
        Delete edge.

        Parameters
        ----------
        e : Edge
            Edge to be deleted.
        """
        # Remove node from mesh.
        self.edges.remove(e)
        del self.edge_table[(e.node1, e.node2)]

    def reduce_edge(self, e):
        """
        Reduce edge.

        Parameters
        ----------
        e : Edge
            Edge.
        """

        a, b = e.node1, e.node2
        a.p = 0.5 * (a.p + b.p)

        # Replace b node with a node in all faces.
        delete_faces = []
        tmp = [f for f in b.faces]
        for f in tmp:
            if f in a.faces:
                delete_faces.append(f)
            else:
                self.replace_face_node_link(f, b, a)

        # Delete extra node and faces.
        for f in delete_faces:
            self.delete_face(f)

        # We need no b node anymore if it is isolated.
        if not b.faces:
            self.delete_node(b)
        self.delete_edge(e)

    def split_edge(self, e, p):
        """
        Split edge with point.

        Parameters
        ----------
        e : Edge
            Edge to be splitted.
        p : Point
            Point for split.
        """

        n = Node(p)
        node1_pair = []
        node2_pair = []
        # We need split both faces for edge.
        for f in [e.face1, e.face2]:

            if f is None:
                node2_pair.append(None)
                node1_pair.append(None)
                continue

            # Data from face.
            a, b, c = f.nodes[0], f.nodes[1], f.nodes[2]
            sorted_nodes = sorted(f.nodes, key=lambda node: node.glo_id)
            a1, b1, c1 = sorted_nodes[0], sorted_nodes[1], sorted_nodes[2]
            f1, f2 = f.copy(), f.copy()
            z = f.zone

            # Add node.
            self.add_node(n, z)

            # Add new faces.
            self.add_face(f1, z)
            self.add_face_nodes_links(f1, [a, b, c])
            self.replace_face_node_link(f1, e.node1, n)
            node2_pair.append(f1)
            self.add_face(f2, z)
            self.add_face_nodes_links(f2, [a, b, c])
            self.replace_face_node_link(f2, e.node2, n)
            node1_pair.append(f2)
            for pair in [(a1,b1),(b1,c1),(a1,c1)]:
                if pair != e.nodes():
                    f_to_replace = f1 if e.node2 in pair else f2
                    self.edge_table[pair].replace_face(f, f_to_replace)
            self.add_edge(Edge(f1, f2))
            # Delete old face.
            self.delete_face(f)

        self.add_edge(Edge(node1_pair[0], node1_pair[1]))
        self.add_edge(Edge(node2_pair[0], node2_pair[1]))
        self.delete_edge(e)

    def split_face(self, f, p):
        """
        Split face with point.

        Parameters
        ----------
        f : Face
            Face to be splitted.
        p : Point
            Point for spllit.
        """

        # Data from old face.
        a, b, c = f.nodes[0], f.nodes[1], f.nodes[2]
        f1, f2, f3 = f.copy(), f.copy(), f.copy()
        z = f.zone

        # New node.
        n = Node(p)
        self.add_node(n, z)
        print(n.glo_id)
        # Delete old face.
        self.delete_face(f)

        # Add new faces.
        self.add_face(f1, z)
        self.add_face_nodes_links(f1, [a, b, n])
        self.add_face(f2, z)
        self.add_face_nodes_links(f2, [b, c, n])
        self.add_face(f3, z)
        self.add_face_nodes_links(f3, [c, a, n])

        self.add_edge(Edge(f1, f2))
        self.add_edge(Edge(f1, f3))
        self.add_edge(Edge(f2, f3))
        pair1 = (a, b) if a.glo_id < b.glo_id else (b, a)
        pair2 = (b, c) if b.glo_id < c.glo_id else (c, b)
        pair3 = (a, c) if a.glo_id < c.glo_id else (c, a)
        self.edge_table[pair1].replace_face(f, f1)
        self.edge_table[pair2].replace_face(f, f2)
        self.edge_table[pair3].replace_face(f, f3)

    def parallel_move(self, v):
        """
        Parallel move all nodes.

        Parameters
        ----------
        v : Vector
            Move vector.
        """

        for n in self.nodes:
            n.p += v

    def unite_with(self, m):
        """
        Unite with another mesh.

        Parameters
        ----------
        m : Mesh
            Mesh.
        """

        # Create new zone and add united mesh to it.
        z = Zone('unite')
        self.zones.append(z)

        # Add nodes and faces to new zone.
        for n in m.nodes:
            self.add_node(n, z)
        for f in m.faces:
            self.add_face(f, z)

    def triangles_list(self):
        """
        Construct triangles list.

        Returns
        -------
        [Triangle]
            Triangles list.
        """

        return [geom.Triangle(f.nodes[0].p, f.nodes[1].p, f.nodes[2].p, f) for f in self.faces]

    def delete_self_intersected_faces(self):
        """
        Delete all self-intersected faces.

        We process the following marking:
          0 - common face,
          1 - face to delete,
          2 - face, adjacent to deleted,
        """

        # Find self-intersected faces.
        tc = geom.TrianglesCloud(self.triangles_list())
        pairs = tc.intersection_with_triangles_cloud(tc)

        #
        # Mark faces
        #

        # First all faces are common.
        for f in self.faces:
            f['M'] = Mesh.ColorCommon

        # If face intersects any - mark it in 1.
        for p in pairs:
            for t in p:
                t.back_ref['M'] = Mesh.ColorToDelete

        # Neighbours of deleted faces are marked in 2.
        for f in self.faces:
            if f['M'] == Mesh.ColorCommon:
                for n in f.nodes:
                    for f1 in n.faces:
                        if f1['M'] == Mesh.ColorToDelete:
                            f['M'] = Mesh.ColorBorder

        # Delete faces.
        faces_to_delete = [f for f in self.faces if f['M'] == Mesh.ColorToDelete]
        for f in faces_to_delete:
            self.delete_face(f)

    def lo_face(self, i):
        """
        Min face.
        Examples:
          i = 0 - left face
          i = 1 - bottom face
          i = 2 - back face

        Parameters
        ----------
        i : int
            Axis.

        Returns
        -------
        Face
            Face with minimal position.
        """

        tl = self.triangles_list()
        tl = geom.Triangle.sorting_by_the_selected_axis(tl, i)

        return tl[0].back_ref

    def hi_face(self, i):
        """
        Max face.
        Examples:
          i = 0 - right face
          i = 1 - up face
          i = 2 - front face

        Parameters
        ----------
        i : int
            Axis.

        Returns
        -------
        Face
            Face with maximum position.
        """

        tl = self.triangles_list()
        tl = geom.Triangle.sorting_by_the_selected_axis(tl, i)

        return tl[-1].back_ref

    def walk_until_border(self, start, mark_color):
        """
        Mark f into mark_color.
        Mark all neighbours of f into mark_color.
        And so on, while not reaching border.

        Parameters
        ----------
        start : Face
            Start face.
        mark_color : int
            Color for mark.
        """

        li = [start]

        while li:
            f = li.pop()
            if f['M'] == mark_color:
                continue
            f['M'] = mark_color
            for n in f.nodes:
                for f1 in n.faces:
                    li.append(f1)

    def delete_faces(self, c):
        """
        Delete faces of the color.

        Parameters
        ----------
        c : int
            Color.
        """

        faces_to_delete = [f for f in self.faces if f['M'] == c]

        for f in faces_to_delete:
            self.delete_face(f)


if __name__ == '__main__':
    pass
