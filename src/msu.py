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


def find_common_faces(n1, n2):
    """
    finds common faces between nodes and return them in sorted order

    Parameters
    ----------
    n1, n2: Node

    Returns
    -------
    [Face]
    """
    faces = []
    for f in n1.faces:
        if f in n2.faces:
            faces.append(f)
    return faces


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
        self.edges = []
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

        return len(self.edges) == 0


class Edge:
    """
    Edge - border between two faces
    """

    def __init__(self, nodes, faces = None):
        """
        Initialization.

        Parameters
        ----------

        nodes: [Node]
            first and second node
        faces: [Face]
            first and second face
        """

        self.glo_id = -1

        if len(nodes) == 2:
            self.faces = faces
            self.nodes = sorted(nodes, key=lambda n: n.glo_id)
        else:
            raise Exception('Nodes count must be 2, got {}'.format(len(nodes)))

    def __eq__(self, other):
        return (self.nodes == other.nodes)

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        return f'Edge {self.glo_id} ({self.nodes[0].glo_id} - {self.nodes[1].glo_id})'

    def points(self):
        """
        Get points.

        Returns
        -------
        (Point, Point)
            Points.
        """
        return self.nodes[0].p, self.nodes[1].p

    def old_points(self):
        """
        Get old points.

        Returns
        -------
        (Point, Point)
            Old points.
        """
        return self.nodes[0].old_p, self.nodes[1].old_p

    def length(self):
        """
        Length of the edge.

        Returns
        -------
        float
            Length of the edge.
        """

        return LA.norm(self.nodes[0].p - self.nodes[1].p)

    def replace_face(self, f, new_f):
        """
        Replace face with new face.

        Parameters
        ----------
        f : Face
            Old face.
        new_f : Face
            New face.
        """

        if self.faces[0] == f:
            self.faces[0] = new_f
        elif self.faces[1] == f:
            self.faces[1] = new_f
        else:
            raise Exception('No such face')

    def flip_nodes(self):
        """
        Flip nodes.
        """

        self.nodes[0], self.nodes[1] = self.nodes[1], self.nodes[0]


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
        self.edges = []
        self.zone = None
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

    def triangle(self):
        """
        Create triangle.

        Returns
        -------
        Triangle
            Triangle.
        """

        return geom.Triangle(self.nodes[0].p, self.nodes[1].p, self.nodes[2].p)


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
        self.edges.clear()
        self.faces.clear()
        self.zones.clear()
        self.rounded_coordinates_bag.clear()

    def print(self):
        """
        Print information.
        """

        print('[MESH]')
        print('Nodes:\n  ', self.nodes)
        print('Edges:\n  ', self.edges)
        print('Faces:\n  ', self.faces)

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

    def max_edge_glo_id(self):
        """
        Get maximum edge global id
        (id of the last edge).

        Returns
        -------
        int
            Maximum edge global id,
            or -1, if there is no edges.
        """

        if self.edges:
            return self.edges[-1].glo_id
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

        Returns
        -------
        Node
            Added node.
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

        return node_to_zone

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

    def link(self, obj1, obj2):
        """
        Link two objects.
        Objects that can be linked:
          - node - edge
          - node - face
          - edge - face

        Parameters
        ----------
        obj1 : Node | Edge
            First object.
        obj2 : Edge | Face
            Second object.
        """

        if isinstance(obj1, Node):
            if isinstance(obj2, Edge):
                obj1.edges.append(obj2)
                obj2.nodes.append(obj1)
            elif isinstance(obj2, Face):
                obj1.faces.append(obj2)
                obj2.nodes.append(obj1)
            else:
                raise Exception('msu.Mesh : wrong object type in link')
        elif isinstance(obj1, Edge):
            if isinstance(obj2, Face):
                obj1.faces.append(obj2)
                obj2.edges.append(obj1)
            else:
                raise Exception('msu.Mesh : wrong object type in link')
        else:
            raise Exception('msu.Mesh : wrong object type in link')

    def unlink(self, obj1, obj2):
        """
        Unlink two objects.

        Parameters
        ----------
        obj1 : Node | Edge
            First object.
        obj2 : Edge | Face
            Second object.
        """

        if isinstance(obj1, Node):
            if isinstance(obj2, Edge):
                obj1.edges.remove(obj2)
                obj2.nodes.remove(obj1)
            elif isinstance(obj2, Face):
                obj1.faces.remove(obj2)
                obj2.nodes.remove(obj1)
            else:
                raise Exception('msu.Mesh : wrong object type in unlink')
        elif isinstance(obj1, Edge):
            if isinstance(obj2, Face):
                obj1.faces.remove(obj2)
                obj2.edges.remove(obj1)
            else:
                raise Exception('msu.Mesh : wrong object type in unlink')
        else:
            raise Exception('msu.Mesh : wrong object type in unlink')

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
                            self.link(n, face)
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

    def reduce_edge(self, e):
        """
        Reduce edge.

        Parameters
        ----------
        e : Edge
            Edge.
        """

        a, b = e.nodes[0], e.nodes[1]
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
        for f in e.faces:
            # Data from face.
            a, b, c = f.nodes[0], f.nodes[1], f.nodes[2]
            sorted_nodes = sorted(f.nodes, key=lambda node: node.glo_id)
            remaining_node = set(f.nodes).difference(set(e.nodes)).pop()
            a1, b1, c1 = sorted_nodes[0], sorted_nodes[1], sorted_nodes[2]
            f1, f2 = f.copy(), f.copy()
            z = f.zone

            # Add node.
            n = self.add_node(n, z)

            # Add new faces.
            self.add_face(f1, z)
            self.add_face_nodes_links(f1, [a, b, c])
            self.replace_face_node_link(f1, e.nodes[0], n)
            node2_pair.append(f1)
            self.add_face(f2, z)
            self.add_face_nodes_links(f2, [a, b, c])
            self.replace_face_node_link(f2, e.nodes[1], n)
            node1_pair.append(f2)
            self.add_edge(Edge([remaining_node, n], [f1, f2]))
            # Delete old face.
            self.delete_face(f)
        self.add_edge(Edge([n, e.nodes[0]], node1_pair))
        self.add_edge(Edge([n, e.nodes[1]], node2_pair))
        self.delete_edge(e)

    def split_face(self, f, p=None):
        """
        Split face with point.

        Parameters
        ----------
        f : Face
            Face to be splitted.
        p : Point
            Point for spllit of None (in this case we split by center).
        """

        if p is None:
            p = f.center()

        # Data from old face.
        a, b, c = f.nodes[0], f.nodes[1], f.nodes[2]
        f1, f2, f3 = f.copy(), f.copy(), f.copy()
        z = f.zone
        # New node.
        n = Node(p)
        n = self.add_node(n, z)
        # Delete old face.
        self.delete_face(f)

        # Add new faces.
        self.add_face(f1, z)
        self.add_face_nodes_links(f1, [a, b, n])
        self.add_face(f2, z)
        self.add_face_nodes_links(f2, [b, c, n])
        self.add_face(f3, z)
        self.add_face_nodes_links(f3, [c, a, n])

        self.add_edge(Edge([b,n], [f1, f2]))
        self.add_edge(Edge([a,n], [f1, f3]))
        self.add_edge(Edge([c,n], [f2, f3]))

    def multisplit_face(self, f, ps):
        """
        Split with several points.

        Parameters
        ----------
        f : Face
            Face to be splitted.
        ps : [Point]
            Points list.
        """

        if not ps:
            return

        hps, tps = ps[0], ps[1:]

        # Split face with first point.
        self.split_face(f, hps)
        # Now split with other points recursively.
        fs = self.faces[-3:]
        ts = [f.triangle() for f in fs]

        # Distribute rest points between last three faces.
        pps = [[], [], []]
        for p in tps:
            areas_diffs = [t.areas_difference(p) for t in ts]
            i = np.argmin(areas_diffs)
            pps[i].append(p)
        for i in range(3):
            self.multisplit_face(fs[i], pps[i])

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
        pairs = list(filter(lambda p: p[0].back_ref.glo_id < p[1].back_ref.glo_id, pairs))

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

    def split_self_intersected_faces(self):
        """
        Split all self-intersected faces.
        """

        # Find self-intersected faces.
        tc = geom.TrianglesCloud(self.triangles_list())
        pairs = tc.intersection_with_triangles_cloud(tc)
        pairs = list(filter(lambda p: p[0].back_ref.glo_id < p[1].back_ref.glo_id, pairs))

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

        # Split faces.
        faces_to_split = [f for f in self.faces if f['M'] == Mesh.ColorToDelete]
        for f in faces_to_split:
            self.split_face(f)

        # Reset colors.
        for f in self.faces:
            f['M'] = Mesh.ColorCommon

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

    def throw_intersection_points_to_faces(self):
        """
        Find all self-intersections of the faces.
        Throw intersection points to them.
        """

        for f in self.faces:
            f.int_points = []

        tc = geom.TrianglesCloud(self.triangles_list())
        pairs = tc.intersection_with_triangles_cloud(tc)
        pairs = list(filter(lambda p: p[0].back_ref.glo_id < p[1].back_ref.glo_id, pairs))

        for pair in pairs:
            [t1, t2] = pair
            ps = t1.find_intersection_with_triangle(t2)
            ps = geom.delete_near_points(ps)
            t1.back_ref.int_points = t1.back_ref.int_points + ps
            t2.back_ref.int_points = t2.back_ref.int_points + ps

        for f in self.faces:
            f.int_points = geom.delete_near_points(f.int_points)

    def multisplit_by_intersection_points(self):
        """
        Multisplit faces by intersection points.
        """

        ff = [f for f in self.faces]
        for f in ff:
            self.multisplit_face(f, f.int_points)


if __name__ == '__main__':
    pass
