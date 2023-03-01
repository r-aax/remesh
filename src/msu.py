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

        return f'Node {self.glo_id} ({self.p})'

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

    def neighbour(self, e):
        """
        Get neighbour by edge.

        Parameters
        ----------
        e : Edge
            Edge.

        Returns
        -------
        Node
            Neighbour node or None.
        """

        if self == e.nodes[0]:
            return e.nodes[1]
        elif self == e.nodes[1]:
            return e.nodes[0]
        else:
            return None

    def neighbourhood(self):
        """
        Get heighbourhood.

        Returns
        -------
        [Node]
            List of neighbour nodes.
        """

        return [self.neighbour(e) for e in self.edges]


class Edge:
    """
    Edge - border between two faces
    """

    def __init__(self):
        """
        Initialization.
        """

        self.glo_id = -1

        self.faces = []
        self.nodes = []

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        id0, id1 = self.nodes[0].glo_id, self.nodes[1].glo_id
        ps_str = '/ps' if self.is_pseudo() else ''

        return f'Edge{ps_str} {self.glo_id} ({id0} - {id1})'

    def is_faces_free(self):
        """
        Check if edge is without incident faces.

        Returns
        -------
        True - if edge is without faces,
        False - otherwise.
        """

        return len(self.faces) == 0

    def is_pseudo(self):
        """
        Check edge for pseudoedge.

        Returns
        -------
        True - if it is pseudo edge,
        False - if it is normal edge.
        """

        return self.nodes[0] == self.nodes[1]

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

    def center(self):
        """
        Center point.

        Returns
        -------
        Point
            Point of center.
        """

        return (self.nodes[0].p + self.nodes[1].p) / 2.0

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

    def __init__(self):
        """
        Initialization.
        """

        # Global identifier.
        self.glo_id = -1

        self.data = None
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

    def set_data(self, variables, values):
        """
        Set data.

        Parameters
        ----------
        variables : [str]
            Variables names.
        values : [object]
            List of values.

        """

        self.data = dict(zip(variables, values))

    def copy_data_from(self, f):
        """
        Copy data from another face.

        Parameters
        ----------
        f : Face
            Another face.
        """

        self.set_data(f.data.keys(), f.data.values())

    def __repr__(self):
        """
        String representation.

        Returns
        -------
        str
            String.
        """

        id0, id1, id2 = self.nodes[0].glo_id, self.nodes[1].glo_id, self.nodes[2].glo_id
        ps_str = '/ps' if self.is_pseudo() else ''
        th_str = '/th' if self.is_thin() else ''

        return f'Face{ps_str}{th_str} {self.glo_id} ({id0}, {id1}, {id2})'

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

    def is_pseudo(self):
        """
        Check if face is pseudo.

        Returns
        -------
        True - if face is pseudo,
        False - if face is not pseudo.
        """

        a, b, c = self.nodes[0], self.nodes[1], self.nodes[2]

        return (a == b) or (b == c) or (a == c)

    def is_thin(self):
        """
        Check face for thin.

        Results
        -------
        True - if face is thin,
        False - if it's not.
        """

        return (not self.is_pseudo()) and (self.triangle().area() < mth.EPS)

    def is_thin_with_border_big_edge(self):
        """
        Check if face thin, and its big side is border of the scope.

        Returns
        -------
        True - if face is thin and its big side is a border,
        False - otherwise.
        """

        if not self.is_thin():
            return False

        s = self.big_edge()

        return len(s.faces) == 1

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

    def big_edge(self):
        """
        Get biggestr edge.

        Returns
        -------
        Edge
            Edge with maximum length.
        """

        return self.edges[np.argmax([e.length() for e in self.edges])]

    def neighbour(self, e):
        """
        Get neighbour by edge.

        Parameters
        ----------
        e : Edge
            Edge.

        Returns
        -------
        Node
            Neighbour face or None.
        """

        assert len(e.faces) == 2

        if self == e.faces[0]:
            return e.faces[1]
        elif self == e.faces[1]:
            return e.faces[0]
        else:
            return None

    def neighbourhood(self):
        """
        Get neighbourhood.

        Returns
        -------
        [Face]
            List of neighbour faces.
        """

        return [self.neighbour(e) for e in self.edges if len(e.faces) == 2]

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

        self.area = geom.points_area(a, b, c)

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

    def third_node(self, e):
        """
        Third node (not e.nodes[0], e.nodes[1]).

        Parameters
        ----------
        e : Edge
            Edge.

        Returns
        -------
        Node
            Third node.
        """
        ns = [self.nodes[0], self.nodes[1], self.nodes[2]]
        ns.remove(e.nodes[0])
        ns.remove(e.nodes[1])

        assert len(ns) == 1

        return ns[0]


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

    def __init__(self, filename=None):
        """
        Initialization.

        Parameters
        ----------
        filename : str
            File for load.
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

        # Load.
        if not filename is None:
            self.load(filename)

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

    def print(self,
              print_edges_with_incident_faces=False,
              print_faces_neighbourhood=False):
        """
        Print information.

        Parameters
        ---------
        print_edges_with_incident_faces : bool
            Flag for print edges with incident faces.
        print_faces_neighbourhood : bool
            Flag for print faces with neighbourhood.
        """

        print('[MESH]')
        print(f'Nodes ({len(self.nodes)}):\n  ', self.nodes)
        print(f'Edges ({len(self.edges)}):\n  ', self.edges)
        print(f'Faces ({len(self.faces)}):\n  ', self.faces)

        if print_edges_with_incident_faces:
            print('[EDGES WITH INCIDENT FACES]')
            for e in self.edges:
                print(e, ' --- ', e.faces)

        if print_faces_neighbourhood:
            print('[FACES WITH NEIGHBOURHOOD]')
            for f in self.faces:
                print(f, ' --- ', f.triangle().area(), ' --- ', f.neighbourhood())

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

    def find_edge(self, a, b):
        """
        Find edge with two nodes.

        Parameters
        ----------
        a : Node
            First node.
        b : Node.
            Second node.

        Returns
        -------
        Edge | None
            Found edge or None.
        """

        for e in a.edges:
            if a.neighbour(e) == b:
                return e

        # Not found.
        return None

    def find_face(self, a, b, c):
        """
        Find face with given nodes.

        Parameters
        ----------
        a : Node
            First node.
        b : Node
            Second node.
        c : Node
            Third node.

        Returns
        -------
        Face | None
        """

        ids = sorted([a.glo_id, b.glo_id, c.glo_id])

        for f in a.faces:
            lids = sorted([n.glo_id for n in f.nodes])
            if ids == lids:
                return f

        return None

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

    def add_node(self, p, zone):
        """
        Add node to mesh.
        This is only way to add node into mesh.

        Parameters
        ----------
        p : Point
            Point.
        zone : Zone
            Zone to add node to.

        Returns
        -------
        Node
            Added node
            (it may be new node or found near node).
        """

        n = Node(p)
        found_node = self.find_near_node(n)

        if found_node is None:
            max_glo_id = self.max_node_glo_id()
            n.glo_id = max_glo_id + 1
            self.nodes.append(n)
            self.rounded_coordinates_bag.add(n.rounded_coordinates())
            node_to_zone = n
        else:
            node_to_zone = found_node

        zone.nodes.append(node_to_zone)

        return node_to_zone

    def add_edge(self, a, b):
        """
        Add edge or return already existing one.

        Parameters
        ----------
        a : Node
            First node.
        b : Node
            Second node.

        Returns
        -------
        Edge
            Edge (found or new).
        """

        e = self.find_edge(a, b)

        if e is None:
            e = Edge()
            max_glo_id = self.max_edge_glo_id()
            e.glo_id = max_glo_id + 1
            self.edges.append(e)
            self.links([(a, e), (b, e)])

        return e

    def add_face(self, a, b, c, zone):
        """
        Add face to mesh.

        Parameters
        ----------
        a : Node
            First node.
        b : Node
            Second node.
        c : Node
            Third node.
        zone : Zone
            Zone to add to.
        """

        f = self.find_face(a, b, c)

        if f is None:
            f = Face()
            max_glo_id = self.max_face_glo_id()
            f.glo_id = max_glo_id + 1
            self.faces.append(f)
            zone.faces.append(f)
            f.zone = zone
            ab, bc, ac = self.add_edge(a, b), self.add_edge(b, c), self.add_edge(a, c)
            self.links([(a, f), (b, f), (c, f), (ab, f), (bc, f), (ac, f)])

        return f

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
                raise Exception(f'msu.Mesh : wrong object type in link ({obj2})')
        elif isinstance(obj1, Edge):
            if isinstance(obj2, Face):
                obj1.faces.append(obj2)
                obj2.edges.append(obj1)
            else:
                raise Exception(f'msu.Mesh : wrong object type in link ({obj2})')
        else:
            raise Exception(f'msu.Mesh : wrong object type in link ({obj1})')

    def links(self, li):
        """
        Multiple links.

        Parameters
        ----------
        li : [(object, object]
            List of pairs for link.
        """

        for obj1, obj2 in li:
            self.link(obj1, obj2)

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
                raise Exception(f'msu.Mesh : wrong object type in unlink ({obj2})')
        elif isinstance(obj1, Edge):
            if isinstance(obj2, Face):
                obj1.faces.remove(obj2)
                obj2.edges.remove(obj1)
            else:
                raise Exception(f'msu.Mesh : wrong object type in unlink ({obj2})')
        else:
            raise Exception(f'msu.Mesh : wrong object type in unlink ({obj1})')

    def unlinks(self, li):
        """
        Multiple unlink.

        Parameters
        ----------
        li : [(object, object)]
            List  of pairs for unlink.
        """

        for obj1, obj2 in li:
            self.unlink(obj1, obj2)

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

    def replace_edge_face_link(self, e, f, new_f):
        """
        Replace node in face-node link.

        Parameters
        ----------
        e : Edge
            Edge.
        f : Face
            Old Face.
        new_f : Face
            New Face.
        """

        i = e.faces.index(f)
        e.faces[i] = new_f
        f.edges.remove(e)
        new_f.edges.append(e)

    def create_edges(self):
        """
        Delete all edges and create them.
        """

        # Delete all edgs manually.
        # After this action the mesh is not consistent.
        for n in self.nodes:
            n.edges = []
        for f in self.faces:
            f.edges = []
        self.edges = []

        # Construct edges.
        for f in self.faces:
            a, b, c = f.nodes[0], f.nodes[1], f.nodes[2]
            for first, second in [(a, b), (b, c), (a, c)]:
                e = self.add_edge(first, second)
                self.link(e, f)

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
                        self.add_node(np.array([c[0][i], c[1][i], c[2][i]]), zone)

                    # Read data for faces.
                    d = []
                    for i in range(face_variables_count):
                        line = f.readline()
                        d.append([float(xi) for xi in line.split()])
                    values = [[d[j][i] for j in range(face_variables_count)] for i in range(faces_to_read)]

                    # Read connectivity lists.
                    for i in range(faces_to_read):
                        line = f.readline()
                        nodes = [zone.nodes[int(ss) - 1] for ss in line.split()]
                        assert len(nodes) == 3
                        face = self.add_face(nodes[0], nodes[1], nodes[2], zone)
                        face.set_data(face_variables, values[i])
                else:
                    raise Exception('Unexpected line : {0}.'.format(line))

                line = f.readline()
            f.close()

        # Create edges.
        self.create_edges()

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

        # Save faces glo_id.
        for f in self.faces:
            f['Id'] = f.glo_id

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

        # Unlink from nodes.
        while f.nodes:
            self.unlink(f.nodes[0], f)

        # Unlink from edges.
        while f.edges:
            self.unlink(f.edges[0], f)

        # Remove from zones.
        for z in self.zones:
            if f in z.faces:
                z.faces.remove(f)

        # Remove from mesh.
        self.faces.remove(f)

    def delete_faces(self, p):
        """
        Delete faces with predicate.

        Parameters
        ----------
        p : lambda
            Predicate for delete face.
        """

        fs = [f for f in self.faces if p(f)]

        for f in fs:
            self.delete_face(f)

    def delete_edge(self, e):
        """
        Delete edge.

        Parameters
        ----------
        e : Edge
            Edge to delete.
        """

        # First we must to delete incident faces.
        while e.faces:
            self.delete_face(e.faces[0])

        # Unlink edge from nodes.
        while e.nodes:
            self.unlink(e.nodes[0], e)

        # Remove from mesh.
        self.edges.remove(e)

    def delete_edges(self, p):
        """
        Delete all edges with predicate.

        Parameters
        ----------
        p : lambda
            Predicate for edge delete.
        """

        es = [e for e in self.edges if p(e)]

        for e in es:
            self.delete_edge(e)

    def delete_node(self, n):
        """
        Delete node.

        Parameters
        ----------
        n : Node
            Node to be deleted.
        """

        # First we must delete all adjacent edges
        while n.edges:
            self.delete_edge(n.edges[0])

        # Remove node from zones.
        for z in self.zones:
            if n in z.nodes:
                z.nodes.remove(n)

        # Remove node from mesh.
        self.nodes.remove(n)

    def delete_nodes(self, p):
        """
        Delete nodes with predicate.

        Parameters
        ----------
        p : lambda
            Predicate for delete.
        """

        ns = [n for n in self.nodes if p(n)]

        for n in ns:
            self.delete_node(n)

    def delete_double_face(self, delete_faces):
        """
        Delete double face.
        """
        double_edge = None
        useful_nodes = None
        # search for double edge
        for n in delete_faces[0].nodes:
            edge_counter = {}
            for e in n.edges:
                # edge can be reverced
                edge_nodes = tuple(sorted(e.nodes, key=lambda n: n.glo_id))
                if edge_nodes not in edge_counter:
                    edge_counter[edge_nodes] = 1
                else:
                    edge_counter[edge_nodes] += 1
            max_count = max(edge_counter.values())
            if max_count > 1:
                # nodes pair what contains double edge
                useful_nodes = list(edge_counter.keys())[list(edge_counter.values()).index(max_count)]
                double_edge = self.find_edge(useful_nodes[0], useful_nodes[1])
                break
        assert double_edge is not None
        unuseful_node = delete_faces[0].third_node(double_edge)
        self.delete_node(unuseful_node)
        while double_edge.faces:
            self.unlink(double_edge, double_edge.faces[0])
        self.delete_edge(double_edge)
        # local restore edge neigbour faces
        a, b = useful_nodes[0], useful_nodes[1]
        useful_edge = self.find_edge(a, b)
        for f in a.faces:
            if f in b.faces and f not in useful_edge.faces:
                self.link(useful_edge, f)

    def reduce_edge(self, e, move=True):
        """
        Reduce edge.

        Parameters
        ----------
        e : Edge
            Edge.
        """

        a, b = e.nodes[0], e.nodes[1]
        if move:
            a.p = 0.5 * (a.p + b.p)

        # Replace b node with a node in all faces.
        delete_faces = [f for f in e.faces]

        #check if this is double triangle
        if len(delete_faces) == 2 and set(delete_faces[0].nodes) == set(delete_faces[1].nodes):
            self.delete_double_face(delete_faces)
            return

        # Delete extra node and faces.
        for f in delete_faces:
            c = f.third_node(e)
            bc_edge = self.find_edge(b, c)
            if len(bc_edge.faces) == 2:
                if bc_edge.faces[0] != f:
                    external_face = bc_edge.faces[0]
                else:
                    external_face = bc_edge.faces[1]
                ac_edge = self.find_edge(a, c)
                #print(ac_edge, ac_edge.faces)
                self.unlink(bc_edge, external_face)
                self.replace_edge_face_link(ac_edge, f, external_face)
                if b in external_face.nodes:
                    self.replace_face_node_link(external_face, b, a)
            self.delete_edge(bc_edge)#f will be also deleted

        change_faces = [f for f in b.faces]
        change_edges = [be for be in b.edges if be!=e]
        # change edges with b node
        for f in change_faces:
            self.replace_face_node_link(f, b, a)
        for edge in change_edges:
                    self.unlink(b, edge)
                    edge.nodes.append(a)
                    a.edges.append(edge)

        # We need no b node anymore if it is isolated.
        if not b.faces:
            self.delete_node(b)

    def split_edge(self, e, p=None):
        """
        Split edge by point.

        Parameters
        ----------
        e : Edge
            Edge to split.
        p : Point | None
            Point (it is in None then split by center).
        """

        # Check for pseudo edge and edge without faces.
        assert not e.is_pseudo()
        assert len(e.faces) > 0

        # Split by default.
        if p is None:
            p = e.center()

        # Split all incident faces.
        for f in e.faces:
            assert not f.is_pseudo()

            # Old data from face.
            a, b, c = f.nodes[0], f.nodes[1], f.nodes[2]
            th = f.third_node(e)
            z = f.zone

            # Add node.
            n = self.add_node(p, z)

            # If node is in edge's nodes then we don't split it.
            if n in e.nodes:
                return

            # Add edges.
            e0 = self.add_edge(e.nodes[0], n)
            e1 = self.add_edge(e.nodes[1], n)
            eth = self.add_edge(th, n)

            # Add faces.
            li = [a, b, c]
            li[li.index(e.nodes[1])] = n
            f0 = self.add_face(li[0], li[1], li[2], z)
            f0.copy_data_from(f)
            li = [a, b, c]
            li[li.index(e.nodes[0])] = n
            f1 = self.add_face(li[0], li[1], li[2], z)
            f1.copy_data_from(f)

        # Delete edge.
        self.delete_edge(e)

    def split_face(self, f, p=None):
        """
        Split face with point.

        Parameters
        ----------
        f : Face
            Face to be splitted.
        p : Point
            Point for split or None (in this case we split by center).
        """

        # Center point by default.
        if p is None:
            p = f.center()

        # Data from old face.
        a, b, c = f.nodes[0], f.nodes[1], f.nodes[2]
        z = f.zone

        # New node.
        n = self.add_node(p, z)

        # issue #7
        # If node in face's nodes then it's nothing to split.
        if n in f.nodes:
            return

        # Add new faces.
        fab = self.add_face(a, b, n, z)
        fbc = self.add_face(b, c, n, z)
        fca = self.add_face(c, a, n, z)
        fab.copy_data_from(f)
        fbc.copy_data_from(f)
        fca.copy_data_from(f)

        # Delete old face.
        self.delete_face(f)

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
            self.add_node(n.p, z)
        for f in m.faces:
            new_f = Face()
            new_f.copy_data_from(f)
            self.add_face(new_f, z)
            self.links([(z.nodes[m.nodes.index(f.nodes[0])], new_f),
                        (z.nodes[m.nodes.index(f.nodes[1])], new_f),
                        (z.nodes[m.nodes.index(f.nodes[2])], new_f)])

        self.create_edges()

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

    def walk_surface(self, start, mark_color):
        """
        Walk mesh surface.

        Parameters
        ----------
        start : Face
            Start face.
        mark_color : int
            Mark color.
        """

        for f in self.faces:
            f['M'] = Mesh.ColorCommon

        li = [start]

        while li:
            f = li.pop()
            if f['M'] == mark_color:
                continue
            f['M'] = mark_color
            for e in f.edges:
                if len(e.faces) == 2:
                    li.append(f.neighbour(e))

        for f in self.faces:
            if f['M'] == Mesh.ColorCommon:
                f['M'] = Mesh.ColorToDelete

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

    def has_thin_triangles(self):
        """
        Check if mesh has thin triangles.

        Returns
        -------
        True - if mesh has thin triangles,
        False - if mesh has not thin triangles.
        """

        return any(map(lambda f: f.is_thin(), self.faces))

    def has_pseudo_edges_faces(self):
        """
        Check if mesh has pseudo edges or faces.

        Returns
        -------
        True - if mesh has pseudo objects,
        False - if mesh has no pseudo objects.
        """

        has_edges = any(map(lambda e: e.is_pseudo(), self.edges))
        has_faces = any(map(lambda f: f.is_pseudo(), self.faces))

        return has_edges or has_faces

    def split_thin_faces(self):
        """
        Split thin triangles.
        """

        # Reset split points.
        for e in self.edges:
            e.split_points = []

        # Find all thin triangles and mark to split their biggest edge.
        for f in self.faces:
            if f.is_thin():
                e = f.big_edge()
                e.split_points.append(f.third_node(e).p)

        # Correct split points of edges.
        edges_to_split = []
        for e in self.edges:
            l = len(e.split_points)
            if l == 0:
                pass
            elif l == 1:
                edges_to_split.append(e)
            elif l == 2:
                p = (e.split_points[0] + e.split_points[1]) / 2.0
                e.split_points = [p]
            else:
                raise Exception(f'msu.Mesh : wrong count of edge split points ({l})')

        # Split edges.
        for e in edges_to_split:
            self.split_edge(e, e.split_points[0])


if __name__ == '__main__':
    pass
