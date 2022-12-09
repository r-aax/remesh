# Remesher.

import sys
import math
import time
import numpy as np
from numpy import linalg as LA
import logging
from logging import StreamHandler, Formatter
from dataclasses import dataclass

if __name__ != '__main__':
    from src import Solver

# Count of valuable digits (after dot) in node coordinates.
# If coordinates of nodes doesn't differ in valuable digits we consider them equal.
NODE_COORDINATES_VALUABLE_DIGITS_COUNT = 10

# String of export.
EXPORT_FORMAT_STRING = '{0:.18e}'

# Small eps.
EPS = 1.0e-10

# Log.
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
log.addHandler(handler)


def init_logging(log_path):
    logging.basicConfig(filename=log_path+'log.txt', encoding='utf-8', level=logging.DEBUG)


def quadratic_equation_smallest_positive_root(a, b, c):
    """
    Smallest positive root of equation ax^2 + bx + c = 0.

    Parameters
    ----------
    a : float
        Coefficient with x^2.
    b : float
        Coefficient with x.
    c : float
        Free coefficient.

    Returns
    -------
        Smallest positive root or None.
    """

    if a != 0.0:
        d = b * b - 4.0 * a * c
        if d >= 0.0:
            d = math.sqrt(d)
            x1, x2 = (-b - d) / (2.0 * a), (-b + d) / (2.0 * a)
            if (x1 > 0.0) and (x2 > 0.0):
                return min(x1, x2)
            elif x1 > 0.0:
                return x1
            elif x2 > 0.0:
                return x2
    elif b != 0.0:
        x = -c / b
        if x > 0.0:
            return x

    return None


def tetrahedra_volume(a, b, c, d):
    """
    Tetrahedra volume.

    Parameters
    ----------
    a : np.array
        First point
    b : np.array
        Second point
    c : np.array
        Third point
    d : np.array
        4-th point

    Returns
    -------
    float
        Volume
    """

    return abs(np.dot((a - d), np.cross(b - d, c - d))) / 6.0


def pseudoprism_volume(a, b, c, na, nb, nc):
    """
    Pseudoprism volume.

    Source: [4] Fig. 1.

    Parameters
    ----------
    a : Vector
        First vector
    b : Vector
        Second vector
    c : Vector
        Third vector
    na : Vector
        New position of the first vector
    nb : Vector
        New position of the second vector
    nc : Vector
        New position of the third vector

    Returns
    -------
    float
        Volume
    """

    return tetrahedra_volume(a, b, c, nc) \
           + tetrahedra_volume(a, b, nb, nc) \
           + tetrahedra_volume(a, na, nb, nc)


def solve_quadratic_equation(a, b, c):
    """
    Solve equation a * x^2 + b * x + c = 0

    Parameters
    ----------
    a : float
        Parameter near x^2.
    b : float
        Parameter near x.
    c : float
        Free parameter.

    Returns
    -------
    Tuple (num, roots)
    num : int
        Number of roots (0, 1, 2, or math.inf).
    roots : [float]
        Roots array (for 1 root or 2 roots) or [].

    Examples
    --------
    0, []
    1, [r]
    2, [r1, r2]
    math.inf, []
    """

    if a == 0.0:
        # Linear equation b * x + c = 0.
        if b == 0.0:
            # Horizontal line c = 0.
            if c == 0.0:
                # All numbers are roots.
                return math.inf, []
            else:
                # No roots.
                return 0, []
        else:
            # One root.
            return 1, [-c / b]
    else:
        # Quadratic equation.
        d = b * b - 4.0 * a * c
        if d < 0.0:
            # No roots.
            return 0, []
        elif d > 0.0:
            # 2 roots.
            sd = math.sqrt(d)
            return 2, [(-b + sd) / (2.0 * a), (-b - sd) / (2.0 * a)]
        else:
            # 1 root.
            return 1, [-b / (2.0 * a)]


def find_local_extremums_kxk_qxxqxq(k_x, q_x2, q_x, q):
    """
    Find extremum point for function
    alfa(x) = k_x * x + sqrt(q_x2 * x^2 + q_x * x + q).

    Parameters
    ----------
    k_x : float
        Parameter.
    k : float
        Parameter.
    q_x2 : float
        Parameter.
    q_x : float
        Parameter.
    q : float
        Parameter.

    Returns
    -------
    [float]
        List of extremums.
    """

    def sq(x):
        return q_x2 * x**2 + q_x * x + q

    def df(x):
        return k_x + (2.0 * q_x2 * x + q_x) / (2.0 * (math.sqrt(sq(x))))

    # Find roots when sq(x) = 0.
    _, r_sq = solve_quadratic_equation(q_x2, q_x, q)

    # Construct quadratic equation for find extremums (df = 0).
    a = 4.0 * (k_x**2 * q_x2 - q_x2**2)
    b = 4.0 * q_x * (k_x**2 - q_x2)
    c = 4.0 * k_x**2 * q - q_x**2

    # Solve equation.
    _, r_df = solve_quadratic_equation(a, b, c)

    return r_sq + list(filter(lambda x: (sq(x) >= 0.0) and (abs(df(x)) <= EPS), r_df))


def time_to_icing_triangle_surface(a, ra, b, rb, c, rc, d):
    """
    Time to the surface of icing triangle.

    Parameters
    ----------
    a : array
        A point.
    ra : float
        A radius.
    b : array
        B point.
    rb : float
        B radius.
    c : array
        C point.
    rc : float
        C radius.
    d : array
        Direction to the surface.

    Returns
    -------
    [(float, float, float)]
        List of tuples (beta, gamma, alpha)
    """

    # Normalize d.
    d = d / LA.norm(d)

    # Points and radiuses differences.
    ab, ac, bc = b - a, c - a, c - b
    rab, rac, rbc = rb - ra, rc - ra, rc - rb

    # Coefficients.
    # alpha(beta, gamma) = k_b * beta + k_g * gamma + sqrt(T).
    # T = q_b2 * beta^2 + q_g2 * gamma^2 + q_bg * beta * gamma + q_b * beta + q_g * gamma + q.
    k_b = d @ ab
    k_g = d @ ac
    q_b2 = (d @ ab)**2 - (LA.norm(ab))**2 + rab**2
    q_g2 = (d @ ac)**2 - (LA.norm(ac))**2 + rac**2
    q_bg = 2.0 * ((d @ ab) * (d @ ac) - (ab @ ac) + rab * rac)
    q_b = 2.0 * ra * rab
    q_g = 2.0 * ra * rac
    q = ra**2

    # General function for alpha.
    def alpha(beta, gamma):
        if (beta < 0.0) or (gamma < 0.0) or (beta + gamma > 1.0):
            return 0.0
        sq = q_b2 * beta**2 + q_g2 * gamma**2 + q_bg * beta * gamma + q_b * beta + q_g * gamma + q
        if sq < 0.0:
            return 0.0
        else:
            return k_b * beta + k_g * gamma + math.sqrt(sq)

    res = []

    #
    # Case 1.
    #

    def normal(t):
        m = np.array([[t, rac * ab[2] - rab * ac[2], rab * ac[1] - rac * ab[1]],
                      [rab * ac[2] - rac * ab[2], t, rac * ab[0] - rab * ac[0]],
                      [rac * ab[1] - rab * ac[1], rab * ac[0] - rac * ab[0], t]])
        n = LA.inv(m) @ (np.cross(ab, ac))
        return n / LA.norm(n)

    def line_plane_intersection(lp, ld, la, lab, lac):
        m = np.array([[ld[0], -lab[0], -lac[0]],
                      [ld[1], -lab[1], -lac[1]],
                      [ld[2], -lab[2], -lac[2]]])
        if LA.det(m) == 0.0:
            return 0.0, 0.0, 0.0
        mi = LA.inv(m)
        return mi @ (la - lp)

    ns = map(normal, [1.0, -1.0])
    for ni in ns:
        a_sh, b_sh, c_sh = a + ni * ra, b + ni * rb, c + ni * rc
        al1, bt1, gm1 = line_plane_intersection(a, d, a_sh, b_sh - a_sh, c_sh - a_sh)
        p1 = a + d * al1
        al2, bt2, gm2 = line_plane_intersection(p1, -ni, a, ab, ac)
        res.append((bt2, gm2, alpha(bt2, gm2)))

    #
    # Case 2.
    #

    c2_betas = [0.0, 1.0] + find_local_extremums_kxk_qxxqxq(k_b, q_b2, q_b, q)
    for c2_b in c2_betas:
        res.append((c2_b, 0.0, alpha(c2_b, 0.0)))

    #
    # Case 3.
    #

    c3_gammas = [0.0, 1.0] + find_local_extremums_kxk_qxxqxq(k_g, q_g2, q_g, q)
    for c3_g in c3_gammas:
        res.append((0.0, c3_g, alpha(0.0, c3_g)))

    #
    # Case 4.
    #

    c4_gammas = [0.0, 1.0] + find_local_extremums_kxk_qxxqxq(k_g - k_b,
                                                             q_b2 + q_g2 - q_bg,
                                                             -2.0 * q_b2 + q_bg - q_b + q_g,
                                                             q_b2 + q_b + q)
    for c4_g in c4_gammas:
        res.append((1.0 - c4_g, c4_g, alpha(1.0 - c4_g, c4_g)))

    return max(map(lambda r: r[2], res))


def primary_and_null_space(A, threshold):
    """
    Calculation of primary and null space of point

    Parameters
    ----------
    A : float matrix
        Matrix A = N.T @ W @ N, N consist of normals to faces connected with point, W is diagonal matrix of weights
    threshold : float
        threshold to separate primary and null space

    Returns
    -------
    float matrix, float matrix, float vector, int
        primary space, null space, eigen values of A, rank of primary space
    """
    eigen_values_original, eigen_vectors_original = LA.eig(A)
    idx = eigen_values_original.argsort()[::-1]
    eigen_values = eigen_values_original[idx]
    eigen_vectors = eigen_vectors_original[:, idx]
    k = sum((eigen_values > threshold * eigen_values[0]))
    primary_space = eigen_vectors[:, :k]
    null_space = eigen_vectors[:, k:]
    return primary_space, null_space, eigen_values, k


def find_common_nodes(face1, face2):
    """
    finds common nodes between faces

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
    return nodes[0], nodes[1]


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
        self.old_p = None
        self.faces = []
        self.A = None
        self.b = None
        # Direction for node moving (we call it normal).
        self.normal = None

    def rounded_coordinates(self):
        """
        Tuple with rounded coordinates.

        Returns
        -------
        tuple
            Rounded coordinates.
        """

        return tuple(map(lambda x: round(x, NODE_COORDINATES_VALUABLE_DIGITS_COUNT), self.p))

    def calculate_A_and_b(self):
        """
        Calculate martrices for equation Ax=b for primary and null space calculation
        """
        N = np.array([f.normal for f in self.faces])
        m = len(self.faces)
        a = np.ones(m)
        W = np.zeros((m, m))
        for i in range(m):
            W[i, i] = self.faces[i].area#inner_angle(self)
        self.b = N.T @ W @ a
        self.A = N.T @ W @ N


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

    def points(self):
        """
        Get points.

        Returns
        -------
        tuple
            Points.
        """

        return self.nodes[0].p, self.nodes[1].p, self.nodes[2].p

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

    def calculate_p_u_vectors(self):
        """
        Ice accreted on face with p1, p2, p3 points and n1, n2, n3 normal directions and n - normal of the face.
        Returns
        -------
        float
            vectors for stable time-step coefficients
        """

        p1, p2, p3 = self.nodes[0].p, self.nodes[1].p, self.nodes[2].p
        n1, n2, n3 = self.nodes[0].normal, self.nodes[1].normal, self.nodes[2].normal
        u1 = n1 / np.dot(self.normal, n1)
        u2 = n2 / np.dot(self.normal, n2)
        u3 = n3 / np.dot(self.normal, n3)
        u21, u31 = u2 - u1, u3 - u1
        p21, p31 = p2 - p1, p3 - p1

        return p21, p31, u21, u31

    def calculate_jiao_coefs(self):
        """
        Function returns a, b, c coefficients for Jiao stability limit.
        """
        p21, p31, u21, u31 = self.calculate_p_u_vectors()
        c0 = np.cross(p21, p31)
        self.jiao_coef_a = c0 @ c0
        self.jiao_coef_b = c0 @ (np.cross(p21, u31) - np.cross(p31,u21))
        self.jiao_coef_c = c0 @ np.cross(u21, u31)

    def calculate_v_coefs(self):
        """
        V(h) = ah + bh^2 + ch^3

        Function returns a, b, c coefficients.
        And we inspect fact is the face contracting or diverging.
        """

        p21, p31, u21, u31 = self.calculate_p_u_vectors()
        self.v_coef_a = 0.5 * LA.norm(np.cross(p21, p31))
        self.v_coef_b = 0.25 * np.dot(np.cross(p21, u31) + np.cross(u21, p31), self.normal)
        self.v_coef_c = 0.25 * np.dot(np.cross(u21, u31), self.normal)

        # V'(h) = a + h * (...)
        # If a > 0 then the face is contracting, otherwise diverging.
        self.is_contracting = self.v_coef_a > 0.0

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

    def calculate_time_step_fraction_jiao(self):
        """
        Calculate time-step fraction jiao.
        Jiao step time fraction is in [0.0, 1.0].
        """

        h = quadratic_equation_smallest_positive_root(self.jiao_coef_c,
                                                      self.jiao_coef_b,
                                                      self.jiao_coef_a)
        if h is not None:
            self.tsf_jiao = min(h, 1.0)
        else:
            self.tsf_jiao = 1.0

        # Stub.
        self.tsf_jiao = 1.0

        # This is is to be exported.
        self['TsfJiao'] = self.tsf_jiao

    def calculate_time_step_fraction(self, time_step_fraction_k, time_step_fraction_jiao):
        """
        Time-step fraction.

        Source: [1] IV.A.4

        Parameters
        ----------
        time_step_fraction_k : float
            Coefficient for define time-step fraction.
        time_step_fraction_jiao : float
            global Jiao time step
        """

        # Equation 3ch^2 + 2bh + a = 0.
        h = quadratic_equation_smallest_positive_root(3.0 * self.v_coef_c,
                                                      2.0 * self.v_coef_b,
                                                      self.v_coef_a)
        if h is not None:
            tsf = time_step_fraction_k \
                  * (self.v_coef_a * h + self.v_coef_b * h * h + self.v_coef_c * h * h * h) / self.target_ice
            self.tsf = min(tsf, time_step_fraction_jiao, 1.0)
        else:
            self.tsf = time_step_fraction_jiao

        # This is is to be exported.
        self['Tsf'] = self.tsf


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


class Edge:
    """
    Edge - border between two faces
    """

    def __init__(self, face1, face2):
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
        self.node1, self.node2 = find_common_nodes(face1, face2)

    def __eq__(self, other):
        return self.face1 == other.face1 and self.face2 == other.face2 \
                or self.face1 == other.face2 and self.face2 == other.face1

    def points(self):
        return self.node1.p, self.node2.p

    def old_points(self):
        return self.node1.old_p, self.node2.old_p


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
        for f in self.faces:
            f.calculate_neighbour_faces()
            es = [Edge(f, neighbour) for neighbour in f.neighbour_faces]
            for e in es:
                if e not in self.edges:
                    self.edges.append(e)

    def remesh_prepare(self):
        """
        Prepare mesh for remeshing
        """

        for f in self.faces:
            f.target_ice = f.area * f['Hi']

        self.initial_target_ice = self.target_ice()

    def generate_accretion_rate(self):
        """
        Generate accretion rate.
        Calculate target ice to accrete in each face.

        Source: [1] IV.A.1
        """

        # Nothing to do.
        pass

    def define_nodal_offset_direction(self, threshold):
        """
        Define nodal offset direction.

        Parameters
        ----------
        threshold : float
            threshold to separate primary and null space
        """

        for n in self.nodes:
            n.calculate_A_and_b()
            primary_space, _, eigen_values, k = primary_and_null_space(n.A, threshold)
            normal = np.array([0.0, 0.0, 0.0])
            for i in range(k):
                normal += (primary_space[:, i] @ n.b) * primary_space[:, i] / eigen_values[i]
            n.normal = normal / np.linalg.norm(normal)

    def normal_smoothing(self, normal_smoothing_steps, normal_smoothing_s, normal_smoothing_k):
        """
        Reduce surface noise by local normal smoothing.

        Function does not change faces normals.
        Faces normal stay faces normals.
        Nodes normals are smoothed after applying the function.

        Source: [1] IV.A.3

        Parameters
        ----------
        normal_smoothing_steps : int
            Steps of normal smoothing.
        normal_smoothing_s : float
            Parameter for local normal smoothing.
        normal_smoothing_k : float
            Parameter for local normal smoothing.
        """

        # Smoothing.
        for _ in range(normal_smoothing_steps):

            # [1] IV.A.3 formula (4)
            for f in self.faces:
                f.smoothed_normal = \
                    sum(map(lambda ln: ln.normal * max(normal_smoothing_s * (1.0 - f.smoothed_normal @ ln.normal),
                                                       normal_smoothing_k),
                            f.nodes))
                f.smoothed_normal = f.smoothed_normal / LA.norm(f.smoothed_normal)

            # [1] IV.A.3 formula (5)
            for n in self.nodes:
                n.normal = sum(map(lambda lf: lf.smoothed_normal / lf.area, n.faces))
                n.normal = n.normal / LA.norm(n.normal)

        # After nodes normals stay unchanged we can calculate V(h) cubic coefficients.
        for f in self.faces:
            f.calculate_v_coefs()
            f.calculate_jiao_coefs()

    def time_step_fraction(self, is_simple_tsf, steps_left, time_step_fraction_k):
        """
        Time-step fraction.

        Source: [1] IV.A.4

        Parameters
        ----------
        is_simple_tsf : bool
            If True - we accrete target_ice / steps ice on each iteration (ignoring mesh problems).
            If False - exact Tong's algorithm.
        steps_left : int
            Left steps count.
        time_step_fraction_k : float
            Coefficient for define time-step fraction.

        Returns
        -------
        float
            Time-step fraction.
        """

        if is_simple_tsf:

            tsf = 1.0 / steps_left;

            for f in self.faces:
                f.tsf_jiao = 1.0
                f.tsf = tsf

        else:

            # Calculate tsf_jiao for all faces.
            for f in self.faces:
                f.calculate_time_step_fraction_jiao()

            tsf_jiao = min(map(lambda lf: lf.tsf_jiao, self.faces))

            # Calculate time step fraction.
            for f in self.faces:
                f.calculate_time_step_fraction(time_step_fraction_k, tsf_jiao)

                tsf = min(map(lambda lf: lf.tsf, self.faces))

        # Chunks initilization.
        for f in self.faces:
            f.ice_chunk = tsf * f.target_ice

        return tsf

    def define_height_field(self):
        """
        Define height field.

        Solve quadratic equation:
        V(h) = ah + bh^2 = target_ice

        TODO: from [1] IV.A.5
        Returns
        -------
        float
            max face height
        """
        maxH = 0
        for f in self.faces:

            a, b = f.v_coef_a, f.v_coef_b

            # Prismas method.
            f.h = f.ice_chunk / f.area

            # Try to solve more accurately (pyramides method).
            if abs(b) > EPS:
                # bh^2 + ah - V = 0
                d = a * a + 4.0 * b * f.ice_chunk
                if d >= 0.0:
                    d = math.sqrt(d)
                    h1, h2 = (-a + d) / (2.0 * b), (-a - d) / (2.0 * b)
                    if (h1 >= 0.0) and (h2 >= 0.0):
                        f.h = min(h1, h2)
                    elif h1 >= 0.0:
                        f.h = h1
                    elif h2 >= 0.0:
                        f.h = h2
            if f.h > maxH:
                maxH = f.h
        return maxH

    def height_smoothing(self, maxH, ah, beta):
        """
        Height smoothing.

        Parameters
        ----------
        maxH : float
            max of height field of mesh
        ah : float
            coefficient
        beta : float
            coefficient
        TODO: [1] IV.A.6
        """
        for e in self.edges:
            f1 = e.face1
            f2 = e.face2
            if (f1.h < f2.h):
                f1, f2 = f2, f1
            dV = f1.area * min(f1.h - f2.h, ah*maxH)
            f1.ice_chunk -= dV * beta
            f2.ice_chunk += dV * beta

    def update_surface_nodal_positions(self):
        """
        Update surface nodal positions.

        Source: [1] IV.A.7
        """

        for n in self.nodes:

            # Magnitude for node point displacement.
            # [1] IV.A.7 formula (11)
            wl_sum = 0.0
            w_sum = 0.0
            for f in n.faces:
                # [1] IV.A.7 formula (12)
                ci = abs(np.dot(f.normal, n.normal)) if f.is_contracting else 1.0
                phi = f.inner_angle(n)
                li = f.h / ci
                wi = phi * ci * ci
                wl_sum += wi * li
                w_sum += wi
            l = wl_sum / w_sum

            # Move p along normal direction with magnituge l.
            # [1] IV.A.7 formula (13)
            n.old_p = n.p.copy()
            n.p += l * n.normal

    def redistribute_remaining_volume(self):
        """
        Redistribute remaining volume.

        Source: [1] IV.A.8
        """

        for f in self.faces:

            # [1] IV.A.8 formula (14)
            f.target_ice -= pseudoprism_volume(f.nodes[0].old_p, f.nodes[1].old_p, f.nodes[2].old_p,
                                               f.nodes[0].p, f.nodes[1].p, f.nodes[2].p)
        for f in self.faces:
            if f.target_ice < 0:
                f_max = max(f.neighbour_faces, key=lambda nf:nf.target_ice)
                f_max.target_ice += f.target_ice
                if f_max.target_ice < 0:
                    f_max.target_ice = 0
                f.target_ice = 0

    def null_space_smoothing(self, threshold, safety_factor=0.2):
        """
        Null-space smoothing.

        Parameters
        __________
        threshold : float
            threshold to separate primary and null space
        safety_factor: float
            0 < safety_factor < 1

        """

        for n in self.nodes:
            n.calculate_A_and_b()
            _, null_space, eigen_values, k = primary_and_null_space(n.A, threshold)
            if k != 3:
                wi = np.array([])
                for f in n.faces:
                    C = abs(np.dot(f.normal, n.normal)) if f.is_contracting else 1.0
                    wi = np.append(wi, f.inner_angle(n) * C * C)
                ci = np.array([np.mean(f.points(), axis=0) - n.p for f in n.faces])
                dv = np.sum([wi[i] * ci[i] for i in range(len(n.faces))], axis=0)/np.sum(wi)
                t = safety_factor * np.sum([np.dot(dv, e)*e for e in null_space.T], axis=0)
                n.old_p = n.p.copy()
                n.p += t
                #logging.debug(f'dv =  {dv}; t = {t}')

    def null_space_smoothing_accretion_volume_interpolation(self):
        """
        Null-space smoothing accretiion volume interpolation.

        TODO: [1] IV.A.10
        """
        for e in self.edges:
            n_e = (e.face1.normal + e.face2.normal)/2
            p1, p2 = e.points()
            p3, p4 = e.old_points()
            n_s = np.cross(p2 - p1, p3 - p1)
            A_s = 0.5 * (LA.norm(np.cross(p2 - p1, p3 - p1)) + LA.norm(np.cross(p4 - p3, p4 - p2)))
            A_r = e.face1.area
            A_l = e.face2.area
            V_flux = 0
            a = np.dot(n_s, n_e)
            if a >= 0:
                V_flux = a * e.face1.ice_chunk * A_s / A_r
            else:
                V_flux = a * e.face2.ice_chunk * A_s / A_l
            e.face2.ice_chunk += V_flux
            e.face1.ice_chunk -= V_flux

    def final_volume_correction_step(self):
        """
        Final volume correction step.
        """

        self.null_space_smoothing_accretion_volume_interpolation()

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

    def remesh(self,
               steps=5,
               is_simple_tsf=False,
               normal_smoothing_steps=10, normal_smoothing_s=10.0, normal_smoothing_k=0.15,
               height_smoothing_steps=20, time_step_fraction_k=0.25, null_space_smoothing_steps=250,
               threshold_for_null_space=0.003, height_smoothing_alpha=0.2, height_smoothing_b=0.1):
        """
        Remesh.

        sources:
            [1] X. Tong, D. Thompson, Q. Arnoldus, E. Collins, E. Luke.
                Three-Dimensional Surface Evolution and Mesh Deformation for Aircraft Icing Applications. //
                Journal of Aircraft, 2016, DOI: 10.2514/1.C033949
            [2] D. Thompson, X. Tong, Q. Arnoldus, E. Collins, D. McLaurin, E. Luke.
                Discrete Surface Evolution and Mesh Deformation for Aircraft Icing Applications. //
                5th AIAA Atmospheric and Space Environments Conference, 2013, DOI: 10.2514/6.2013-2544
            [3] X. Jiao.
                Face Offsetting: A Unified Approach for Explicit Moving Interfaces. //
                Journal of Computational Physics, 2007, pp. 612-625, DOI: 10.1016/j.jcp.2006.05.021
            [4] X. Jiao.
                Volume and Feature Preservation in Surface Mesh Optimization. //
                College of Computing, Georgia Institute of Technology.

        Parameters
        ----------
        steps : int
            Maximum number of steps.
        is_simple_tsf : bool
            If True - we accrete target_ice / steps ice on each iteration (ignoring mesh problems).
            If False - exact Tong's algorithm.
        normal_smoothing_steps : int
            Steps of normal smoothing.
        normal_smoothing_s : float
            Parameter for local normal smoothing.
        normal_smoothing_k : float
            Parameter for local normal smoothing.
        height_smoothing_steps : int
            Steps of height smoothing.
        time_step_fraction_k : float
            Coefficient for define time-step fraction.
        null_space_smoothing_steps : int
            Steps of null space smoothing
        threshold_for_null_space : float
            threshold to separate primary and null space
        height_smoothing_alpha : float
            Coefficient for height_smoothing, 0 < alpha < 1
        height_smoothing_b : float
            Coefficient for height_smoothing, 0 < b < 0.5
        """

        self.calculate_faces_areas()
        self.calculate_faces_normals()
        self.remesh_prepare()
        self.generate_accretion_rate()
        self.calculate_edges()
        step_i = 0

        while True:

            step_i += 1
            self.define_nodal_offset_direction(threshold_for_null_space)
            self.normal_smoothing(normal_smoothing_steps,
                                  normal_smoothing_s,
                                  normal_smoothing_k)

            # When we define time-step fraction, we also set ice_chunks.
            tsf = self.time_step_fraction(is_simple_tsf, steps - step_i + 1, time_step_fraction_k)
            log.info(f'step_i = {step_i}, tsf = {tsf}')

            max_face_height = self.define_height_field()
            for _ in range(height_smoothing_steps):
                self.height_smoothing(max_face_height, height_smoothing_alpha, height_smoothing_b)
                max_face_height = self.define_height_field()

            self.update_surface_nodal_positions()
            self.redistribute_remaining_volume()

            for _ in range(null_space_smoothing_steps):
                self.null_space_smoothing(threshold_for_null_space)
                self.calculate_faces_areas()
                self.calculate_faces_normals()
                self.null_space_smoothing_accretion_volume_interpolation()

            # Break on total successfull remesh.
            if tsf == 1.0:
                log.info(f'break on tsf = 1.0')
                break

            # Break on maximum steps number.
            if step_i == steps:
                log.info(f'break on max_steps ({steps})')
                break

            # Recalculate areas and normals for next iteration.
            self.calculate_faces_areas()
            self.calculate_faces_normals()

        self.final_volume_correction_step()
        self.add_additional_data_for_analysis()

    def calculate_nodes_normals(self):
        """
        Calculate nodes normals.
        """

        for n in self.nodes:
            n.normal = sum(map(lambda f: f.normal, n.faces)) / len(n.faces)

    def new_remesh(self,
                   steps=1):
        """
        New remesh algorithm.

        Parameters
        ----------
        steps : int
            Steps count.
        """

        # Prepare.
        self.calculate_faces_areas()
        self.calculate_faces_normals()
        self.calculate_nodes_normals()
        self.remesh_prepare()

        for step_i in range(steps, 0, -1):

            log.info(f'new_remesh : step, trying to accrete part {step_i} of target ice')

            # Calculate ice_chunk for current iteration and height.
            for f in self.faces:
                f.chunk = f.target_ice / step_i
                f.shift = f.chunk / f.area

            # Define node shifts.
            for n in self.nodes:
                n.shift = (sum(map(lambda f: f.shift, n.faces))) / len(n.faces)

            # Define node shifts 2.
            for n in self.nodes:
                alfa = 0.0
                a = n
                for f in n.faces:
                    if f.nodes[0] == a:
                        b, c = f.nodes[1], f.nodes[2]
                    elif f.nodes[1] == a:
                        b, c = f.nodes[0], f.nodes[2]
                    elif f.nodes[2] == a:
                        b, c = f.nodes[0], f.nodes[1]
                    alfa = max(alfa,
                               time_to_icing_triangle_surface(a.p, a.shift, b.p, b.shift, c.p, c.shift, a.normal))
                n.shift2 = alfa
            for n in self.nodes:
                n.shift = n.shift2

            # Define new points positions.
            for n in self.nodes:
                n.old_p = n.p.copy()
                n.p = n.old_p + (n.normal * n.shift)

            # Correct target ice.
            for f in self.faces:
                f.target_ice -= pseudoprism_volume(f.nodes[0].old_p, f.nodes[1].old_p, f.nodes[2].old_p,
                                                   f.nodes[0].p, f.nodes[1].p, f.nodes[2].p)

            # Recalculate geometry.
            self.calculate_faces_areas()
            self.calculate_faces_normals()
            self.calculate_nodes_normals()


def lrs(name_in, name_out):
    """
    Load, remesh, store.

    Parameters
    ----------
    name_in : str
        Name of input mesh file.
    name_out : str
        Name of output mesh file.
    """

    log.info(f'remesh start : {name_in} -> {name_out}')
    g = Mesh()
    g.load(name_in)
    t0 = time.time()
    g.new_remesh()
    t = time.time() - t0
    target_ice = g.target_ice()
    target_ice_perc = 100.0 * (target_ice / g.initial_target_ice)
    g.store(name_out)
    log.info(f'remesh end : time = {t:.5f} s, target_ice = {target_ice} ({target_ice_perc}%)')


if __name__ != '__main__':

    @dataclass
    class RemeshInputer(Solver):
        mesh_file_in: str

        @classmethod
        def from_config(cls, config: dict) -> "Solver":
            name = config.get("inputer", None)
            if name is None:
                raise KeyError("remesh not found `solver` or `inputer` or `outputer` keyword")
            assert name == "remesh"
            mesh_file_in = cls.extract(config, "mesh_file_in", str)
            return cls(mesh_file_in)

        def solve(self, pool: dict):
            g = Mesh()
            g.load(self.mesh_file_in)
            pool['surface_mesh'] = g


    @dataclass
    class RemeshSolver(Solver):

        @classmethod
        def from_config(cls, config: dict) -> "Solver":
            name = config.get("solver", None)
            if name is None:
                raise KeyError("remesh not found `solver` or `inputer` or `outputer` keyword")
            assert name == "remesh"
            return cls()

        def solve(self, pool: dict):
            g = pool['surface_mesh']
            t0 = time.time()
            g.remesh()
            t = time.time() - t0
            target_ice = g.target_ice()
            log.info(f'\ttime = {t:.5f} s, target_ice = {target_ice:.8f}')


    @dataclass
    class RemeshOutputer(Solver):
        mesh_file_out: str

        @classmethod
        def from_config(cls, config: dict) -> "Solver":
            name = config.get("outputer", None)
            if name is None:
                raise KeyError("remesh not found `solver` or `inputer` or `outputer` keyword")
            assert name == "remesh"
            mesh_file_out = cls.extract(config, "mesh_file_out", str)
            return cls(mesh_file_out)

        def solve(self, pool: dict):
            g = pool['surface_mesh']
            g.store(self.mesh_file_out)


if __name__ == '__main__':
    # lrs('../cases/naca/naca_t05.dat', '../res_naca_t05.dat')
    # lrs('../cases/naca/naca_t12.dat', '../res_naca_t12.dat')
    # lrs('../cases/naca/naca_t25.dat', '../res_naca_t25.dat')
    # lrs('../cases/blender_custom_meshes/holes.dat', '../res_holes.dat')
    # lrs('../cases/blender_custom_meshes/snowman.dat', '../res_snowman.dat')
    lrs('../cases/bunny.dat', '../res_bunny.dat')
