import math
import mth
import numpy as np
from numpy import linalg as LA
from remesher import Remesher


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

    def normalized(v):
        return v / LA.norm(v)

    # Normalize d.
    d = normalized(d)

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

    # Initial alphas for triangle nodes.
    alphas = [alpha(0.0, 0.0), alpha(1.0, 0.0), alpha(0.0, 1.0)]

    #
    # Case 1.
    #

    def normal(t):
        m = np.array([[t, rac * ab[2] - rab * ac[2], rab * ac[1] - rac * ab[1]],
                      [rab * ac[2] - rac * ab[2], t, rac * ab[0] - rab * ac[0]],
                      [rac * ab[1] - rab * ac[1], rab * ac[0] - rac * ab[0], t]])
        return normalized(LA.inv(m) @ (np.cross(ab, ac)))

    def line_plane_intersection(lp, ld, la, lab, lac):
        m = np.array([ld, -lab, -lac]).transpose()
        if LA.det(m) == 0.0:
            return 0.0, 0.0, 0.0
        else:
            return LA.inv(m) @ (la - lp)

    ns = map(normal, [1.0, -1.0])
    for n in ns:
        a_sh, b_sh, c_sh = a + n * ra, b + n * rb, c + n * rc
        surf_alpha, _, _ = line_plane_intersection(a, d, a_sh, b_sh - a_sh, c_sh - a_sh)
        surf_point = a + d * surf_alpha
        _, beta, gamma = line_plane_intersection(surf_point, -n, a, ab, ac)
        alphas.append(alpha(beta, gamma))

    # Case 2.
    for beta in mth.find_extremums_kx_qx2qxq(k_b, q_b2, q_b, q):
        alphas.append(alpha(beta, 0.0))

    # Case 3.
    for gamma in mth.find_extremums_kx_qx2qxq(k_g, q_g2, q_g, q):
        alphas.append(alpha(0.0, gamma))

    # Case 4.
    for gamma in mth.find_extremums_kx_qx2qxq(k_g - k_b,
                                              q_b2 + q_g2 - q_bg,
                                              -2.0 * q_b2 + q_bg - q_b + q_g,
                                              q_b2 + q_b + q):
        alphas.append(alpha(1.0 - gamma, gamma))

    return max(alphas)


class RemesherIsotropic(Remesher):
    """
    Isotropic remesher.
    """

    def __init__(self):
        """
        Constructor.
        """

        Remesher.__init__(self)
        self.name = 'isotropic'

    def inner_remesh(self,
                     mesh,
                     steps=1):
        """
        New remesh algorithm.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        steps : int
            Steps count.
        """

        # Prepare.
        mesh.calculate_faces_areas()
        mesh.calculate_faces_normals()
        mesh.calculate_nodes_normals()
        self.remesh_prepare(mesh)

        for step_i in range(steps, 0, -1):

            print(f'new_remesh : step, trying to accrete part {step_i} of target ice')

            # Calculate ice_chunk for current iteration and height.
            for f in mesh.faces:
                f.chunk = f.target_ice / step_i
                f.shift = f.chunk / f.area

            # Define node shifts.
            for n in mesh.nodes:
                n.shift = (sum(map(lambda f: f.shift, n.faces))) / len(n.faces)

            # Define node shifts 2.
            for n in mesh.nodes:
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
            for n in mesh.nodes:
                n.shift = n.shift2

            # Define new points positions.
            for n in mesh.nodes:
                n.old_p = n.p.copy()
                n.p = n.old_p + (n.normal * n.shift)

            # Correct target ice.
            for f in mesh.faces:
                f.target_ice -= mth.pseudoprism_volume(f.nodes[0].old_p, f.nodes[1].old_p, f.nodes[2].old_p,
                                                       f.nodes[0].p, f.nodes[1].p, f.nodes[2].p)

            # Recalculate geometry.
            mesh.calculate_faces_areas()
            mesh.calculate_faces_normals()
            mesh.calculate_nodes_normals()
