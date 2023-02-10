import math
import mth
import numpy as np
from numpy import linalg as LA
from remesher import Remesher
import time

def node_calculate_A_and_b(node):
    """
    Calculate martrices for equation Ax=b for primary and null space calculation
    """
    N = np.array([f.normal for f in node.faces])
    m = len(node.faces)
    a = np.ones(m)
    W = np.zeros((m, m))
    for i in range(m):
        W[i, i] = node.faces[i].area  # inner_angle(self)
    node.b = N.T @ W @ a
    node.A = N.T @ W @ N


def face_calculate_p_u_vectors(face):
    """
    Ice accreted on face with p1, p2, p3 points and n1, n2, n3 normal directions and n - normal of the face.
    Returns
    -------
    float
        vectors for stable time-step coefficients
    """

    p1, p2, p3 = face.nodes[0].p, face.nodes[1].p, face.nodes[2].p
    n1, n2, n3 = face.nodes[0].normal, face.nodes[1].normal, face.nodes[2].normal
    u1 = n1 / np.dot(face.normal, n1)
    u2 = n2 / np.dot(face.normal, n2)
    u3 = n3 / np.dot(face.normal, n3)
    u21, u31 = u2 - u1, u3 - u1
    p21, p31 = p2 - p1, p3 - p1

    return p21, p31, u21, u31


def face_calculate_jiao_coefs(face):
    """
    Function returns a, b, c coefficients for Jiao stability limit.
    """
    p21, p31, u21, u31 = face_calculate_p_u_vectors(face)
    c0 = np.cross(p21, p31)
    face.jiao_coef_a = c0 @ c0
    face.jiao_coef_b = c0 @ (np.cross(p21, u31) - np.cross(p31,u21))
    face.jiao_coef_c = c0 @ np.cross(u21, u31)


def face_calculate_v_coefs(face):
    """
    V(h) = ah + bh^2 + ch^3

    Function returns a, b, c coefficients.
    And we inspect fact is the face contracting or diverging.
    """

    p21, p31, u21, u31 = face_calculate_p_u_vectors(face)
    face.v_coef_a = 0.5 * LA.norm(np.cross(p21, p31))
    face.v_coef_b = 0.25 * np.dot(np.cross(p21, u31) + np.cross(u21, p31), face.normal)
    face.v_coef_c = 0.25 * np.dot(np.cross(u21, u31), face.normal)

    # V'(h) = a + h * (...)
    # If a > 0 then the face is contracting, otherwise diverging.
    face.is_contracting = face.v_coef_a > 0.0


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


def face_calculate_time_step_fraction_jiao(face):
    """
    Calculate time-step fraction jiao.
    Jiao step time fraction is in [0.0, 1.0].
    """

    h = mth.quadratic_equation_smallest_positive_root(face.jiao_coef_c,
                                                      face.jiao_coef_b,
                                                      face.jiao_coef_a)
    if h is not None:
        face.tsf_jiao = min(h, 1.0)
    else:
        face.tsf_jiao = 1.0

    # Stub.
    face.tsf_jiao = 1.0

    # This is is to be exported.
    face['TsfJiao'] = face.tsf_jiao


def face_calculate_time_step_fraction(face, time_step_fraction_k, time_step_fraction_jiao):
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
    h = mth.quadratic_equation_smallest_positive_root(3.0 * face.v_coef_c,
                                                          2.0 * face.v_coef_b,
                                                          face.v_coef_a)
    if h is not None and face.target_ice > 0:
        tsf = time_step_fraction_k \
              * (face.v_coef_a * h + face.v_coef_b * h * h + face.v_coef_c * h * h * h) / face.target_ice
        face.tsf = min(tsf, time_step_fraction_jiao, 1.0)
    else:
        face.tsf = time_step_fraction_jiao

    # This is is to be exported.
    face['Tsf'] = face.tsf


class RemesherTong(Remesher):
    """
    Tong remesher.
    """

    def __init__(self):
        """
        Constant.
        """

        Remesher.__init__(self)
        self.name = 'tong'

    def inner_remesh(self,
                     mesh,
                     steps=2,
                     is_simple_tsf=False,
                     normal_smoothing_steps=10, normal_smoothing_s=10.0, normal_smoothing_k=0.15,
                     height_smoothing_steps=20, time_step_fraction_k=0.25, null_space_smoothing_steps=1,
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
        mesh : Mesh
            Mesh.
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

        mesh.calculate_faces_areas()
        mesh.calculate_faces_normals()
        self.remesh_prepare(mesh)
        self.generate_accretion_rate(mesh)
        calculate_edges_time = time.time()
        mesh.calculate_edges()
        calculate_edges_time = time.time() - calculate_edges_time
        print(f'calculate edges time = {calculate_edges_time}, edges count = {len(mesh.edges)}')
        step_i = 0

        while True:

            step_i += 1
            self.define_nodal_offset_direction(mesh, threshold_for_null_space)
            self.normal_smoothing(mesh,
                                  normal_smoothing_steps,
                                  normal_smoothing_s,
                                  normal_smoothing_k)

            # When we define time-step fraction, we also set ice_chunks.
            tsf = self.time_step_fraction(mesh, is_simple_tsf, steps - step_i + 1, time_step_fraction_k)
            print(f'step_i = {step_i}, tsf = {tsf}')

            max_face_height = self.define_height_field(mesh)
            for _ in range(height_smoothing_steps):
                self.height_smoothing(mesh, max_face_height, height_smoothing_alpha, height_smoothing_b)
                max_face_height = self.define_height_field(mesh)

            self.update_surface_nodal_positions(mesh)
            self.redistribute_remaining_volume(mesh)

            for _ in range(null_space_smoothing_steps):
                self.null_space_smoothing(mesh, threshold_for_null_space)
                mesh.calculate_faces_areas()
                mesh.calculate_faces_normals()
                self.null_space_smoothing_accretion_volume_interpolation(mesh)

            # Break on total successfull remesh.
            if tsf == 1.0:
                print(f'break on tsf = 1.0')
                break

            # Break on maximum steps number.
            if step_i == steps:
                print(f'break on max_steps ({steps})')
                break

            # Recalculate areas and normals for next iteration.
            mesh.calculate_faces_areas()
            mesh.calculate_faces_normals()

        self.final_volume_correction_step(mesh)
        mesh.add_additional_data_for_analysis()

    def generate_accretion_rate(self, mesh):
        """
        Generate accretion rate.
        Calculate target ice to accrete in each face.

        Source: [1] IV.A.1

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        # Nothing to do.
        pass

    def define_nodal_offset_direction(self, mesh, threshold):
        """
        Define nodal offset direction.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        threshold : float
            threshold to separate primary and null space
        """

        for n in mesh.nodes:
            node_calculate_A_and_b(n)
            primary_space, _, eigen_values, k = primary_and_null_space(n.A, threshold)
            normal = np.array([0.0, 0.0, 0.0])
            for i in range(k):
                normal += (primary_space[:, i] @ n.b) * primary_space[:, i] / eigen_values[i]
            n.normal = normal / np.linalg.norm(normal)

    def normal_smoothing(self, mesh, normal_smoothing_steps, normal_smoothing_s, normal_smoothing_k):
        """
        Reduce surface noise by local normal smoothing.

        Function does not change faces normals.
        Faces normal stay faces normals.
        Nodes normals are smoothed after applying the function.

        Source: [1] IV.A.3

        Parameters
        ----------
        mesh : Mesh
            Mesh.
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
            for f in mesh.faces:
                f.smoothed_normal = \
                    sum(map(lambda ln: ln.normal * max(normal_smoothing_s * (1.0 - f.smoothed_normal @ ln.normal),
                                                       normal_smoothing_k),
                            f.nodes))
                f.smoothed_normal = f.smoothed_normal / LA.norm(f.smoothed_normal)

            # [1] IV.A.3 formula (5)
            for n in mesh.nodes:
                n.normal = sum(map(lambda lf: lf.smoothed_normal / lf.area, n.faces))
                n.normal = n.normal / LA.norm(n.normal)

        # After nodes normals stay unchanged we can calculate V(h) cubic coefficients.
        for f in mesh.faces:
            face_calculate_v_coefs(f)
            face_calculate_jiao_coefs(f)

    def time_step_fraction(self, mesh, is_simple_tsf, steps_left, time_step_fraction_k):
        """
        Time-step fraction.

        Source: [1] IV.A.4

        Parameters
        ----------
        mesh : Mesh
            Mesh.
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

            for f in mesh.faces:
                f.tsf_jiao = 1.0
                f.tsf = tsf

        else:

            # Calculate tsf_jiao for all faces.
            for f in mesh.faces:
                face_calculate_time_step_fraction_jiao(f)

            tsf_jiao = min(map(lambda lf: lf.tsf_jiao, mesh.faces))

            # Calculate time step fraction.
            for f in mesh.faces:
                face_calculate_time_step_fraction(f, time_step_fraction_k, tsf_jiao)

                tsf = min(map(lambda lf: lf.tsf, mesh.faces))

        # Chunks initilization.
        for f in mesh.faces:
            f.ice_chunk = tsf * f.target_ice

        return tsf

    def define_height_field(self, mesh):
        """
        Define height field.

        Solve quadratic equation:
        V(h) = ah + bh^2 = target_ice

        TODO: from [1] IV.A.5

        Parameters
        ----------
        mesh : Mesh
            Mesh.

        Returns
        -------
        float
            max face height
        """
        maxH = 0
        for f in mesh.faces:

            a, b = f.v_coef_a, f.v_coef_b

            # Prismas method.
            f.h = f.ice_chunk / f.area

            # Try to solve more accurately (pyramides method).
            if abs(b) > mth.EPS:
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

    def height_smoothing(self, mesh, maxH, ah, beta):
        """
        Height smoothing.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        maxH : float
            max of height field of mesh
        ah : float
            coefficient
        beta : float
            coefficient
        TODO: [1] IV.A.6
        """

        for e in mesh.edges:
            f1 = e.face1
            f2 = e.face2
            if (f1.h < f2.h):
                f1, f2 = f2, f1
            dV = f1.area * min(f1.h - f2.h, ah*maxH)
            f1.ice_chunk -= dV * beta
            f2.ice_chunk += dV * beta

    def update_surface_nodal_positions(self, mesh):
        """
        Update surface nodal positions.

        Source: [1] IV.A.7

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        for n in mesh.nodes:

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

    def redistribute_remaining_volume(self, mesh):
        """
        Redistribute remaining volume.

        Source: [1] IV.A.8

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        for f in mesh.faces:

            # [1] IV.A.8 formula (14)
            f.target_ice -= mth.pseudoprism_volume(f.nodes[0].old_p, f.nodes[1].old_p, f.nodes[2].old_p,
                                                   f.nodes[0].p, f.nodes[1].p, f.nodes[2].p)
        for f in mesh.faces:
            if f.target_ice < 0:
                f_max = max(f.neighbour_faces, key=lambda nf:nf.target_ice)
                f_max.target_ice += f.target_ice
                if f_max.target_ice < 0:
                    f_max.target_ice = 0
                f.target_ice = 0

    def null_space_smoothing(self, mesh, threshold, safety_factor=0.2):
        """
        Null-space smoothing.

        Parameters
        __________
        mesh : Mesh
            Mesh.
        threshold : float
            threshold to separate primary and null space
        safety_factor: float
            0 < safety_factor < 1

        """

        for n in mesh.nodes:
            node_calculate_A_and_b(n)
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

    def null_space_smoothing_accretion_volume_interpolation(self, mesh):
        """
        Null-space smoothing accretiion volume interpolation.

        TODO: [1] IV.A.10

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """
        for e in mesh.edges:
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

    def final_volume_correction_step(self, mesh):
        """
        Final volume correction step.

        Parameters
        ----------
        mesh : Mesh
            Mesh.
        """

        self.null_space_smoothing_accretion_volume_interpolation(mesh)
