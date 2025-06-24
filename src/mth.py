import math
import cmath
import numpy as np

#---------------------------------------------------------------------------------------------------

# Small eps.
EPS = 1.0e-10

#---------------------------------------------------------------------------------------------------

def intervals_intersect(a, b):
    """
    Check intervals intersection.

    Parameters
    ----------
    a : (float, float)
        First interval.
    b : (float, float)
        Second interval.

    Returns
    -------
    bool
        True - if intervals intersect,
        False - if intervals do not intersect.
    """

    not_intersect = (a[1] < b[0]) or (b[1] < a[0])

    return not not_intersect

#---------------------------------------------------------------------------------------------------

def solve_linear_equation(a, b):
    """
    Solve equation a * x + b = 0

    Parameters
    ----------
    a : float
        Parameter for x.
    b : float
        Free parameter.

    Returns
    -------
    Tuple (num, roots)
    num : int
        Number of roots (0, 1, or math.inf).
    roots : [float]
        Roots array (for 1 root) or [].

    Examples
    --------
    0, []
    1, [r]
    math.inf, []
    """

    if a != 0.0:

        # True linear equation, one root.
        return 1, [-b / a]

    # Equation b = 0.

    if b == 0.0:
        return math.inf, []
    else:
        return 0, []

#---------------------------------------------------------------------------------------------------

def solve_quadratic_equation(a, b, c):
    """
    Solve equation a * x^2 + b * x + c = 0

    Parameters
    ----------
    a : float
        Parameter for x^2.
    b : float
        Parameter for x.
    c : float
        Free parameter.

    Returns
    -------
    Tuple (num, roots)
    num : int
        Number of real roots (0, 1, 2, or math.inf).
    roots : [float]
        Real roots array (for 1 root or 2 roots) or [].

    Examples
    --------
    0, []
    1, [r]
    2, [r1, r2]
    math.inf, []
    """

    if a != 0.0:

        # True quadratic equation.

        # Discriminant.
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
    else:
        return solve_linear_equation(b, c)

#---------------------------------------------------------------------------------------------------

def quadratic_equation_smallest_positive_root(a, b, c) -> float:
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

    _, roots = solve_quadratic_equation(a, b, c)
    roots = [r for r in roots if r > 0.0]

    if not roots:
        return None
    else:
        return min(roots)

#---------------------------------------------------------------------------------------------------

def solve_cubic_equation(a, b, c, d):
    """
    Solve equation a * x^3 + b * x^2 + c * x + d = 0

    Parameters
    ----------
    a : float
        Parameter for x^3.
    b : float
        Parameter for x^2.
    c : float
        Parameter for x.
    d : float
        Free parameter.

    Returns
    -------
    Tuple (num, roots)
    num : int
        Number of real roots (0, 1, 2, 3, or math.inf).
    roots : [float]
        Complex roots array (for 1, 2 or 3 roots) or [].

    Examples
    --------
    0, []
    1, [r]
    2, [r1, r2]
    3, [r1, r2, r3]
    math.inf, []
    """

    if a != 0.0:

        def cbrt(p):
            r1 = p ** (1.0 / 3.0)
            s = math.sqrt(3.0) * 1j
            r2 = r1 * (-1.0 + s) / 2.0
            r3 = r1 * (-1.0 - s) / 2.0
            return [r1, r2, r3]

        # True cubic equation.
        # Solve it with Cardano method.
        p = (3.0 * a * c - b**2) / (3.0 * a**2)
        q = (2.0 * b**3 - 9.0 * a * b * c + 27.0 * a**2 * d) / (27.0 * a**3)

        Q = (p / 3.0)**3.0 + (q / 2.0)**2.0
        sQ = cmath.sqrt(Q)
        alfa = cbrt(-q / 2.0 + sQ)
        beta = cbrt(-q / 2.0 - sQ)

        roots = []
        for i in alfa:
            for j in beta:
                if abs((i * j) + p / 3.0) < EPS:
                    x = i + j - b / (3.0 * a)
                    roots.append(x)

                    # Go to next alfa.
                    break

        return len(roots), roots

    else:
        return solve_quadratic_equation(b, c, d)

#---------------------------------------------------------------------------------------------------

def tetrahedra_volume(a, b, c, d) -> float:
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

#---------------------------------------------------------------------------------------------------

def pseudoprism_volume(a, b, c, na, nb, nc) -> float:
    """
    Pseudoprism volume.

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

    volume = tetrahedra_volume(a, b, c, nc) \
        + tetrahedra_volume(a, b, nb, nc) \
        + tetrahedra_volume(a, na, nb, nc)

    return volume

#---------------------------------------------------------------------------------------------------

def find_extremums_kx_qx2qxq(k_x, q_x2, q_x, q):
    """
    Find extremums point for function
    f(x) = k_x * x + sqrt(q_x2 * x^2 + q_x * x + q).

    Parameters
    ----------
    k_x : float
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

    return r_sq + list(filter(lambda x: (sq(x) > 0.0) and (abs(df(x)) <= EPS), r_df))

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Linear equation.
    assert solve_linear_equation(1.0, 1.0) == (1, [-1.0])
    assert solve_linear_equation(0.0, 0.0) == (math.inf, [])
    assert solve_linear_equation(0.0, 1.0) == (0, [])

    # Quadratic equation.
    assert solve_quadratic_equation(1.0, 0.0, -1.0) == (2, [1.0, -1.0])
    assert solve_quadratic_equation(1.0, -2.0, 1.0) == (1, [1.0])
    assert solve_quadratic_equation(1.0, 0.0, 1.0) == (0, [])

    # Cubic equation.
    print(solve_cubic_equation(1.0, -3.0, 3.0, -1.0))
    print(solve_cubic_equation(1.0, 0.0, 0.0, -1.0))
    print(solve_cubic_equation(1.0, 0.0, 0.0, 0.0))
    print(solve_cubic_equation(1.0, 1.0, -1.0, -1.0))

#---------------------------------------------------------------------------------------------------
