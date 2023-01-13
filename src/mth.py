import math


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

    n, roots = solve_quadratic_equation(a, b, c)

    if (n == 1) or (n == 2):
        return min(filter(lambda r: r > 0.0, roots))
    else:
        return None


if __name__ == '__main__':

    # Linear equation.
    assert solve_linear_equation(1.0, 1.0) == (1, [-1.0])
    assert solve_linear_equation(0.0, 0.0) == (math.inf, [])
    assert solve_linear_equation(0.0, 1.0) == (0, [])

    # Quadratic equation.
    assert solve_quadratic_equation(1.0, 0.0, -1.0) == (2, [1.0, -1.0])
    assert solve_quadratic_equation(1.0, -2.0, 1.0) == (1, [1.0])
    assert solve_quadratic_equation(1.0, 0.0, 1.0) == (0, [])
