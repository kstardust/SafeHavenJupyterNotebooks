import numpy as np


def bisection_root(f, x1, x2, max_iters=200, eps=1e-10):
    y1, y2 = f(x1), f(x2)
    if y1 == 0:
        return x1
    if y2 == 0:
        return x2

    if y1 * y2 > 0:
        raise ValueError("f(x1) and f(x2) must have different signs")

    for _ in range(max_iters):
        mid = (x1 + x2) * 0.5
        ym = f(mid)
        if abs(ym) < eps or abs(x2-x1) < eps:
            return mid
        if y1 * ym < 0:
            y2, x2 = ym, mid
        else:
            y1, x1 = ym, mid

    return None


# bernouli's expected utility
def bev(xr, w, payoffs):
    payoffs = np.asarray(payoffs)
    n = len(payoffs)
    ending_balances = (1.0-xr) + payoffs/w
    return np.exp(np.log(np.prod(ending_balances))/n)


def run_case(name, f, a, b, expected, tol=1e-8):
    root = bisection_root(f, a, b)
    err = abs(root - expected)
    print(f"{name}: root={root:.12f} err={err:.3e} f(root)={f(root):.3e}")
    assert err < tol
    return root


def test():
    cases = [
        ("cubic", lambda x: x**3 - x - 2, 1.0, 2.0, 1.5213797068045676, 1e-8),
        ("sqrt2", lambda x: x**2 - 2, 1.0, 2.0, np.sqrt(2.0), 1e-10),
        ("cos-x", lambda x: np.cos(x) - x, 0.0, 1.0, 0.7390851332151607, 1e-8),
        ("linear", lambda x: x - 5, 0.0, 10.0, 5.0, 1e-12),
    ]

    _ = [run_case(*case) for case in cases]

    # Endpoint root case
    root_endpoint = bisection_root(lambda x: x * (x - 1), 0.0, 2.0)
    print(f"endpoint: root={root_endpoint:.12f}")
    assert root_endpoint == 0.0

    # Error case: no sign change
    try:
        bisection_root(lambda x: x**2 + 1, -1.0, 1.0)
        raise AssertionError("expected ValueError for no sign change")
    except ValueError:
        pass