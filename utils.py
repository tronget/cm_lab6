def runge_error(yh, yh2, p):
    if len(yh2) < 2 * len(yh) - 1:
        return [float("nan")] * len(yh)
    errs = []
    factor = 2 ** p - 1
    for i in range(len(yh)):
        err = abs(yh2[2 * i] - yh[i]) / factor
        errs.append(err)
    return errs


def solve(method, ode, x0, y0, xn, h, eps, p, gui):
    MAX_POINTS = 150000
    while True:
        n = int(round((xn - x0) / h))
        if n >= MAX_POINTS:
            print("Невозможно вычислить")
            raise OverflowError
        xs1, ys1 = method(ode, x0, y0, xn, h)
        xs2, ys2 = method(ode, x0, y0, xn, h / 2)
        denom = 2 ** p - 1
        err = max(abs(a - b) / denom for a, b in zip(ys1, ys2[::2]))

        if err <= eps:
            return xs1, ys1, err, h

        h /= 2
