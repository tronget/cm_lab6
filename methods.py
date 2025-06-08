def euler_method(ode, x0, y0, xn, h):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    while x < xn:
        y = y + h * ode.f(x, y)
        x = x + h
        xs.append(x)
        ys.append(y)
    return xs, ys


def improved_euler_method(ode, x0, y0, xn, h):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    while x < xn:
        k1 = ode.f(x, y)
        y_pred = y + h * k1
        k2 = ode.f(x + h, y_pred)
        y = y + (h / 2) * (k1 + k2)
        x = x + h
        xs.append(x)
        ys.append(y)
    return xs, ys


def milne_method(ode, x0, y0, xn, h):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(3):
        k1 = ode.f(x, y)
        y_pred = y + h * k1
        k2 = ode.f(x + h, y_pred)
        y = y + (h / 2) * (k1 + k2)
        x = x + h
        xs.append(x)
        ys.append(y)

    f_vals = [ode.f(xx, yy) for xx, yy in zip(xs, ys)]

    i = 3
    while xs[-1] + 1e-14 < xn:
        x_next = xs[-1] + h
        y_pred = ys[i - 3] + (4 * h / 3) * (2 * f_vals[i] - f_vals[i - 1] + 2 * f_vals[i - 2])
        f_pred = ode.f(x_next, y_pred)
        y_corr = ys[i - 1] + (h / 3) * (f_vals[i - 1] + 4 * f_vals[i] + f_pred)
        xs.append(x_next)
        ys.append(y_corr)
        f_vals.append(ode.f(x_next, y_corr))
        i += 1
    return xs, ys
