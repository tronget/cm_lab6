def runge_error(yh, yh2, p):
    if len(yh2) < 2 * len(yh) - 1:
        return [float("nan")] * len(yh)
    errs = []
    factor = 2 ** p - 1
    for i in range(len(yh)):
        err = abs(yh2[2 * i] - yh[i]) / factor
        errs.append(err)
    return errs
