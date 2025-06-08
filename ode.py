import math


class ODEProblem:
    def __init__(self, name, rhs, exact):
        self.name = name
        self.f = rhs
        self.exact = exact


ode1 = ODEProblem(
    "y' = y - x^2 + 1",
    rhs=lambda x, y: y - x * x + 1,
    exact=lambda x, c: (x + 1) ** 2 + c * math.exp(x),
)

ode2 = ODEProblem(
    "y' = x + y",
    rhs=lambda x, y: x + y,
    exact=lambda x, c: c * math.exp(x) - x - 1,
)

ode3 = ODEProblem(
    "y' = sin(x) + y",
    rhs=lambda x, y: math.sin(x) + y,
    exact=lambda x, c: c * math.exp(x) - 0.5 * (math.sin(x) + math.cos(x)),
)

ODE_PROBLEMS = [ode1, ode2, ode3]
