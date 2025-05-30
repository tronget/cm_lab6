# Lab 6 – Numerical Solution of ODEs (Euler, Improved Euler, Milne)
# Author: ChatGPT
# Requirements: Python 3.10+, PyQt5, matplotlib
# External numerical libraries (NumPy/SciPy, etc.) are NOT used – all algorithms are coded manually.

import sys
import math
from typing import Callable, List, Tuple

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


###############################################################################
# ODE problems (at least three) with analytic solutions for error estimation  #
###############################################################################

class ODEProblem:
    """Container for an ODE of form y' = f(x, y) with an analytic solution."""

    def __init__(self, name: str, rhs: Callable[[float, float], float], exact: Callable[[float], float]):
        self.name = name
        self.f = rhs
        self.exact = exact


# 1) y' = y - x^2 + 1; exact: y = (x + 1)^2 - 0.5 * e^x
ode1 = ODEProblem(
    "y' = y - x^2 + 1",
    rhs=lambda x, y: y - x * x + 1,
    exact=lambda x: (x + 1) ** 2 - 0.5 * math.exp(x),
)

# 2) y' = x + y; exact through integrating factor → y = 2 e^x - x - 1 (for y(0)=1)
ode2 = ODEProblem(
    "y' = x + y",
    rhs=lambda x, y: x + y,
    exact=lambda x: 2 * math.exp(x) - x - 1,
)

# 3) y' = sin(x) + y; exact with integrating factor → y = (2 - 0.5*cos(0) - 0.5*sin(0)) e^x + 0.5 (sin x - cos x)
#    For initial condition y(0)=2, constant C = 1.5. Compact exact formula:
ode3 = ODEProblem(
    "y' = sin(x) + y",
    rhs=lambda x, y: math.sin(x) + y,
    exact=lambda x: 1.5 * math.exp(x) + 0.5 * (math.sin(x) - math.cos(x)),
)

ODE_PROBLEMS = [ode1, ode2, ode3]


################################################################################
# Numerical algorithms (no NumPy)                                              #
################################################################################

def euler_method(ode, x0, y0, xn, h, epsilon):
    """Classic (explicit) Euler method – first‑order."""
    xs, ys = [x0], [y0]
    x, y = x0, y0
    while epsilon < xn - x:
        y = y + h * ode.f(x, y)
        x = x + h
        xs.append(x)
        ys.append(y)
    return xs, ys


def improved_euler_method(ode, x0, y0, xn, h, epsilon):
    """Improved Euler (Heun) – second‑order."""
    xs, ys = [x0], [y0]
    x, y = x0, y0
    while epsilon < xn - x:
        k1 = ode.f(x, y)
        y_pred = y + h * k1  # predictor (Euler)
        k2 = ode.f(x + h, y_pred)
        y = y + (h / 2) * (k1 + k2)
        x = x + h
        xs.append(x)
        ys.append(y)
    return xs, ys


def milne_method(ode, x0, y0, xn, h, epsilon):
    """Milne's predictor‑corrector – fourth‑order global accuracy.

    We start by obtaining the first three additional points with the improved Euler method.
    """
    # Bootstrap with Improved Euler for four initial points (indices 0..3)
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

    # Pre‑compute derivatives f_i
    f_vals = [ode.f(xx, yy) for xx, yy in zip(xs, ys)]

    i = 3  # current last known index
    while xs[-1] + 1e-14 < xn:
        x_next = xs[-1] + h
        # Predictor (Milne)
        y_pred = ys[i - 3] + (4 * h / 3) * (2 * f_vals[i] - f_vals[i - 1] + 2 * f_vals[i - 2])
        f_pred = ode.f(x_next, y_pred)
        # Corrector
        y_corr = ys[i - 1] + (h / 3) * (f_vals[i - 1] + 4 * f_vals[i] + f_pred)
        xs.append(x_next)
        ys.append(y_corr)
        f_vals.append(ode.f(x_next, y_corr))
        i += 1
    return xs, ys


################################################################################
# Error estimation                                                              #
################################################################################


def runge_error(yh: List[float], yh2: List[float], p: int) -> List[float]:
    """Runge estimate between step h and h/2 solutions for method of order p."""
    if len(yh2) < 2 * len(yh) - 1:
        return [float("nan")] * len(yh)
    errs = []
    factor = 2 ** p - 1
    for i in range(len(yh)):
        err = abs(yh2[2 * i] - yh[i]) / factor
        errs.append(err)
    return errs


################################################################################
# GUI                                                                          #
################################################################################

class PlotCanvas(FigureCanvas):
    def __init__(self, parent: QWidget = None):
        fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, xs_exact, ys_exact, results):
        self.axes.clear()
        self.axes.plot(xs_exact, ys_exact, label="Exact", linestyle="--")
        for name, (xs, ys) in results.items():
            self.axes.plot(xs, ys, label=name)
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y")
        self.axes.legend()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 6 – Numerical Solution of ODEs")
        self.resize(900, 600)
        # Widgets
        central = QWidget()
        self.setCentralWidget(central)

        # Left panel – controls
        form_layout = QFormLayout()
        self.ode_combo = QComboBox()
        for prob in ODE_PROBLEMS:
            self.ode_combo.addItem(prob.name)
        form_layout.addRow("ODE:", self.ode_combo)

        self.x0_edit = QLineEdit("0.0")
        self.y0_edit = QLineEdit("1.0")
        self.xn_edit = QLineEdit("2.0")
        self.h_edit = QLineEdit("0.1")
        self.err_edit = QLineEdit("0.001")
        form_layout.addRow("x₀:", self.x0_edit)
        form_layout.addRow("y₀:", self.y0_edit)
        form_layout.addRow("xₙ:", self.xn_edit)
        form_layout.addRow("h:", self.h_edit)
        form_layout.addRow("epsilon:", self.err_edit)

        self.compute_btn = QPushButton("Compute")
        form_layout.addRow(self.compute_btn)
        self.compute_btn.clicked.connect(self.on_compute)

        # Table to display values
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "i",
            "x",
            "Euler y",
            "Err Euler",
            "Impr. Euler y",
            "Err Impr.",
            "Milne y",
            "Err Milne",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Figure
        self.canvas = PlotCanvas()

        # Layouts
        left_widget = QWidget()
        left_widget.setLayout(form_layout)
        left_widget.setMaximumWidth(250)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas, stretch=3)
        right_layout.addWidget(self.table, stretch=2)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, stretch=1)

        central.setLayout(main_layout)

    #####################################################################
    # Helpers
    #####################################################################
    def parse_input(self):
        try:
            ode = ODE_PROBLEMS[self.ode_combo.currentIndex()]
            x0 = float(self.x0_edit.text())
            y0 = float(self.y0_edit.text())
            xn = float(self.xn_edit.text())
            h = float(self.h_edit.text())
            epsilon = float(self.err_edit.text())
            if xn <= x0 or h <= 0 or epsilon <= 0:
                raise ValueError
            return ode, x0, y0, xn, h, epsilon
        except ValueError:
            raise

    def on_compute(self):
        try:
            ode, x0, y0, xn, h, epsilon = self.parse_input()
        except ValueError:
            QMessageBox.critical(self, "Input error", "Please check numerical inputs.")
            return

        # Numerical solutions at step h
        xs_e, ys_e = euler_method(ode, x0, y0, xn, h, epsilon)
        xs_ie, ys_ie = improved_euler_method(ode, x0, y0, xn, h, epsilon)
        xs_m, ys_m = milne_method(ode, x0, y0, xn, h, epsilon)

        # Finer step for Runge error (h/2)
        xs_e2, ys_e2 = euler_method(ode, x0, y0, xn, h / 2, epsilon)
        xs_ie2, ys_ie2 = improved_euler_method(ode, x0, y0, xn, h / 2, epsilon)

        err_e = runge_error(ys_e, ys_e2, p=1)
        err_ie = runge_error(ys_ie, ys_ie2, p=2)
        err_milne = [abs(ode.exact(x) - y) for x, y in zip(xs_m, ys_m)]

        # Build exact for plot
        xs_exact = [x0 + i * h / 10 for i in range(int((xn - x0) / (h / 10)) + 1)]
        ys_exact = [ode.exact(x) for x in xs_exact]

        # Plot
        self.canvas.plot(xs_exact, ys_exact, {
            "Euler": (xs_e, ys_e),
            "Improved Euler": (xs_ie, ys_ie),
            "Milne": (xs_m, ys_m),
        })

        # Table update (truncating to min length of lists)
        nrows = min(len(xs_e), len(xs_ie), len(xs_m))
        self.table.setRowCount(nrows)
        for i in range(nrows):
            row_items = [
                str(i),
                f"{xs_e[i]:.5g}",
                f"{ys_e[i]:.5g}",
                f"{err_e[i]:.2e}",
                f"{ys_ie[i]:.5g}",
                f"{err_ie[i]:.2e}",
                f"{ys_m[i]:.5g}",
                f"{err_milne[i]:.2e}",
            ]
            for j, text in enumerate(row_items):
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

        # Optionally show max error dialog
        max_e_err = max(err_e)
        max_ie_err = max(err_ie)
        max_m_err = max(err_milne)
        QMessageBox.information(
            self,
            "Computation finished",
            (
                f"Max errors:\n"
                f"  Euler: {max_e_err:.2e}\n  Improved Euler: {max_ie_err:.2e}\n  Milne: {max_m_err:.2e}"
            ),
        )


################################################################################
# Entrypoint                                                                    #
################################################################################

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
