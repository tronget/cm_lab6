import math

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from methods import *
from ode import ODE_PROBLEMS
from utils import runge_error


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
            x0 = float(self.x0_edit.text().replace(",", "."))
            y0 = float(self.y0_edit.text().replace(",", "."))
            xn = float(self.xn_edit.text().replace(",", "."))
            h = float(self.h_edit.text().replace(",", "."))
            epsilon = float(self.err_edit.text().replace(",", "."))
            if xn <= x0 or h <= 0 or epsilon <= 0:
                raise ValueError
            return ode, x0, y0, xn, h, epsilon
        except ValueError:
            raise

    def on_compute(self):
        try:
            ode, x0, y0, xn, h, epsilon = self.parse_input()
        except ValueError:
            QMessageBox.critical(self, "Input error", "Некорректный ввод!")
            return

        match self.ode_combo.currentIndex():
            case 0:
                c = (y0 - (x0 + 1) ** 2) / math.exp(x0)
            case 1:
                c = (y0 + x0 + 1) / math.exp(x0)
            case 2:
                c = (y0 + 0.5 * (math.sin(x0) + math.cos(x0))) / math.exp(x0)

        # Numerical solutions at step h
        xs_e, ys_e = euler_method(ode, x0, y0, xn, h)
        xs_ie, ys_ie = improved_euler_method(ode, x0, y0, xn, h)
        xs_m, ys_m = milne_method(ode, x0, y0, xn, h)

        # Finer step for Runge error (h/2)
        xs_e2, ys_e2 = euler_method(ode, x0, y0, xn, h / 2)
        xs_ie2, ys_ie2 = improved_euler_method(ode, x0, y0, xn, h / 2)

        err_e = runge_error(ys_e, ys_e2, p=1)
        err_ie = runge_error(ys_ie, ys_ie2, p=2)
        err_milne = [abs(ode.exact(x, c) - y) for x, y in zip(xs_m, ys_m)]

        xs_exact = [x0 + i * h / 10 for i in range(int((xn - x0) / (h / 10)) + 1)]
        ys_exact = [ode.exact(x, c) for x in xs_exact]

        self.canvas.plot(xs_exact, ys_exact, {
            "Euler": (xs_e, ys_e),
            "Improved Euler": (xs_ie, ys_ie),
            "Milne": (xs_m, ys_m),
        })

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
