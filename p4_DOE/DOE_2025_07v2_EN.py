import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton, QLabel,
                             QLineEdit, QDialog, QScrollArea, QGroupBox, QGridLayout,
                             QHeaderView, QSizePolicy, QCheckBox, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from pyDOE2 import pbdesign, fracfact, fullfact, ff2n

# Set global font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ======================== PB Design Matrices ========================
PB_DESIGNS = {
    12: np.array([
        [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1],
        [1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1],
        [-1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1],
        [1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1],
        [1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1],
        [-1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1],
        [-1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1],
        [-1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1],
        [1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
        [-1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]),
    16: np.array([
        [1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1],
        [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1],
        [1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1],
        [-1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
        [-1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1],
        [1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1],
        [-1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1],
        [1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
        [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
        [-1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
        [-1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1],
        [-1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1]
    ]),
    20: np.array([
        [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1],
        [1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1],
        [1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1],
        [1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1],
        [-1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1],
        [1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1],
        [-1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1],
        [1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1],
        [-1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1],
        [-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
        [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
        [-1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1],
        [-1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1],
        [-1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1],
        [1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1],
        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1]
    ])
}


class DesignTypeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Experimental Design Type")
        self.setGeometry(300, 300, 450, 250)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Please select the experimental design type")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Design type selection
        type_group = QGroupBox("Design Type")
        type_layout = QVBoxLayout()

        self.design_type_combo = QComboBox()
        self.design_type_combo.addItem("Full Factorial Design")
        self.design_type_combo.addItem("Fractional Factorial Design")
        self.design_type_combo.addItem("Plackett-Burman Design")
        type_layout.addWidget(self.design_type_combo)

        # Center points option
        center_layout = QHBoxLayout()
        self.center_points_check = QCheckBox("Add center points")
        self.center_points_check.setChecked(True)
        center_layout.addWidget(self.center_points_check)

        self.center_points_edit = QLineEdit("3")
        self.center_points_edit.setValidator(QIntValidator(1, 10))
        self.center_points_edit.setFixedWidth(50)
        center_layout.addWidget(self.center_points_edit)
        center_layout.addWidget(QLabel("points"))
        center_layout.addStretch()

        type_layout.addLayout(center_layout)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Explanation
        explanation = QLabel("Explanation:\n"
                             "- Full Factorial: Tests all factor combinations, runs = 2^k\n"
                             "- Fractional Factorial: Tests partial combinations, suitable for many factors\n"
                             "- Plackett-Burman: Efficient for factor screening")
        explanation.setStyleSheet("color: blue; font-size: 12px;")
        layout.addWidget(explanation)

        # Button area
        btn_layout = QHBoxLayout()
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_design_type(self):
        return self.design_type_combo.currentIndex()

    def add_center_points(self):
        return self.center_points_check.isChecked()

    def center_points_count(self):
        try:
            return int(self.center_points_edit.text())
        except:
            return 3


class VariableInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Experimental Variables Setup")
        self.setGeometry(300, 300, 650, 450)

        self.variables = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Please enter experimental variable information (at least 2 variables, unique names)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Variable input area
        self.vars_group = QGroupBox("Variable Settings")
        self.vars_layout = QGridLayout()

        # Headers
        headers = ["Variable Name", "Minimum Value", "Maximum Value", ""]
        for col, text in enumerate(headers):
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold;")
            self.vars_layout.addWidget(label, 0, col)

        # Add initial rows
        self.add_variable_row()
        self.add_variable_row()

        self.vars_group.setLayout(self.vars_layout)
        scroll_layout.addWidget(self.vars_group)

        # Add variable button
        add_btn = QPushButton("Add Variable")
        add_btn.clicked.connect(self.add_variable_row)
        scroll_layout.addWidget(add_btn)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Button area
        btn_layout = QHBoxLayout()
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def add_variable_row(self):
        row = self.vars_layout.rowCount()

        # Skip header row
        if row == 0: row = 1

        name_edit = QLineEdit(f"x{row}")
        min_edit = QLineEdit("0")
        max_edit = QLineEdit("10")

        # Set validation - numbers only
        min_edit.setValidator(QDoubleValidator())
        max_edit.setValidator(QDoubleValidator())

        # Delete button
        delete_btn = QPushButton("Delete")
        delete_btn.setFixedWidth(60)

        # Store row index for deletion
        delete_btn.row_index = len(self.variables)
        delete_btn.clicked.connect(self.delete_variable_row)

        self.vars_layout.addWidget(name_edit, row, 0)
        self.vars_layout.addWidget(min_edit, row, 1)
        self.vars_layout.addWidget(max_edit, row, 2)
        self.vars_layout.addWidget(delete_btn, row, 3)

        # Store references
        self.variables.append({
            'name_edit': name_edit,
            'min_edit': min_edit,
            'max_edit': max_edit,
            'delete_btn': delete_btn
        })

    def delete_variable_row(self):
        btn = self.sender()
        row_index = btn.row_index

        # Remove widgets from layout
        for key in ['name_edit', 'min_edit', 'max_edit', 'delete_btn']:
            widget = self.variables[row_index][key]
            self.vars_layout.removeWidget(widget)
            widget.deleteLater()

        # Remove from variables list
        del self.variables[row_index]

        # Update row indices for remaining buttons
        for idx, var in enumerate(self.variables):
            var['delete_btn'].row_index = idx

    def get_variables(self):
        vars_info = []
        var_names = set()

        for var in self.variables:
            name = var['name_edit'].text().strip()
            min_val = var['min_edit'].text()
            max_val = var['max_edit'].text()

            if not name or not min_val or not max_val:
                continue

            # Check for duplicate names
            if name in var_names:
                QMessageBox.warning(self, "Duplicate Name", f"Variable name '{name}' already exists. Please use a different name.")
                return []
            var_names.add(name)

            try:
                min_val = float(min_val)
                max_val = float(max_val)

                if min_val >= max_val:
                    QMessageBox.warning(self, "Range Error", f"Minimum value for '{name}' must be less than maximum value.")
                    return []

                vars_info.append({
                    'name': name,
                    'min': min_val,
                    'max': max_val
                })
            except ValueError:
                QMessageBox.warning(self, "Input Error", f"Invalid input for variable '{name}'. Please enter numeric values.")
                return []

        if len(vars_info) < 2:
            QMessageBox.warning(self, "Insufficient Variables", "At least 2 variables are required.")
            return []

        return vars_info


class DesignResultWindow(QMainWindow):
    def __init__(self, design_df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Experimental Design Plan")
        self.setGeometry(100, 100, 1200, 800)

        self.full_design_df = design_df  # Save complete data
        self.design_df = self.get_actual_values_df()  # Show actual values by default
        self.plot_type = "parallel"  # Default to parallel coordinates

        self.initUI()

    def get_actual_values_df(self):
        """Return DataFrame with only actual values and run type"""
        # Get all actual value columns
        actual_cols = [col for col in self.full_design_df.columns if not col.endswith('_coded')]
        return self.full_design_df[actual_cols]

    def get_coded_values_df(self):
        """Return DataFrame with only coded values"""
        # Get all coded value columns
        coded_cols = [col for col in self.full_design_df.columns if col.endswith('_coded')]
        coded_df = self.full_design_df[coded_cols].copy()
        # Rename columns (remove _coded suffix)
        coded_df.columns = [col.replace('_coded', '') for col in coded_df.columns]
        coded_df['Run Type'] = self.full_design_df['Run Type']
        return coded_df

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("Experimental Design Plan and Design Matrix Plot")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title)

        # Control panel
        control_layout = QHBoxLayout()

        # Show coded values option
        self.show_coded_check = QCheckBox("Show coded values")
        self.show_coded_check.stateChanged.connect(self.toggle_coded_values)
        control_layout.addWidget(self.show_coded_check)

        # Plot type selection
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(QLabel("Plot Type:"))

        self.plot_combo = QComboBox()
        self.plot_combo.addItem("Parallel Coordinates")
        self.plot_combo.addItem("Scatter Plot Matrix")
        self.plot_combo.currentIndexChanged.connect(self.change_plot_type)
        plot_layout.addWidget(self.plot_combo)

        control_layout.addLayout(plot_layout)
        control_layout.addStretch()

        # Export button
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self.export_csv)
        control_layout.addWidget(export_btn)

        main_layout.addLayout(control_layout)

        # Table
        table_label = QLabel("Experimental Parameters Table:")
        main_layout.addWidget(table_label)

        # Add explanation
        explanation = QLabel("Note: Coded values -1 represent minimum, 0 represents center point, 1 represents maximum")
        explanation.setStyleSheet("color: blue; font-size: 12px;")
        main_layout.addWidget(explanation)

        self.table = QTableWidget()
        self.update_table()

        # Adjust table size
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.table)

        # Plot area
        fig_label = QLabel("Design Matrix Plot:")
        main_layout.addWidget(fig_label)

        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.plot_design_matrix()
        main_layout.addWidget(self.canvas)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def update_table(self):
        """Update table display"""
        df = self.design_df
        self.table.clear()

        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))

        # Set headers
        self.table.setHorizontalHeaderLabels(df.columns)

        # Populate data
        for i, row in enumerate(df.values):
            for j, value in enumerate(row):
                # Format float display
                if isinstance(value, float):
                    # Choose appropriate precision based on value magnitude
                    if abs(value) < 0.001 or abs(value) >= 10000:
                        item_text = f"{value:.4f}"
                    elif abs(value) < 1:
                        item_text = f"{value:.3f}"
                    else:
                        item_text = f"{value:.2f}"
                else:
                    item_text = str(value)

                item = QTableWidgetItem(item_text)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

    def toggle_coded_values(self, state):
        """Toggle between actual and coded values"""
        if state == Qt.Checked:
            self.design_df = self.get_coded_values_df()
        else:
            self.design_df = self.get_actual_values_df()

        self.update_table()

    def change_plot_type(self, index):
        """Change plot type"""
        self.plot_type = "parallel" if index == 0 else "scatter"
        self.plot_design_matrix()

    def plot_design_matrix(self):
        """Plot design matrix visualization"""
        self.figure.clear()

        # Extract actual value columns
        actual_cols = [col for col in self.design_df.columns if col != 'Run Type']
        if not actual_cols:
            return

        plot_data = self.design_df[actual_cols].copy()

        if self.plot_type == "parallel":
            self.plot_parallel_coordinates(plot_data)
        else:
            self.plot_scatter_matrix(plot_data)

        self.canvas.draw()

    def plot_parallel_coordinates(self, plot_data):
        """Plot parallel coordinates"""
        ax = self.figure.add_subplot(111)

        # Create parallel coordinates plot
        n_vars = len(plot_data.columns)
        n_runs = len(plot_data)

        # Set color map
        colors = plt.cm.viridis(np.linspace(0, 1, n_runs))

        # Plot each experiment line
        for i in range(n_runs):
            values = plot_data.iloc[i].values
            ax.plot(range(n_vars), values, color=colors[i], marker='o', alpha=0.7)

        # Set axis labels
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(plot_data.columns, rotation=45)
        ax.set_xlabel("Variables")
        ax.set_ylabel("Actual Values")
        ax.set_title("Experimental Design Matrix - Parallel Coordinates")
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add legend (experiment numbers)
        handles = []
        for i in range(n_runs):
            handles.append(plt.Line2D([0], [0], color=colors[i], marker='o', label=f'Run {i + 1}'))
        ax.legend(handles=handles, title="Run Number", bbox_to_anchor=(1.005, 1), loc='upper left')

    def plot_scatter_matrix(self, plot_data):
        """Plot scatter plot matrix"""
        from pandas.plotting import scatter_matrix

        ax = scatter_matrix(plot_data, figsize=(10, 8), diagonal='hist',
                            marker='o', alpha=0.7, ax=self.figure.gca())

        # Set title
        self.figure.suptitle("Experimental Design Matrix - Scatter Plot Matrix", fontsize=16)

        # Adjust layout
        self.figure.tight_layout()
        self.figure.subplots_adjust(top=0.95)

    def export_csv(self):
        """Export design to CSV file"""
        file_name = "experimental_design.csv"
        self.full_design_df.to_csv(file_name, index=False)
        self.statusBar().showMessage(f"Design exported to {file_name}", 3000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experimental Design System")
        self.setGeometry(100, 100, 500, 300)

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Experimental Design System")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Description
        desc = QLabel("This system provides various experimental design methods: Full Factorial, Fractional Factorial, and Plackett-Burman designs")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        # Start button
        start_btn = QPushButton("Start Experimental Design")
        start_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        start_btn.clicked.connect(self.start_design)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        # Footer information
        footer = QLabel("Implemented with PyQt5 and pyDOE2 | Supports multiple experimental design methods")
        footer.setStyleSheet("font-size: 10px; color: gray; margin-top: 30px;")
        layout.addWidget(footer, alignment=Qt.AlignCenter)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def start_design(self):
        # Step 1: Select design type
        type_dialog = DesignTypeDialog(self)
        if type_dialog.exec_() != QDialog.Accepted:
            return

        design_type = type_dialog.get_design_type()
        add_center_points = type_dialog.add_center_points()
        center_points_count = type_dialog.center_points_count()

        # Step 2: Input variables
        var_dialog = VariableInputDialog(self)
        if var_dialog.exec_() == QDialog.Accepted:
            variables = var_dialog.get_variables()
            if not variables:
                return

            design_df = self.generate_experimental_design(variables, design_type,
                                                          add_center_points, center_points_count)
            if design_df is not None:
                self.result_window = DesignResultWindow(design_df)
                self.result_window.show()

    def generate_experimental_design(self, variables, design_type, add_center_points=True, center_points_count=3):
        k = len(variables)  # Number of variables
        design_matrix = None

        try:
            # 1. Generate design matrix based on design type
            if design_type == 0:  # Full factorial design
                # Full factorial generates 2^k experiments
                design_matrix = ff2n(k)  # Returns matrix with -1 and 1
                runs = 2 ** k

            elif design_type == 1:  # Fractional factorial design
                # Use fractional factorial, runs = 2^(k-1)
                if k < 2:
                    QMessageBox.warning(self, "Insufficient Variables", "Fractional factorial design requires at least 2 variables")
                    return None

                # Create generator string
                gen_str = " ".join([chr(97 + i) for i in range(k - 1)])
                design_matrix = fracfact(gen_str)

                # Ensure design matrix is 2D array
                if design_matrix.ndim == 1:
                    design_matrix = design_matrix.reshape(-1, k - 1)

                # Add last variable
                last_var = np.ones(len(design_matrix))
                design_matrix = np.column_stack((design_matrix, last_var))
                runs = len(design_matrix)

            else:  # Plackett-Burman design
                if k <= 11:
                    # Use Plackett-Burman design
                    design_matrix = pbdesign(k)
                    if design_matrix is None:
                        QMessageBox.warning(self, "Design Error", "Plackett-Burman design does not support this number of variables")
                        return None

                    # Ensure design matrix is 2D array
                    if design_matrix.ndim == 1:
                        design_matrix = design_matrix.reshape(-1, k)

                    # PB design returns 0/1, convert to -1/1
                    design_matrix = design_matrix * 2 - 1
                    runs = len(design_matrix)
                else:
                    # For more than 11 variables
                    if k in PB_DESIGNS:
                        # Use predefined PB design
                        design_matrix = PB_DESIGNS[k]
                        # Take required number of columns
                        design_matrix = design_matrix[:, :k]
                        runs = len(design_matrix)
                    else:
                        # Use fractional factorial as fallback
                        QMessageBox.warning(self, "Design Warning",
                                            f"Plackett-Burman design does not support {k} variables, automatically switching to fractional factorial design")
                        gen_str = " ".join([chr(97 + i) for i in range(k - 1)])
                        design_matrix = fracfact(gen_str)

                        if design_matrix.ndim == 1:
                            design_matrix = design_matrix.reshape(-1, k - 1)

                        last_var = np.ones(len(design_matrix))
                        design_matrix = np.column_stack((design_matrix, last_var))
                        runs = len(design_matrix)

                        # Convert to -1/1
                        design_matrix = design_matrix * 2 - 1

            # 2. Add center points if needed
            if add_center_points:
                center_point = np.zeros((center_points_count, k))
                full_design_matrix = np.vstack([design_matrix, center_point])
                total_runs = runs + center_points_count
            else:
                full_design_matrix = design_matrix
                total_runs = runs

            # 3. Create result DataFrame
            design_data = {}

            # Add actual and coded value columns
            for i, var in enumerate(variables):
                name = var['name']
                min_val = var['min']
                max_val = var['max']

                # Coded values column
                coded_values = full_design_matrix[:, i]
                design_data[f"{name}_coded"] = coded_values

                # Critical fix: Normalize coded values
                # Ensure all values are -1, 0, or 1
                coded_values = np.where(coded_values < -1, -1, coded_values)
                coded_values = np.where(coded_values > 1, 1, coded_values)
                # Round to nearest integer (handle floating point precision issues)
                coded_values = np.round(coded_values).astype(int)

                design_data[f"{name}_coded"] = coded_values


                # Convert to actual values
                actual_values = []
                for coded in coded_values:
                    if coded == 0:  # Center point
                        actual = (min_val + max_val) / 2.0
                    else:
                        # Ensure coded value is within [-1, 1] range
                        clamped_coded = max(min(coded, 1.0), -1.0)
                        actual = min_val + (clamped_coded + 1) * (max_val - min_val) / 2.0
                    actual_values.append(actual)

                design_data[name] = actual_values

            # Add run type column
            if add_center_points:
                run_types = ['Corner Point'] * total_runs
                for i in range(total_runs - center_points_count, total_runs):  # Last ones are center points
                    run_types[i] = 'Center Point'
            else:
                run_types = ['Corner Point'] * total_runs
            design_data['Run Type'] = run_types

            # Create DataFrame
            design_df = pd.DataFrame(design_data)

            # Reorder columns
            ordered_columns = []
            for var in variables:
                ordered_columns.append(var['name'])
                ordered_columns.append(f"{var['name']}_coded")
            ordered_columns.append('Run Type')

            return design_df[ordered_columns]

        except Exception as e:
            QMessageBox.critical(self, "Design Generation Error", f"Error generating design: {str(e)}")
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern UI style

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())