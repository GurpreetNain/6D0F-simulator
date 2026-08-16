import os
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, 
                             QMessageBox, QLineEdit, QComboBox, QListWidget, 
                             QGroupBox, QFormLayout, QApplication, QTabWidget,
                             QSlider, QScrollArea, QGridLayout)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from src.common.dynamic_loader import load_external_controller
from src.gui.simulation_worker import SimulationWorker
from src.controllers.trajectory_controller import TrajectoryController 
from src.controllers.acceleration_controller import AccelerationController 

class SimulationHost(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swarm Simulation Host")
        self.resize(1100, 900)
        
        self.custom_controller_class = None
        self.swarm_config_list = []  
        self.drone_counter = 1
        self.trajectory_file = None 
        self.log_data = None
        
        # --- Create Tabbed Interface ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        self.sim_tab = QWidget()
        self.analysis_tab = QWidget()
        
        self.tabs.addTab(self.sim_tab, "Simulation Setup")
        self.tabs.addTab(self.analysis_tab, "Post-Flight Analysis")
        self.main_layout.addWidget(self.tabs)
        
        self.sim_layout = QVBoxLayout(self.sim_tab)
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        
        self.setup_trajectory_panel()
        self.setup_expression_panel()
        self.setup_swarm_panel()
        self.setup_execution_panel()
        self.setup_analysis_panel()
        
        self.combo_controller.currentTextChanged.connect(self.update_panel_visibility)
        self.update_panel_visibility(self.combo_controller.currentText())
        
    def setup_trajectory_panel(self):
        self.group_trajectory = QGroupBox("1. Trajectory Setpoints (Strictly for TrajectoryController)")
        layout = QHBoxLayout()
        
        self.lbl_trajectory = QLabel("No CSV loaded. Drones using TrajectoryController will hover.")
        btn_load_traj = QPushButton("Load CSV Setpoints")
        btn_load_traj.clicked.connect(self.open_trajectory_dialog)
        
        layout.addWidget(btn_load_traj)
        layout.addWidget(self.lbl_trajectory)
        self.group_trajectory.setLayout(layout)
        self.sim_layout.addWidget(self.group_trajectory)

    def setup_expression_panel(self):
        self.group_generator = QGroupBox("2. Custom Acceleration Law Generator")
        layout = QVBoxLayout()
        
        toolbar = QHBoxLayout()
        self.combo_vars = QComboBox()
        
        variables_map = [
            ("Time (t)", "t"),
            ("Own ID", "id"),
            ("Own X Position", "x"),
            ("Own Y Position", "y"),
            ("Own Z Position", "z"),
            ("Own X Velocity", "vx"),
            ("Own Y Velocity", "vy"),
            ("Own Z Velocity", "vz"),
            ("If-Else Condition", "10.0 if id == 1 else 0.0"),
            ("Sine", "np.sin()"),
            ("Cosine", "np.cos()"),
            ("Tangent", "np.tan()"),
            ("ArcSine (Inverse Sin)", "np.arcsin()"),
            ("ArcCosine (Inverse Cos)", "np.arccos()"),
            ("ArcTangent (Inverse Tan)", "np.arctan()")
        ]
        
        for display_text, python_code in variables_map:
            self.combo_vars.addItem(display_text, userData=python_code)
            
        self.combo_vars.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        btn_insert = QPushButton("Insert Variable/Operator")
        btn_insert.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_insert.clicked.connect(self.insert_variable)
        
        toolbar.addWidget(QLabel("Available Elements:"))
        toolbar.addWidget(self.combo_vars)
        toolbar.addWidget(btn_insert)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        form = QFormLayout()
        self.expr_ax = QLineEdit("0.0")
        self.expr_ay = QLineEdit("0.0")
        self.expr_az = QLineEdit("0.0")
        
        font_style = "font-family: Consolas, 'Courier New', monospace; background-color: #f0f0f0;"
        for widget in [self.expr_ax, self.expr_ay, self.expr_az]:
            widget.setStyleSheet(font_style)

        form.addRow("Target Ax =", self.expr_ax)
        form.addRow("Target Ay =", self.expr_ay)
        form.addRow("Target Az =", self.expr_az)
        layout.addLayout(form)
        
        btn_generate = QPushButton("Generate & Load Custom Law")
        btn_generate.setStyleSheet("font-weight: bold; padding: 5px;")
        btn_generate.clicked.connect(self.generate_and_load_controller)
        layout.addWidget(btn_generate)
        
        self.group_generator.setLayout(layout)
        self.sim_layout.addWidget(self.group_generator)
        
    def setup_swarm_panel(self):
        group_box = QGroupBox("3. Swarm Configuration")
        h_layout = QHBoxLayout()
        
        form_layout = QFormLayout()
        
        ctrl_layout = QHBoxLayout()
        self.combo_controller = QComboBox()
        self.combo_controller.addItem("Built-In: TrajectoryController", userData=TrajectoryController)
        self.combo_controller.addItem("Built-In: AccelerationController", userData=AccelerationController)
        
        btn_load_py = QPushButton("Load External .py")
        btn_load_py.clicked.connect(self.open_file_dialog)
        
        ctrl_layout.addWidget(self.combo_controller)
        ctrl_layout.addWidget(btn_load_py)
        
        self.input_x = QLineEdit("0.0")
        self.input_y = QLineEdit("0.0")
        self.input_z = QLineEdit("0.0")
        
        btn_add_drone = QPushButton("Add Drone to Swarm")
        btn_add_drone.clicked.connect(self.add_drone)
        
        form_layout.addRow("Control Law:", ctrl_layout)
        form_layout.addRow("Start X (m):", self.input_x)
        form_layout.addRow("Start Y (m):", self.input_y)
        form_layout.addRow("Start Z (Down is +):", self.input_z)
        form_layout.addRow("", btn_add_drone)
        
        self.list_drones = QListWidget()
        
        h_layout.addLayout(form_layout)
        h_layout.addWidget(self.list_drones)
        group_box.setLayout(h_layout)
        self.sim_layout.addWidget(group_box)

    def setup_execution_panel(self):
        group_box = QGroupBox("4. Execution")
        panel = QVBoxLayout()
        
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self.start_simulation)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        panel.addWidget(self.btn_run)
        panel.addWidget(self.progress_bar)
        group_box.setLayout(panel)
        self.sim_layout.addWidget(group_box)

    def setup_analysis_panel(self):
        # --- Toolbar 1: Setup ---
        toolbar1 = QHBoxLayout()
        btn_load_log = QPushButton("Load gui_flight_log.csv")
        btn_load_log.clicked.connect(self.load_log_file)
        
        self.combo_plot_type = QComboBox()
        self.combo_plot_type.addItems([
            "3D Trajectory", "Camera View", "X Position", "Y Position", "Z Position", 
            "X Velocity", "Y Velocity", "Z Velocity",
            "Roll", "Pitch", "Yaw",
            "Roll Rate (P)", "Pitch Rate (Q)", "Yaw Rate (R)"
        ])
        
        btn_add_plot = QPushButton("Add Plot to Grid")
        btn_add_plot.clicked.connect(self.add_plot)
        
        btn_clear = QPushButton("Clear Grid")
        btn_clear.clicked.connect(self.clear_plots)
        
        toolbar1.addWidget(btn_load_log)
        toolbar1.addWidget(QLabel("Plot Quantity:"))
        toolbar1.addWidget(self.combo_plot_type)
        toolbar1.addWidget(btn_add_plot)
        toolbar1.addWidget(btn_clear)
        toolbar1.addStretch()
        
        # --- Toolbar 2: Playback ---
        toolbar2 = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setFixedWidth(80)
        
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        
        toolbar2.addWidget(self.btn_play)
        toolbar2.addWidget(self.time_slider)
        
        # --- Scrollable Grid Layout ---
        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.plot_grid = QGridLayout(self.plot_container)
        self.plot_scroll.setWidget(self.plot_container)
        
        self.analysis_layout.addLayout(toolbar1)
        self.analysis_layout.addLayout(toolbar2)
        self.analysis_layout.addWidget(self.plot_scroll)
        
        # --- Animation State ---
        self.active_plots = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_tick)
        self.current_frame = 0
        self.frame_step = 25 

    def update_panel_visibility(self, controller_name):
        if "Trajectory" in controller_name:
            self.group_trajectory.setVisible(True)
            self.group_generator.setVisible(False)
        elif "Acceleration" in controller_name:
            self.group_trajectory.setVisible(False)
            self.group_generator.setVisible(True)
        else:
            self.group_trajectory.setVisible(False)
            self.group_generator.setVisible(False)

    def insert_variable(self):
        focused_widget = QApplication.focusWidget()
        valid_widgets = [self.expr_ax, self.expr_ay, self.expr_az]
        
        if isinstance(focused_widget, QLineEdit) and focused_widget in valid_widgets:
            focused_widget.insert(self.combo_vars.currentData())
        else:
            QMessageBox.information(self, "Info", "Click inside one of the Target input boxes first.")

    def open_trajectory_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Trajectory Setpoints", "", "CSV Files (*.csv)")
        if filepath:
            self.trajectory_file = filepath
            self.lbl_trajectory.setText(f"Loaded: {os.path.basename(filepath)}")
    
    def open_file_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Custom Control Law", "", "Python Files (*.py)")
        if filepath:
            try:
                custom_class = load_external_controller(filepath)
                filename = os.path.basename(filepath)
                self.combo_controller.addItem(f"External: {filename}", userData=custom_class)
                self.combo_controller.setCurrentIndex(self.combo_controller.count() - 1)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def generate_and_load_controller(self):
        expressions = {
            'ax': self.expr_ax.text(),
            'ay': self.expr_ay.text(),
            'az': self.expr_az.text()
        }
        
        file_template = f'''import numpy as np
from src.controllers.base_controller import BaseController
from src.controllers.PositionController import PositionController
from src.controllers.AttitudeController import AttitudeController
from src.controllers.RateController import RateController

class GeneratedAccelerationController(BaseController):
    def __init__(self, pos_gains, att_gains, rate_gains, mass, gravity):
        self.mass = mass
        self.gravity = gravity
        self.pos_controller = PositionController(pos_gains['Kp'], pos_gains['Kd'], mass, gravity, 0.85, 0.0)
        self.att_controller = AttitudeController(att_gains['Kp'])
        self.rate_controller = RateController(rate_gains['Kp'], rate_gains['Ki'], rate_gains['Kd'])
        
        self.t = 0.0
        self.id = 0
        self.swarm = []

    def update_context(self, t, agent_id, swarm):
        self.t = t
        self.id = agent_id
        self.swarm = swarm

    def compute_control(self, state, setpoint, dt):
        t = self.t
        id = self.id
        drones = {{}}
        for a in self.swarm:
            s = a.get_state()
            drones[a.id] = {{'x': s[0], 'y': s[1], 'z': s[2], 'vx': s[3], 'vy': s[4], 'vz': s[5]}}
            
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]

        try:
            target_ax = float({expressions['ax']})
            target_ay = float({expressions['ay']})
            target_az = float({expressions['az']})
        except Exception as e:
            target_ax, target_ay, target_az = 0.0, 0.0, 0.0
            
        accel_cmd = np.array([target_ax, target_ay, target_az])

        self.pos_controller._dt = dt
        self.pos_controller._compute_thrust_cmd(accel_cmd)
        q_des, q_des_dot = self.pos_controller.compute_desired_quats(0.0)

        w_des = self.att_controller.compute_desired_rates(state[6:10], q_des, q_des_dot)
        torque_cmd = self.rate_controller.compute_torque_commands(state[10:13], w_des, dt)

        thrust_mag = np.linalg.norm(self.pos_controller._thrust_vector)
        return np.array([thrust_mag, torque_cmd[0], torque_cmd[1], torque_cmd[2]])
'''
        current_dir = os.path.dirname(__file__)
        filepath = os.path.abspath(os.path.join(current_dir, '..', 'controllers', 'generated_accel_law.py'))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                f.write(file_template)
                
            custom_class = load_external_controller(filepath)
            self.combo_controller.addItem("Generated Acceleration Law", userData=custom_class)
            self.combo_controller.setCurrentIndex(self.combo_controller.count() - 1)
            
            QMessageBox.information(self, "Success", "Acceleration Law Generated and added to the Swarm Configuration dropdown!")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", str(e))
                
    def add_drone(self):
        try:
            x = float(self.input_x.text())
            y = float(self.input_y.text())
            z = float(self.input_z.text())
            ctrl_class = self.combo_controller.currentData()
            
            drone_info = {
                "id": self.drone_counter,
                "start_pos": [x, y, z],
                "controller_class": ctrl_class
            }
            self.swarm_config_list.append(drone_info)
            
            ctrl_name = self.combo_controller.currentText()
            
            if "Acceleration" in ctrl_name and "Generated" in ctrl_name:
                ax_str = self.expr_ax.text()
                ay_str = self.expr_ay.text()
                az_str = self.expr_az.text()
                ctrl_name = f"Accel Law (Ax: {ax_str}, Ay: {ay_str}, Az: {az_str})"

            self.list_drones.addItem(f"Drone {self.drone_counter} | Pos: ({x}, {y}, {z}) | Ctrl: {ctrl_name}")
            
            state_keys = [('X Position', 'x'), ('Y Position', 'y'), ('Z Position', 'z'), 
                          ('X Velocity', 'vx'), ('Y Velocity', 'vy'), ('Z Velocity', 'vz')]
            
            if self.drone_counter == 1:
                self.combo_vars.insertSeparator(self.combo_vars.count())
                
            for label, state in state_keys:
                display = f"Drone {self.drone_counter} {label}"
                code = f"drones[{self.drone_counter}]['{state}']"
                self.combo_vars.addItem(display, userData=code)
            
            self.drone_counter += 1
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Coordinates must be valid numbers.")

    def start_simulation(self):
        if not self.swarm_config_list:
            QMessageBox.warning(self, "Warning", "Please add at least one drone to the swarm.")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Computing...")
        self.progress_bar.setValue(0)
        
        self.worker = SimulationWorker(
            t_final=60.0, 
            dt=0.002, 
            swarm_configs=self.swarm_config_list,
            trajectory_file=self.trajectory_file
        )
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.simulation_finished.connect(self.on_simulation_complete)
        self.worker.error_occurred.connect(self.on_simulation_error)
        self.worker.start()

    def on_simulation_complete(self, message):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        QMessageBox.information(self, "Success", message)
        
        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'gui_flight_log.csv'))
        if os.path.exists(log_path):
            self.load_log_file(log_path)

    def on_simulation_error(self, error_message):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Physics Engine Error", error_message)

    # =========================================================================
    # POST-FLIGHT ANALYSIS & ANIMATION LOGIC
    # =========================================================================

    def load_log_file(self, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self, "Select Flight Log CSV", "", "CSV Files (*.csv)")
        
        if filepath:
            try:
                self.log_data = pd.read_csv(filepath)
                
                # Pre-calculate Euler Angles
                drones = [col.split('_')[0] for col in self.log_data.columns if 'PosX' in col]
                for d in drones:
                    if f'{d}_Qw' in self.log_data.columns:
                        qw = self.log_data[f'{d}_Qw']
                        qx = self.log_data[f'{d}_Qx']
                        qy = self.log_data[f'{d}_Qy']
                        qz = self.log_data[f'{d}_Qz']
                        
                        self.log_data[f'{d}_Roll'] = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
                        
                        sinp = 2*(qw*qy - qz*qx)
                        sinp = np.clip(sinp, -1.0, 1.0) 
                        self.log_data[f'{d}_Pitch'] = np.arcsin(sinp)
                        
                        self.log_data[f'{d}_Yaw'] = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

                QMessageBox.information(self, "Loaded", f"Successfully loaded {os.path.basename(filepath)}")
                
                self.time_slider.setMaximum(max(0, len(self.log_data) - 1))
                self.clear_plots()
                self.add_plot() # Automatically add the first plot
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")

    def clear_plots(self):
        for pstate in self.active_plots:
            self.plot_grid.removeWidget(pstate['canvas'])
            pstate['canvas'].deleteLater()
            pstate['fig'].clear()
        
        self.active_plots.clear()
        self.timer.stop()
        self.btn_play.setText("Play")
        self.current_frame = 0
        self.time_slider.setValue(0)

    def add_plot(self):
        if self.log_data is None or self.log_data.empty:
            QMessageBox.warning(self, "No Data", "Please load a valid flight log CSV first.")
            return

        qty = self.combo_plot_type.currentText()
        is_3d = (qty == "3D Trajectory")
        is_cam = (qty == "Camera View")

        # Create new isolated canvas
        fig = Figure(figsize=(6, 5), dpi=100)
        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(450, 400) # Prevents grid squash
        
        # Determine Grid Position (2 Columns wide)
        idx = len(self.active_plots)
        cols = 2
        row = idx // cols
        col = idx % cols
        self.plot_grid.addWidget(canvas, row, col)
        
        plot_state = {
            'fig': fig,
            'canvas': canvas,
            'qty': qty,
            'is_3d': is_3d,
            'is_cam': is_cam,
            'lines_global': {},
            'points_global': {},
            'arms_global': {},
            'lines_local': {},
            'points_local': {},
            'arms_local': {},
            'axes_local': {},
            'drone_cols': {}
        }
        self.active_plots.append(plot_state)
        
        self._init_single_plot(plot_state)
        self.update_frame(redraw=True)

    def toggle_playback(self):
        if self.log_data is None: return
        
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            if self.current_frame >= len(self.log_data) - 1:
                self.current_frame = 0
            self.timer.start(50)
            self.btn_play.setText("Pause")

    def on_slider_changed(self, value):
        self.current_frame = value
        self.update_frame(redraw=True)

    def timer_tick(self):
        df = self.log_data
        self.current_frame += self.frame_step
        
        if self.current_frame >= len(df) - 1:
            self.current_frame = len(df) - 1
            self.timer.stop()
            self.btn_play.setText("Play")
        
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.current_frame)
        self.time_slider.blockSignals(False)
        
        self.update_frame(redraw=True)

    def _init_single_plot(self, pstate):
        df = self.log_data
        qty = pstate['qty']
        is_3d = pstate['is_3d']
        is_cam = pstate.get('is_cam', False)
        fig = pstate['fig']

        drones = []
        for col in df.columns:
            if 'PosX' in col:
                d_id = col.replace('PosX', '').strip('_')
                if d_id not in drones:
                    drones.append(d_id)
                
        num_drones = max(1, len(drones))
        gs = gridspec.GridSpec(2, num_drones, figure=fig, height_ratios=[2, 1])

        if is_3d:
            ax_global = fig.add_subplot(gs[0, :], projection='3d')
            ax_global.set_title("Global Swarm Viewport (NED)")
            ax_global.set_xlabel("X (North) [m]")
            ax_global.set_ylabel("Y (East) [m]")
            ax_global.set_zlabel("Z (Down) [m]")
            ax_global.invert_zaxis()
        elif is_cam:
            ax_global = fig.add_subplot(gs[0, :])
            ax_global.set_title("Camera View Active (See Local Focus Below)")
            ax_global.axis('off')
        else:
            ax_global = fig.add_subplot(gs[0, :])
            ax_global.set_title(f"Global Viewport: {qty}")
            ax_global.set_xlabel("Time (s)")
            ax_global.set_ylabel(qty)
            ax_global.grid(True)

        g_xmin, g_xmax = np.inf, -np.inf
        g_ymin, g_ymax = np.inf, -np.inf
        g_zmin, g_zmax = np.inf, -np.inf

        for i, d in enumerate(drones):
            drone_color = f'C{i % 10}'
            clean_name = f"Drone {d.replace('D', '')}"
            
            if is_3d:
                cols = {'x': f'{d}_PosX', 'y': f'{d}_PosY', 'z': f'{d}_PosZ',
                        'qw': f'{d}_Qw', 'qx': f'{d}_Qx', 'qy': f'{d}_Qy', 'qz': f'{d}_Qz'}
                pstate['drone_cols'][d] = cols
                
                x_data, y_data, z_data = df[cols['x']].dropna(), df[cols['y']].dropna(), df[cols['z']].dropna()
                
                if not x_data.empty and not y_data.empty and not z_data.empty:
                    g_xmin, g_xmax = min(g_xmin, x_data.min()), max(g_xmax, x_data.max())
                    g_ymin, g_ymax = min(g_ymin, y_data.min()), max(g_ymax, y_data.max())
                    g_zmin, g_zmax = min(g_zmin, z_data.min()), max(g_zmax, z_data.max())

                clean_name = f"Drone {d.replace('D', '')}"
                
                line_g, = ax_global.plot([], [], [], color=drone_color, label=clean_name, linewidth=1.5, alpha=0.7)
                point_g, = ax_global.plot([], [], [], color=drone_color, marker='o', markersize=4)
                arm1_g, = ax_global.plot([], [], [], color=drone_color, linewidth=2)
                arm2_g, = ax_global.plot([], [], [], color=drone_color, linewidth=2)
                
                pstate['lines_global'][d] = line_g
                pstate['points_global'][d] = point_g
                pstate['arms_global'][d] = [arm1_g, arm2_g]

                ax_loc = fig.add_subplot(gs[1, i], projection='3d')
                ax_loc.set_title(f"{clean_name} Focus")
                ax_loc.set_xlabel("X")
                ax_loc.set_ylabel("Y")
                ax_loc.invert_zaxis()
                
                line_l, = ax_loc.plot([], [], [], color=drone_color, label=clean_name, linewidth=2, alpha=0.5)
                point_l, = ax_loc.plot([], [], [], color=drone_color, marker='o', markersize=4)
                arm1_l, = ax_loc.plot([], [], [], color=drone_color, linewidth=3)
                arm2_l, = ax_loc.plot([], [], [], color=drone_color, linewidth=3)
                
                pstate['lines_local'][d] = line_l
                pstate['points_local'][d] = point_l
                pstate['arms_local'][d] = [arm1_l, arm2_l]
                pstate['axes_local'][d] = ax_loc

            elif is_cam:
                cols = {'x': f'{d}_PosX', 'y': f'{d}_PosY', 'z': f'{d}_PosZ',
                        'qw': f'{d}_Qw', 'qx': f'{d}_Qx', 'qy': f'{d}_Qy', 'qz': f'{d}_Qz'}
                pstate['drone_cols'][d] = cols
                
                ax_loc = fig.add_subplot(gs[1, i])
                ax_loc.set_title(f"{clean_name} Camera Frame")
                ax_loc.set_xlabel("X (Image Right)")
                ax_loc.set_ylabel("Y (Image Down)")
                
                # Assume a normalized 90-degree FOV image plane [-1, 1]
                ax_loc.set_xlim(-1.2, 1.2)
                ax_loc.set_ylim(1.2, -1.2)  # Invert Y to match image coordinates
                ax_loc.set_aspect('equal')
                ax_loc.grid(True, linestyle='--', alpha=0.5)
                
                pstate['axes_local'][d] = ax_loc
                pstate['points_local'][d] = {}
                
                # Initialize a point for every *other* drone in this drone's camera
                for other_d in drones:
                    if other_d != d:
                        color = f'C{drones.index(other_d) % 10}'
                        pt, = ax_loc.plot([], [], marker='o', markersize=8, color=color, 
                                          label=f'Drone {other_d.replace("D", "")}', linestyle='None')
                        pstate['points_local'][d][other_d] = pt
                        
                if len(drones) > 1:
                    ax_loc.legend(loc='upper right', fontsize='x-small')

            else:
                col_map = {
                    'X Position': 'PosX', 'Y Position': 'PosY', 'Z Position': 'PosZ',
                    'X Velocity': 'VelX', 'Y Velocity': 'VelY', 'Z Velocity': 'VelZ',
                    'Roll': 'Roll', 'Pitch': 'Pitch', 'Yaw': 'Yaw',
                    'Roll Rate (P)': 'RateP', 'Pitch Rate (Q)': 'RateQ', 'Yaw Rate (R)': 'RateR'
                }
                pstate['drone_cols'][d] = {'val': f"{d}_{col_map[qty]}"}
                clean_name = f"Drone {d.replace('D', '')}"
                
                line_g, = ax_global.plot([], [], color=drone_color, label=clean_name, linewidth=1.5, alpha=0.7)
                point_g, = ax_global.plot([], [], color=drone_color, marker='o', markersize=6)
                pstate['lines_global'][d] = line_g
                pstate['points_global'][d] = point_g

                ax_loc = fig.add_subplot(gs[1, i])
                ax_loc.set_title(f"{clean_name} Focus")
                ax_loc.grid(True)
                ax_loc.set_xlabel("Time (s)")
                
                line_l, = ax_loc.plot([], [], color=drone_color, label=clean_name, linewidth=2)
                point_l, = ax_loc.plot([], [], color=drone_color, marker='o', markersize=6)
                pstate['lines_local'][d] = line_l
                pstate['points_local'][d] = point_l
                pstate['axes_local'][d] = ax_loc
        
        if drones:
            ax_global.legend(loc='upper right', fontsize='small')

        if is_3d:
            if np.isinf(g_xmin) or np.isnan(g_xmin):
                g_xmin, g_xmax, g_ymin, g_ymax, g_zmin, g_zmax = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            margin_x = max(1.0, (g_xmax - g_xmin) * 0.1)
            margin_y = max(1.0, (g_ymax - g_ymin) * 0.1)
            margin_z = max(1.0, (g_zmax - g_zmin) * 0.1)
            
            ax_global.set_xlim(g_xmin - margin_x, g_xmax + margin_x)
            ax_global.set_ylim(g_ymin - margin_y, g_ymax + margin_y)
            ax_global.set_zlim(g_zmax + margin_z, g_zmin - margin_z) 
        else:
            time_max = df.iloc[:, 0].max() if not df.empty else 1.0
            ax_global.set_xlim(0, time_max)
            all_vals = []
            for d in drones:
                try: 
                    all_vals.extend(df[pstate['drone_cols'][d]['val']].dropna().tolist())
                except KeyError:
                    pass
            if all_vals:
                margin = max(0.5, (max(all_vals) - min(all_vals))*0.1)
                ax_global.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
            else:
                ax_global.set_ylim(-1, 1)

        fig.tight_layout()
        pstate['canvas'].draw()

    def update_frame(self, redraw=True):
        if self.log_data is None or self.log_data.empty: return
        df = self.log_data
        idx = self.current_frame
        
        for pstate in self.active_plots:
            for d, cols in pstate['drone_cols'].items():
                if pstate['is_3d']:
                    x = df[cols['x']].values[:idx+1]
                    y = df[cols['y']].values[:idx+1]
                    z = df[cols['z']].values[:idx+1]
                    
                    pstate['lines_global'][d].set_data(x, y)
                    pstate['lines_global'][d].set_3d_properties(z)
                    pstate['points_global'][d].set_data(x[-1:], y[-1:])
                    pstate['points_global'][d].set_3d_properties(z[-1:])
                    
                    pstate['lines_local'][d].set_data(x, y)
                    pstate['lines_local'][d].set_3d_properties(z)
                    pstate['points_local'][d].set_data(x[-1:], y[-1:])
                    pstate['points_local'][d].set_3d_properties(z[-1:])
                    
                    try:
                        qw = df[cols['qw']].values[idx]
                        qx = df[cols['qx']].values[idx]
                        qy = df[cols['qy']].values[idx]
                        qz = df[cols['qz']].values[idx]
                        
                        R = np.array([
                            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
                            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                        ])
                        
                        L = 0.25 
                        arm1_body = np.array([[L, -L], [L, -L], [0, 0]])
                        arm2_body = np.array([[L, -L], [-L, L], [0, 0]])
                        
                        arm1_global = R @ arm1_body
                        arm2_global = R @ arm2_body
                        
                        cx, cy, cz = x[-1], y[-1], z[-1]
                        
                        a1_x = cx + arm1_global[0]
                        a1_y = cy + arm1_global[1]
                        a1_z = cz + arm1_global[2]
                        
                        a2_x = cx + arm2_global[0]
                        a2_y = cy + arm2_global[1]
                        a2_z = cz + arm2_global[2]
                        
                        pstate['arms_global'][d][0].set_data(a1_x, a1_y)
                        pstate['arms_global'][d][0].set_3d_properties(a1_z)
                        pstate['arms_global'][d][1].set_data(a2_x, a2_y)
                        pstate['arms_global'][d][1].set_3d_properties(a2_z)
                        
                        pstate['arms_local'][d][0].set_data(a1_x, a1_y)
                        pstate['arms_local'][d][0].set_3d_properties(a1_z)
                        pstate['arms_local'][d][1].set_data(a2_x, a2_y)
                        pstate['arms_local'][d][1].set_3d_properties(a2_z)
                    except KeyError:
                        pass 
                    
                    win = 2.0
                    pstate['axes_local'][d].set_xlim(x[-1] - win, x[-1] + win)
                    pstate['axes_local'][d].set_ylim(y[-1] - win, y[-1] + win)
                    pstate['axes_local'][d].set_zlim(z[-1] + win, z[-1] - win) 
                elif pstate.get('is_cam', False):
                    # Constant Gimbal-to-Camera Matrix
                    R_g_c = np.array([
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0]
                    ])
                    
                    for d in pstate['drone_cols']:
                        try:
                            cols = pstate['drone_cols'][d]
                            ox = df[cols['x']].values[idx]
                            oy = df[cols['y']].values[idx]
                            oz = df[cols['z']].values[idx]
                            qw = df[cols['qw']].values[idx]
                            qx = df[cols['qx']].values[idx]
                            qy = df[cols['qy']].values[idx]
                            qz = df[cols['qz']].values[idx]
                            
                            # Body to Inertial
                            R_b_i = np.array([
                                [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                                [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
                                [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                            ])
                            
                            # Inertial to Body (Transpose)
                            R_i_b = R_b_i.T
                            
                            for other_d, pt in pstate['points_local'][d].items():
                                ocols = pstate['drone_cols'][other_d]
                                tx = df[ocols['x']].values[idx]
                                ty = df[ocols['y']].values[idx]
                                tz = df[ocols['z']].values[idx]
                                
                                # Relative vector in inertial frame
                                v_i = np.array([tx - ox, ty - oy, tz - oz])
                                
                                # Map to camera frame
                                v_c = R_g_c @ (R_i_b @ v_i) 
                                Xc, Yc, Zc = v_c[0], v_c[1], v_c[2]
                                
                                # If target is in front of the camera
                                if Zc > 0:
                                    f = 1.0  # Normalized focal length
                                    x_img = f * Xc / Zc
                                    y_img = f * Yc / Zc
                                    pt.set_data([x_img], [y_img])
                                else:
                                    # Target is behind the camera; hide it
                                    pt.set_data([np.nan], [np.nan])
                                    
                        except KeyError:
                            pass
                else:
                    try:
                        t = df.iloc[:, 0].values[:idx+1]
                        v = df[cols['val']].values[:idx+1]
                        
                        pstate['lines_global'][d].set_data(t, v)
                        pstate['points_global'][d].set_data(t[-1:], v[-1:])
                        
                        pstate['lines_local'][d].set_data(t, v)
                        pstate['points_local'][d].set_data(t[-1:], v[-1:])
                        
                        t_win = 5.0
                        v_margin = max(0.5, abs(v[-1]) * 0.5) 
                        pstate['axes_local'][d].set_xlim(max(0, t[-1] - t_win), max(t_win, t[-1] + t_win*0.1))
                        pstate['axes_local'][d].set_ylim(v[-1] - v_margin, v[-1] + v_margin)
                    except KeyError:
                        pass
                        
            if redraw:
                pstate['canvas'].draw_idle()