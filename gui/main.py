"""KARA desktop dashboard (PySide6 / Qt Widgets).

Program mappings, robot controls and safety interlocks are kept in the existing
launcher backend. This module only orchestrates them with a desktop interface.
"""
from __future__ import annotations

import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PySide6.QtGui import QColor, QCloseEvent, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

# Supports "python launch_kara_gui.py" or "python -m gui.main".
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from programs.kara_launcher_core import (  # noqa: E402
    PROGRAMS, PROGRAM_BY_KEY, PROJECT_ROOT, ROBOT_HOST, ROBOT_PORT,
    SERVER_COMMANDS, action_command, find_server_command, locate_python,
    program_command,
)
from gui.theme import (  # noqa: E402
    ACCENT as COLOR_ACCENT, DANGER as COLOR_RED, SUCCESS as COLOR_SUCCESS,
    TEXT as COLOR_TEXT, WARNING as COLOR_AMBER, STYLESHEET,
)
from gui.camera_panel import CameraPanel  # noqa: E402


def _label(text: str, *, obj: str | None = None, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if obj:
        widget.setObjectName(obj)
    widget.setWordWrap(wrap)
    return widget


def _row(parent: QWidget | None = None, spacing: int = 10) -> QHBoxLayout:
    layout = QHBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    return layout


def _column(parent: QWidget | None = None, spacing: int = 10) -> QVBoxLayout:
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    return layout


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName('separator')
    line.setFixedHeight(1)
    return line


class KaraProgramsGUI(QMainWindow):
    """Local control dashboard for existing allowlisted KARA programs."""

    GROUPS = (
        ('Vision', 'Vision & camera', 'Detect objects, calibrate and pick blocks'),
        ('Interactive', 'Manual control', 'Command and jog your robot'),
        ('Demos', 'Motion library', 'Run repeatable motion demonstrations'),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle('KARA  |  Robot control center')
        self.resize(1550, 950)
        self.setMinimumSize(1140, 760)

        self.server_process: QProcess | None = None
        self.server_name: str | None = None
        self.program_process: QProcess | None = None
        self.current_program_key: str | None = None
        self.stop_pending = False
        self.closing = False
        self.last_backend_status: bool | None = None
        self.last_backend_check = 0.0
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.program_buttons: dict[str, QPushButton] = {}
        self.program_cards: dict[str, QFrame] = {}
        self.group_grids: dict[str, QGridLayout] = {}
        self.group_cards: dict[str, list[tuple[object, QFrame]]] = {}
        self.action_buttons: dict[str, QPushButton] = {}
        self.nav_buttons: list[QPushButton] = []
        self.resume_camera_after_program = False

        self._build_ui()
        self.log('KARA dashboard ready. Select a server and choose a program.')
        self.log('Motion is locked by default. Test camera detection with DRY RUN first.')
        self.log('Use Start camera for the workspace preview; camera scripts temporarily own that device.')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(160)
        self._select_group(0)
        self._update_button_states()

    def _build_ui(self):
        app_root = QWidget()
        app_root.setObjectName('centralContent')
        self.setCentralWidget(app_root)
        outer = QHBoxLayout(app_root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_sidebar())

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(25, 20, 25, 14)
        main_layout.setSpacing(15)
        outer.addWidget(main, 1)

        top = _row(spacing=12)
        intro = QWidget()
        intro_col = _column(intro, 3)
        intro_col.addWidget(_label('WORKSPACE  /  PAROL6', obj='eyebrow'))
        intro_col.addWidget(_label('Robot control center', obj='headline'))
        intro_col.addWidget(_label('A single place to connect, operate and monitor your KARA arm.', obj='subtitle'))
        top.addWidget(intro, 1)
        status_col = QWidget()
        status_layout = _column(status_col, 7)
        self.server_state_label = _label('●  127.0.0.1:5001  ·  OFFLINE', obj='statusPill')
        self.server_state_label.setAlignment(Qt.AlignCenter)
        self.motion_state_label = _label('MOTION LOCKED', obj='motionPill')
        self.motion_state_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.server_state_label)
        status_layout.addWidget(self.motion_state_label)
        top.addWidget(status_col)
        main_layout.addLayout(top)

        main_layout.addWidget(self._build_connection_panel())
        main_layout.addWidget(self._build_safety_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_workbench())
        monitor = QSplitter(Qt.Vertical)
        monitor.setChildrenCollapsible(False)
        self.camera_panel = CameraPanel(index_provider=lambda: self.camera_spin.value())
        self.camera_panel.message.connect(self._on_camera_message)
        monitor.addWidget(self.camera_panel)
        monitor.addWidget(self._build_output())
        monitor.setSizes([380, 280])
        monitor.setStretchFactor(0, 3)
        monitor.setStretchFactor(1, 2)
        splitter.addWidget(monitor)
        splitter.setSizes([750, 580])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter, 1)

        foot = _row()
        foot.addWidget(_label('KARA DESKTOP  •  PySide6 / Qt', obj='eyebrow'))
        foot.addStretch(1)
        caution = _label('Software stop is NOT a hardware emergency stop.', obj='hint')
        foot.addWidget(caution)
        main_layout.addLayout(foot)

    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName('sidebar')
        side.setFixedWidth(225)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(16, 22, 16, 19)
        lay.setSpacing(14)

        brand = _row(spacing=12)
        mark = _label('K', obj='brandMark')
        mark.setFixedSize(48, 48)
        mark.setAlignment(Qt.AlignCenter)
        brand.addWidget(mark)
        name = QWidget()
        names = _column(name, 1)
        names.addWidget(_label('KARA', obj='brandTitle'))
        names.addWidget(_label('ROBOTICS LAB', obj='brandSubtitle'))
        brand.addWidget(name, 1)
        lay.addLayout(brand)
        lay.addSpacing(10)
        lay.addWidget(_divider())
        lay.addSpacing(10)
        lay.addWidget(_label('PROGRAMS', obj='eyebrow'))

        for index, text in enumerate(('◉   Vision & camera', '▤   Manual control', '▦   Motion library')):
            button = QPushButton(text)
            button.setObjectName('navButton')
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, i=index: self._select_group(i))
            lay.addWidget(button)
            self.nav_buttons.append(button)

        lay.addSpacing(10)
        lay.addWidget(_divider())
        lay.addSpacing(8)
        lay.addWidget(_label('SYSTEM', obj='eyebrow'))
        tip = _label('Choose a program to see the command output in real time.', obj='mutedText', wrap=True)
        lay.addWidget(tip)
        lay.addStretch(1)

        safety_note = QFrame()
        safety_note.setObjectName('sideInfo')
        safety_lay = QVBoxLayout(safety_note)
        safety_lay.setContentsMargins(12, 13, 12, 14)
        safety_lay.setSpacing(7)
        safety_lay.addWidget(_label('●   SAFETY REMINDER', obj='hint'))
        safety_lay.addWidget(_label('Keep the hardware E-STOP within reach whenever the arm is powered.',
                                     obj='mutedText', wrap=True))
        lay.addWidget(safety_note)
        return side

    def _build_connection_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName('panel')
        col = QVBoxLayout(panel)
        col.setContentsMargins(18, 15, 18, 16)
        col.setSpacing(11)
        heading = _row()
        heading.addWidget(_label('Robot connection', obj='panelTitle'))
        heading.addStretch(1)
        heading.addWidget(_label('SERVER  •  CAMERA  •  ENVIRONMENT', obj='eyebrow'))
        col.addLayout(heading)

        primary = QHBoxLayout()
        primary.setSpacing(10)
        primary.addWidget(_label('Server', obj='smallLabel'))
        self.server_combo = QComboBox()
        self.server_combo.addItems(list(SERVER_COMMANDS))
        self.server_combo.setMinimumWidth(195)
        primary.addWidget(self.server_combo, 3)
        self.start_server_btn = QPushButton('Start server')
        self.start_server_btn.setObjectName('primaryButton')
        self.start_server_btn.clicked.connect(self.start_server)
        primary.addWidget(self.start_server_btn)
        self.stop_server_btn = QPushButton('Stop server')
        self.stop_server_btn.setObjectName('ghostButton')
        self.stop_server_btn.clicked.connect(self.stop_server)
        primary.addWidget(self.stop_server_btn)
        primary.addSpacing(10)
        primary.addWidget(_label('COM port', obj='smallLabel'))
        self.com_port = QLineEdit()
        self.com_port.setPlaceholderText('e.g. COM5')
        self.com_port.setMaximumWidth(112)
        primary.addWidget(self.com_port)
        primary.addWidget(_label('Camera', obj='smallLabel'))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 20)
        self.camera_spin.setValue(0)
        self.camera_spin.setFixedWidth(62)
        self.camera_spin.setToolTip('Local USB camera index used by the embedded preview and OpenCV scripts')
        primary.addWidget(self.camera_spin)
        col.addLayout(primary)

        secondary = _row(spacing=10)
        secondary.addWidget(_label('Python', obj='smallLabel'))
        self.python_edit = QLineEdit(locate_python())
        self.python_edit.setToolTip('The Python interpreter used to run robot and OpenCV scripts')
        secondary.addWidget(self.python_edit, 1)
        browse_btn = QPushButton('Browse…')
        browse_btn.setObjectName('ghostButton')
        browse_btn.clicked.connect(self.browse_python)
        secondary.addWidget(browse_btn)
        col.addLayout(secondary)
        return panel

    def _build_safety_bar(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName('safetyPanel')
        bar = _row(panel, 12)
        bar.setContentsMargins(16, 11, 13, 11)
        self.allow_motion = QCheckBox('Enable real robot motion')
        self.allow_motion.toggled.connect(self._update_button_states)
        bar.addWidget(self.allow_motion)
        bar.addWidget(_label('Locked by default · enabling will require confirmation for every move',
                             obj='mutedText', wrap=True), 1)
        self.stop_robot_btn = QPushButton('STOP ROBOT  (software)')
        self.stop_robot_btn.setObjectName('dangerButton')
        self.stop_robot_btn.clicked.connect(self.stop_robot)
        bar.addWidget(self.stop_robot_btn)
        return panel

    def _build_workbench(self) -> QWidget:
        left = QWidget()
        left.setMinimumWidth(465)
        col = _column(left, 11)
        heading = _row()
        heading.addWidget(_label('Quick actions', obj='sectionTitle'))
        heading.addStretch(1)
        heading.addWidget(_label('PAROL6', obj='eyebrow'))
        col.addLayout(heading)
        actions = QGridLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setHorizontalSpacing(8)
        actions.setVerticalSpacing(8)
        action_list = (
            ('home', '⌂  HOME'), ('park', '↙  PARK'), ('gripper_open', 'OPEN GRIPPER'),
            ('gripper_close', 'CLOSE GRIPPER'), ('pose', '◎  READ POSE'),
        )
        for index, (key, title) in enumerate(action_list):
            button = QPushButton(title)
            button.setObjectName('quickAction')
            button.setMinimumHeight(44)
            button.clicked.connect(lambda checked=False, k=key: self.start_action(k))
            actions.addWidget(button, index // 3, index % 3)
            self.action_buttons[key] = button
        for c in range(3):
            actions.setColumnStretch(c, 1)
        col.addLayout(actions)

        library = QFrame()
        library.setObjectName('programArea')
        library_lay = QVBoxLayout(library)
        library_lay.setContentsMargins(15, 14, 15, 11)
        library_lay.setSpacing(10)
        title_row = _row()
        self.program_section_label = _label('Vision & camera', obj='sectionTitle')
        title_row.addWidget(self.program_section_label)
        title_row.addStretch(1)
        self.program_count_label = _label('6 PROGRAMS', obj='programCount')
        title_row.addWidget(self.program_count_label)
        library_lay.addLayout(title_row)
        self.group_description = _label('Detect objects, calibrate and pick blocks', obj='mutedText')
        library_lay.addWidget(self.group_description)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText('Search programs…')
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter_cards)
        library_lay.addWidget(self.search_edit)
        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()  # Navigation is in the sidebar.
        for group, label, _description in self.GROUPS:
            self.tabs.addTab(self._build_program_tab(group), label)
        library_lay.addWidget(self.tabs, 1)
        col.addWidget(library, 1)
        return left

    def _build_program_tab(self, group: str) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 3, 6, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.group_grids[group] = grid
        group_cards = []
        for index, spec in enumerate(p for p in PROGRAMS if p.group == group):
            card = QFrame()
            card.setObjectName('programCard')
            card.setMinimumHeight(138)
            cell = QVBoxLayout(card)
            cell.setContentsMargins(13, 12, 13, 12)
            cell.setSpacing(7)
            row = _row()
            title = _label(spec.title, obj='cardTitle', wrap=True)
            row.addWidget(title, 1)
            if not spec.moves_robot:
                category = 'NO MOTION'
                color = '#79DFB3'
            elif spec.interactive:
                category = 'INTERACTIVE'
                color = '#D5B0FF'
            else:
                category = 'MOTION'
                color = '#EFC585'
            badge = _label(category, obj='smallLabel')
            badge.setStyleSheet(f'color: {color}; font-size: 8pt; font-weight: 750;')
            row.addWidget(badge)
            cell.addLayout(row)
            description = _label(spec.description, obj='mutedText', wrap=True)
            description.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            cell.addWidget(description, 1)
            footer = _row()
            if spec.camera:
                footer.addWidget(_label('●  CAMERA', obj='smallLabel'))
            elif spec.requires_robot:
                footer.addWidget(_label('●  ROBOT', obj='smallLabel'))
            else:
                footer.addWidget(_label('●  UTILITY', obj='smallLabel'))
            footer.addStretch(1)
            button = QPushButton('Run  →')
            button.setObjectName('runButton')
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, k=spec.key: self.start_program(k))
            footer.addWidget(button)
            cell.addLayout(footer)
            r, c = divmod(index, 2)
            grid.addWidget(card, r, c)
            self.program_buttons[spec.key] = button
            self.program_cards[spec.key] = card
            group_cards.append((spec, card))
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch((len(group_cards) + 1) // 2, 1)
        self.group_cards[group] = group_cards
        scroll.setWidget(body)
        return scroll

    def _select_group(self, index: int):
        self.tabs.setCurrentIndex(index)
        group, name, description = self.GROUPS[index]
        self.program_section_label.setText(name)
        self.group_description.setText(description)
        for i, button in enumerate(self.nav_buttons):
            button.setProperty('active', i == index)
            button.style().unpolish(button)
            button.style().polish(button)
        self._filter_cards(self.search_edit.text())

    def _filter_cards(self, search: str):
        group = self.GROUPS[self.tabs.currentIndex()][0]
        items = self.group_cards[group]
        grid = self.group_grids[group]
        visible = [(spec, card) for spec, card in items
                   if search.casefold().strip() in (spec.title + ' ' + spec.description).casefold()]
        for _, card in items:
            grid.removeWidget(card)
            card.hide()
        for i, (_, card) in enumerate(visible):
            grid.addWidget(card, i // 2, i % 2)
            card.show()
        self.program_count_label.setText(f'{len(visible)} PROGRAMS')

    def _build_output(self) -> QWidget:
        right = QFrame()
        right.setObjectName('consolePanel')
        right.setMinimumWidth(365)
        col = QVBoxLayout(right)
        col.setContentsMargins(15, 15, 15, 14)
        col.setSpacing(12)
        heading = _row()
        title_block = QWidget()
        title_col = _column(title_block, 3)
        title_col.addWidget(_label('Live activity', obj='sectionTitle'))
        title_col.addWidget(_label('Real-time server and program messages', obj='mutedText'))
        heading.addWidget(title_block, 1)
        clear = QPushButton('Clear')
        clear.setObjectName('ghostButton')
        clear.clicked.connect(lambda: self.console.clear())
        heading.addWidget(clear)
        col.addLayout(heading)
        self.script_state_label = _label('●  No running program', obj='runningPill')
        col.addWidget(self.script_state_label)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setObjectName('programOutput')
        self.console.setMaximumBlockCount(10000)
        self.console.setPlaceholderText('KARA output will appear here once a program is started…')
        col.addWidget(self.console, 1)

        col.addWidget(_label('INTERACTIVE COMMAND', obj='smallLabel'))
        command_row = _row(spacing=7)
        self.command_entry = QLineEdit()
        self.command_entry.setPlaceholderText('e.g. home  or  200,150')
        self.command_entry.returnPressed.connect(self.send_command)
        command_row.addWidget(self.command_entry, 1)
        self.send_btn = QPushButton('Send  ↗')
        self.send_btn.setObjectName('softButton')
        self.send_btn.clicked.connect(self.send_command)
        command_row.addWidget(self.send_btn)
        col.addLayout(command_row)
        self.stop_program_btn = QPushButton('■  Stop current program')
        self.stop_program_btn.setObjectName('ghostButton')
        self.stop_program_btn.clicked.connect(self.stop_program)
        col.addWidget(self.stop_program_btn)
        col.addWidget(_divider())
        status = _row()
        status.addWidget(_label('STATUS', obj='smallLabel'))
        status.addStretch(1)
        self.console_status = _label('Idle / waiting for commands', obj='statusLabel')
        status.addWidget(self.console_status)
        col.addLayout(status)
        return right

    def _on_camera_message(self, message: str):
        self.log(message, 'system')

    def _resume_preview_if_needed(self):
        if self.resume_camera_after_program:
            self.resume_camera_after_program = False
            # Allow OpenCV subprocesses time to release the USB device.
            QTimer.singleShot(450, self.camera_panel.start_preview)

    def browse_python(self):
        current = self.python_edit.text().strip().strip('"')
        directory = str(Path(current).parent) if current else str(PROJECT_ROOT)
        chosen, _ = QFileDialog.getOpenFileName(self, 'Select Python executable', directory)
        if chosen:
            self.python_edit.setText(chosen)

    def _python(self) -> str:
        exe = self.python_edit.text().strip().strip('"')
        if not exe:
            raise ValueError('Select a Python interpreter first.')
        if not Path(exe).is_file() and not shutil.which(exe):
            raise FileNotFoundError(f'Python interpreter not found: {exe}')
        return exe

    @staticmethod
    def _backend_online() -> bool:
        try:
            with socket.create_connection((ROBOT_HOST, ROBOT_PORT), timeout=0.20):
                return True
        except OSError:
            return False

    @staticmethod
    def _busy(process: QProcess | None) -> bool:
        return process is not None and process.state() != QProcess.NotRunning

    def log(self, message: str, tag: str = 'system'):
        self._append(f'[{datetime.now().strftime("%H:%M:%S")}] {message}\n', tag)

    def _append(self, text: str, tag: str = ''):
        if not text:
            return
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        char_format = QTextCharFormat()
        char_format.setForeground(QColor({
            'error': COLOR_RED,
            'server': COLOR_AMBER,
            'system': COLOR_ACCENT,
        }.get(tag, COLOR_TEXT)))
        cursor.insertText(text, char_format)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def _read_process(self, process: QProcess, name: str):
        data = bytes(process.readAllStandardOutput())
        if data:
            self._append(f'[{name}] ' + data.decode('utf-8', errors='replace'),
                         'server' if name == 'server' else '')

    def _spawn(self, argv: list[str], *, name: str, env: dict[str, str] | None = None) -> QProcess:
        process = QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        if env is not None:
            process_env = QProcessEnvironment.systemEnvironment()
            for key, value in env.items():
                process_env.insert(key, value)
            process.setProcessEnvironment(process_env)
        # Windows .cmd files must be run through cmd.exe, never through a shell
        # derived from untrusted text. All program paths come from the allowlist.
        if os.name == 'nt' and Path(argv[0]).suffix.lower() in ('.cmd', '.bat'):
            argv = ['cmd.exe', '/c', *argv]
        process.setProgram(argv[0])
        process.setArguments(argv[1:])
        process.readyReadStandardOutput.connect(lambda p=process, n=name: self._read_process(p, n))
        process.errorOccurred.connect(lambda error, p=process, n=name: self._process_error(p, n, error))
        process.finished.connect(lambda code, status, p=process, n=name: self._process_finished(p, n, code))
        process.start()
        self.log(f'Starting {name}: {" ".join(argv)}')
        return process

    def _process_error(self, process: QProcess, name: str, error: QProcess.ProcessError):
        self.log(f'{name} process error: {process.errorString()}', 'error')
        if error == QProcess.FailedToStart:
            if name == 'server' and self.server_process is process:
                self.server_process = None
                self.server_name = None
            elif name == 'program' and self.program_process is process:
                self.program_process = None
                self.current_program_key = None
                self.script_state_label.setText('●  No running program')
                self.camera_panel.set_reserved(False)
                self._resume_preview_if_needed()
            process.deleteLater()
            self._update_button_states()

    def _process_finished(self, process: QProcess, name: str, code: int):
        self._read_process(process, name)
        if name == 'program' and self.program_process is process:
            self.log(f'Program {self.current_program_key} exited (code {code}).',
                     'error' if code else 'system')
            self.program_process = None
            self.current_program_key = None
            self.script_state_label.setText('●  No running program')
            self.camera_panel.set_reserved(False)
            self._resume_preview_if_needed()
        elif name == 'server' and self.server_process is process:
            self.log(f'{self.server_name} exited (code {code}).', 'error' if code else 'system')
            self.server_process = None
            self.server_name = None
        process.deleteLater()
        self._update_button_states()

    def start_server(self):
        if self._busy(self.server_process):
            QMessageBox.information(self, 'Server already running', 'Stop the server owned by this GUI first.')
            return
        name = self.server_combo.currentText()
        if name == 'PAROL6 Robot Server' and self._backend_online():
            QMessageBox.information(self, 'Backend already active',
                'Port 5001 already has a listener. Use the existing backend.\n\n'
                'The GUI will not launch a second PAROL6 server on that port.')
            return
        try:
            executable = find_server_command(name, self._python())
            if not executable:
                raise FileNotFoundError(
                    f'{SERVER_COMMANDS[name]} is not installed in the selected environment.\n'
                    'Set up the PAROL6/Waldo backend first (README_KARA_GUI.md).')
            port = self.com_port.text().strip()
            env = {'PAROL6_COM_PORT': port} if port else None
            self.server_process = self._spawn(executable, name='server', env=env)
            self.server_name = name
        except (ValueError, OSError, KeyError) as exc:
            self._error_dialog('Could not start server', exc)
        self._update_button_states()

    def stop_server(self):
        if self.stop_pending:
            QMessageBox.warning(self, 'Stop in progress', 'Let the robot stop request finish first.')
            return
        if self._busy(self.program_process):
            QMessageBox.warning(self, 'Program running', 'Stop the active program before stopping its server.')
            return
        if not self._busy(self.server_process):
            QMessageBox.information(self, 'External server',
                                    'This GUI does not own a running server. Stop an external server in its own terminal.')
            return
        if QMessageBox.question(self, 'Stop server?', f'Stop {self.server_name}?') == QMessageBox.Yes:
            self._terminate(self.server_process)
            self.log(f'Stopping {self.server_name}.')

    def _validate_job(self, *, needs_robot: bool, moves_robot: bool,
                      description: str, caution: str = '') -> bool:
        if self.stop_pending:
            raise RuntimeError('Software stop in progress. Start another action after it finishes.')
        if self._busy(self.program_process):
            raise RuntimeError('A program is already running. Stop it before launching another one.')
        if needs_robot and not self._backend_online():
            raise RuntimeError('PAROL6 backend not reachable at 127.0.0.1:5001.\n'
                               'Start PAROL6 Robot Server, not only the web UI.')
        if moves_robot:
            if not self.allow_motion.isChecked():
                raise RuntimeError('Robot motion is locked. Tick Enable real robot motion first.')
            warning = f'\n\nCAUTION: {caution}' if caution else ''
            return QMessageBox.question(
                self, 'Confirm real robot movement',
                f'{description}{warning}\n\nClear the work area and keep the hardware E-STOP accessible.\nRun now?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) == QMessageBox.Yes
        if caution:
            return QMessageBox.question(
                self, 'Confirm calibration operation',
                f'{description}\n\nCAUTION: {caution}\n\nContinue?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) == QMessageBox.Yes
        return True

    def start_program(self, key: str):
        spec = PROGRAM_BY_KEY[key]
        try:
            if not self._validate_job(needs_robot=spec.requires_robot,
                moves_robot=spec.moves_robot, description=spec.description, caution=spec.caution):
                return
            argv = program_command(key, self._python(), self.camera_spin.value() if spec.camera else 0)
            if spec.camera and self.camera_panel.has_session():
                # Only one process may own a local USB camera at once.
                if not self.camera_panel.stop_preview():
                    raise RuntimeError('Camera is still releasing. Stop preview before launching this script.')
                self.resume_camera_after_program = True
                self.log('Live preview paused: this program uses the camera directly.')
            if spec.camera:
                self.camera_panel.set_reserved(True)
            self.current_program_key = key
            self.program_process = self._spawn(argv, name='program')
            self.script_state_label.setText(f'●  RUNNING · {spec.title}')
            if spec.interactive:
                self.command_entry.setFocus()
            elif key in ('cartesian_keyboard', 'joint_keyboard'):
                self.log('Global keyboard listener: refer to the program controls, not the command box.')
        except (OSError, RuntimeError, ValueError) as exc:
            self.camera_panel.set_reserved(False)
            self._resume_preview_if_needed()
            self._error_dialog('Could not start program', exc)
        self._update_button_states()

    def start_action(self, action: str):
        names = {
            'home': 'Move robot HOME', 'park': 'Move robot to Commander.PARK_ANGLES',
            'gripper_open': 'Open pneumatic gripper', 'gripper_close': 'Close pneumatic gripper',
            'pose': 'Read current robot pose',
        }
        try:
            if not self._validate_job(needs_robot=True, moves_robot=action != 'pose',
                description=names[action],
                caution='PARK angles come from Commander.py; verify the pose for your robot.'
                if action == 'park' else ''):
                return
            self.current_program_key = f'action:{action}'
            self.program_process = self._spawn(action_command(action, self._python()), name='program')
            self.script_state_label.setText(f'●  RUNNING · {names[action]}')
        except (OSError, RuntimeError, ValueError) as exc:
            self._error_dialog('Command rejected', exc)
        self._update_button_states()

    def send_command(self):
        proc = self.program_process
        spec = PROGRAM_BY_KEY.get(self.current_program_key or '')
        if not self._busy(proc) or spec is None or not spec.interactive:
            QMessageBox.information(self, 'Interactive control',
                'Start Commander, Cartesian XYZ Console, or a pose console first.\n'
                'Keyboard jog programs listen for GLOBAL key presses.')
            return
        command = self.command_entry.text().strip()
        if not command or '\r' in command or '\n' in command or len(command) > 160:
            return
        if proc.write((command + '\n').encode('utf-8')) < 0:
            self.log('Could not send command: ' + proc.errorString(), 'error')
            return
        self.log(f'> {command}')
        self.command_entry.clear()

    @staticmethod
    def _terminate(proc: QProcess | None):
        if proc is None or proc.state() == QProcess.NotRunning:
            return
        proc.terminate()

        def kill_if_needed():
            # The process may have finished and been deleteLater()'d already.
            try:
                if proc.state() != QProcess.NotRunning:
                    proc.kill()
            except RuntimeError:
                pass

        QTimer.singleShot(2400, kill_if_needed)

    def _run_software_stop(self, python: str):
        try:
            kwargs: dict = dict(cwd=str(PROJECT_ROOT), capture_output=True,
                                 text=True, encoding='utf-8', errors='replace', timeout=7)
            if os.name == 'nt':
                kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            result = subprocess.run(action_command('stop', python), **kwargs)
            output = (result.stdout or '') + (result.stderr or '')
            self.events.put(('system' if result.returncode == 0 else 'error',
                             output or f'Backend stop exited with code {result.returncode}.\n'))
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            self.events.put(('error', f'Software stop failed / not confirmed: {exc}\n'))
        finally:
            self.events.put(('_stop_complete', ''))

    def _software_stop_async(self):
        self.stop_pending = True
        self._update_button_states()
        try:
            python = self._python()
        except (ValueError, FileNotFoundError) as exc:
            self.log(str(exc), 'error')
            self.stop_pending = False
            self._update_button_states()
            return
        threading.Thread(target=self._run_software_stop, args=(python,), daemon=True).start()

    def stop_program(self):
        if not self._busy(self.program_process):
            self.log('No program is running.')
            return
        spec = PROGRAM_BY_KEY.get(self.current_program_key or '')
        moves = spec.moves_robot if spec is not None else self.current_program_key != 'action:pose'
        if moves and self.stop_pending:
            return
        self.log('Stopping program...' + (' Sending backend stop request.' if moves else ''))
        self._terminate(self.program_process)
        if moves:
            self._software_stop_async()

    def stop_robot(self):
        # Always reachable, even when motion-enable checkbox is unchecked.
        if self.stop_pending:
            return
        self.log('STOP ROBOT requested; software stop is NOT a hardware E-STOP.', 'error')
        if self._busy(self.program_process):
            self._terminate(self.program_process)
        self._software_stop_async()

    def _update_button_states(self, *_args):
        busy = self._busy(self.program_process)
        unavailable = busy or self.stop_pending
        for spec in PROGRAMS:
            self.program_buttons[spec.key].setEnabled(
                not unavailable and (not spec.moves_robot or self.allow_motion.isChecked()))
        for action, button in self.action_buttons.items():
            button.setEnabled(not unavailable and (action == 'pose' or self.allow_motion.isChecked()))
        self.stop_program_btn.setEnabled(busy)
        spec = PROGRAM_BY_KEY.get(self.current_program_key or '')
        interactive = busy and spec is not None and spec.interactive
        self.command_entry.setEnabled(interactive)
        self.send_btn.setEnabled(interactive)
        self.start_server_btn.setEnabled(not self._busy(self.server_process))
        self.stop_server_btn.setEnabled(self._busy(self.server_process))
        self.stop_robot_btn.setEnabled(not self.stop_pending)
        enabled = self.allow_motion.isChecked()
        self.motion_state_label.setText('●  MOTION ENABLED' if enabled else '●  MOTION LOCKED')
        self.motion_state_label.setStyleSheet(
            'background: #4F392D; border: 1px solid #786044; color: #F8CC8E;'
            if enabled else 'background: #1A3340; border: 1px solid #385367; color: #A9CBDB;'
        )
        status = ('Software stop in progress' if self.stop_pending else
                  f'Running {self.current_program_key}' if busy else 'Idle / waiting for commands')
        self.console_status.setText(status)

    def _tick(self):
        if self.closing:
            return
        for _ in range(160):
            try:
                name, line = self.events.get_nowait()
            except queue.Empty:
                break
            if name == '_stop_complete':
                self.stop_pending = False
                self.log('Software-stop attempt finished. Confirm the robot is stationary before continuing.')
                self._update_button_states()
            else:
                self._append(f'[{name}] {line}', name if name in ('system', 'error', 'server') else '')
        if time.monotonic() - self.last_backend_check >= 1.0:
            self.last_backend_check = time.monotonic()
            online = self._backend_online()
            self.server_state_label.setText(f'●  127.0.0.1:{ROBOT_PORT}  ·  ' + ('ONLINE' if online else 'OFFLINE'))
            self.server_state_label.setStyleSheet(
                'background: #173B34; border: 1px solid #2B6250; color: #77E1AD;'
                if online else 'background: #372837; border: 1px solid #694255; color: #FC97A7;'
            )
            if online != self.last_backend_status:
                self.last_backend_status = online
                self.log('PAROL6 backend port is ' + ('reachable.' if online else 'unreachable.'))

    def _error_dialog(self, title: str, exc: Exception):
        self.log(str(exc), 'error')
        QMessageBox.critical(self, title, str(exc))

    def closeEvent(self, event: QCloseEvent):
        active = self.stop_pending or self._busy(self.program_process) or self._busy(self.server_process)
        if active:
            if QMessageBox.question(
                self, 'Close KARA Qt?',
                'A program, server, or stop operation is active.\n'
                'Closing terminates GUI-owned processes, but the arm may still be moving.\n'
                'Use the physical E-STOP if there is danger.\n\nClose?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) != QMessageBox.Yes:
                event.ignore()
                return
        if not self.camera_panel.stop_preview():
            QMessageBox.warning(self, 'Camera busy',
                                'The camera driver has not released the preview yet. Try closing again.')
            event.ignore()
            return
        self.closing = True
        self.timer.stop()
        # Synchronously wait briefly to avoid QProcess destroyed-while-running errors
        # at application shutdown; software stop remains a separate robot command.
        for proc in (self.program_process, self.server_process):
            if self._busy(proc):
                proc.terminate()
                if not proc.waitForFinished(700):
                    proc.kill()
                    proc.waitForFinished(500)
        event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(STYLESHEET)
    window = KaraProgramsGUI()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
