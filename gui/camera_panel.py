"""Embedded USB workspace camera for KARA's Qt GUI.

Only this module opens local cameras; robot and calibration programs remain
independent. A camera-consuming program must first release this preview.

OpenCV is an optional runtime dependency so the rest of the robot dashboard
can start without a camera or cv2 being installed.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)


class CameraWorker(QThread):
    """Read frames on a worker thread, never on Qt's GUI thread."""

    frame_ready = Signal(QImage)
    camera_ready = Signal(str)
    failed = Signal(str)

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.index = index
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.failed.emit('OpenCV is missing. Install OpenCV/requirements.txt into the GUI Python environment.')
            return

        cap = None
        try:
            # DirectShow is often more reliable for physical USB webcams on Windows.
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap.release()
                cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                self.failed.emit(f'Cannot open camera {self.index}. Check the index, USB connection, and other apps.')
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera_ready.emit(f'USB CAMERA {self.index}  ·  LIVE')
            bad_reads = 0
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if self._stop.is_set():
                    break
                if not ok or bgr is None:
                    bad_reads += 1
                    if bad_reads >= 20:
                        self.failed.emit('Camera stopped returning frames. Check the cable or camera connection.')
                        break
                    self.msleep(70)
                    continue
                bad_reads = 0
                # Deep-copy before emitting: OpenCV reuses the frame buffer.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width, _ = rgb.shape
                self.frame_ready.emit(QImage(rgb.data, width, height,
                                             rgb.strides[0], QImage.Format_RGB888).copy())
                # Keep the GUI responsive without overloading low-powered machines.
                self.msleep(35)
        except Exception as exc:
            self.failed.emit(f'Camera error: {exc}')
        finally:
            if cap is not None:
                cap.release()


class CameraCanvas(QLabel):
    """Show video preserving its aspect ratio; keep the latest frame for resizes."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('cameraCanvas')
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(160)
        self.setText('NO CAMERA SIGNAL\n\nSelect a camera and click Start camera')
        self._image: QImage | None = None

    def show_frame(self, frame: QImage) -> None:
        self._image = frame
        self._render()

    def show_message(self, message: str) -> None:
        self._image = None
        self.clear()
        self.setText(message)

    def current_frame(self) -> QImage | None:
        return self._image

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image is not None:
            self._render()

    def _render(self):
        if self._image is None:
            return
        self.setPixmap(QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class CameraPopup(QDialog):
    """Optional larger view; reuses frames from the same camera thread."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle('KARA  |  Workspace camera')
        self.setObjectName('cameraDialog')
        self.resize(1100, 720)
        self.setMinimumSize(480, 320)
        layout = QVBoxLayout(self)
        self.canvas = CameraCanvas()
        layout.addWidget(self.canvas, 1)


class CameraPanel(QFrame):
    """Qt camera controls + in-dashboard video without a second capture handle."""

    message = Signal(str)

    def __init__(self, index_provider: Callable[[], int], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('cameraPanel')
        self.index_provider = index_provider
        self._worker: CameraWorker | None = None
        self._popup: CameraPopup | None = None
        self._last_image: QImage | None = None
        self._reserved = False
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 14, 15, 13)
        layout.setSpacing(9)
        top = QHBoxLayout()
        top.setSpacing(9)
        heading = QVBoxLayout()
        heading.setSpacing(3)
        title = QLabel('Workspace camera')
        title.setObjectName('sectionTitle')
        heading.addWidget(title)
        sub = QLabel('Live USB camera feed  ·  view your table and robot workspace')
        sub.setObjectName('mutedText')
        heading.addWidget(sub)
        top.addLayout(heading, 1)
        self.state = QLabel('●  CAMERA OFF')
        self.state.setObjectName('cameraStatus')
        top.addWidget(self.state)
        layout.addLayout(top)

        self.canvas = CameraCanvas()
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.start_btn = QPushButton('▶  Start camera')
        self.start_btn.setObjectName('primaryButton')
        self.start_btn.clicked.connect(self.start_preview)
        controls.addWidget(self.start_btn)
        self.stop_btn = QPushButton('■  Stop')
        self.stop_btn.setObjectName('ghostButton')
        self.stop_btn.clicked.connect(self.stop_preview)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        self.snapshot_btn = QPushButton('Snapshot')
        self.snapshot_btn.setObjectName('ghostButton')
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        controls.addWidget(self.snapshot_btn)
        self.expand_btn = QPushButton('Expand ↗')
        self.expand_btn.setObjectName('softButton')
        self.expand_btn.clicked.connect(self.expand_preview)
        controls.addWidget(self.expand_btn)
        layout.addLayout(controls)
        self.note = QLabel('Camera is shared: preview pauses while OpenCV detection programs run.')
        self.note.setObjectName('mutedText')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self._update_controls()

    def is_active(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def has_session(self) -> bool:
        """Includes a worker that is still opening a slow USB camera."""
        return self._worker is not None

    def set_reserved(self, reserved: bool) -> None:
        self._reserved = reserved
        if reserved:
            self.state.setText('●  IN USE BY PROGRAM')
            self.canvas.show_message('CAMERA CONTROLLED BY OPENCV PROGRAM\n\nThe workspace preview resumes when the script exits')
            self._last_image = None
            if self._popup is not None:
                self._popup.canvas.show_message('CAMERA IN USE BY OPENCV PROGRAM')
        elif not self.has_session():
            self.state.setText('●  CAMERA OFF')
            self.canvas.show_message('CAMERA OFF\n\nClick Start camera to view your workspace')
        self._update_controls()

    def start_preview(self) -> None:
        if self._reserved:
            self.message.emit('Camera preview unavailable while an OpenCV program owns this camera.')
            return
        if self._worker is not None:
            # A driver may still be releasing the previous worker.
            if self._worker.isRunning():
                return
            self._worker = None
        index = self.index_provider()
        self._last_image = None
        self.canvas.show_message(f'CONNECTING TO CAMERA {index}…')
        self.state.setText('●  CONNECTING')
        worker = CameraWorker(index, self)
        self._worker = worker
        worker.frame_ready.connect(self._on_frame)
        worker.camera_ready.connect(self._on_ready)
        worker.failed.connect(self._on_failure)
        worker.finished.connect(lambda w=worker: self._on_finished(w))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._update_controls()
        self.message.emit(f'Opening workspace camera {index}.')

    def stop_preview(self) -> bool:
        """Return False if driver hasn't released camera (caller must not launch cv2)."""
        worker = self._worker
        if worker is None:
            return True
        worker.request_stop()
        if not worker.wait(5000):
            self.state.setText('●  CAMERA BUSY')
            self.message.emit('Camera driver did not release within 5 seconds. Another camera script was NOT started.')
            return False
        if self._worker is worker:
            self._worker = None
        self.canvas.show_message('CAMERA PAUSED\n\nClick Start camera to resume the workspace view')
        self._last_image = None
        self.state.setText('●  CAMERA OFF')
        if self._popup is not None:
            self._popup.canvas.show_message('CAMERA PAUSED')
        self._update_controls()
        return True

    def _on_frame(self, image: QImage):
        if self._worker is None or not self._worker.isRunning():
            return
        self._last_image = image
        self.canvas.show_frame(image)
        if self._popup is not None and self._popup.isVisible():
            self._popup.canvas.show_frame(image)
        self.snapshot_btn.setEnabled(True)

    def _on_ready(self, label: str):
        if self.is_active():
            self.state.setText('●  LIVE')
            self.message.emit(label)

    def _on_failure(self, message: str):
        self.state.setText('●  UNAVAILABLE')
        self.canvas.show_message(message + '\n\nCheck camera permissions, index, and whether another script owns it.')
        self.message.emit(message)

    def _on_finished(self, worker: CameraWorker):
        if self._worker is worker:
            self._worker = None
            if 'UNAVAILABLE' not in self.state.text():
                self.state.setText('●  CAMERA OFF')
            self._update_controls()

    def _update_controls(self):
        running = self.has_session()
        self.start_btn.setEnabled(not running and not self._reserved)
        self.stop_btn.setEnabled(running)
        self.snapshot_btn.setEnabled(self._last_image is not None)
        self.expand_btn.setEnabled(True)

    def save_snapshot(self) -> None:
        if self._last_image is None:
            QMessageBox.information(self, 'No image', 'Start the camera and capture a frame first.')
            return
        default = str(Path.home() / f'KARA_workspace_{time.strftime("%Y%m%d_%H%M%S")}.png')
        path, _ = QFileDialog.getSaveFileName(self, 'Save workspace snapshot', default,
                                              'PNG Image (*.png);;JPEG Image (*.jpg)')
        if path:
            if not self._last_image.save(path):
                QMessageBox.warning(self, 'Save failed', f'Could not save image to {path}.')
            else:
                self.message.emit(f'Workspace snapshot saved: {path}')

    def expand_preview(self):
        if self._popup is None:
            self._popup = CameraPopup(self)
        if self._last_image is not None:
            self._popup.canvas.show_frame(self._last_image)
        else:
            self._popup.canvas.show_message('NO LIVE FRAME\n\nStart camera from the KARA dashboard.')
        self._popup.show()
        self._popup.raise_()
        self._popup.activateWindow()
