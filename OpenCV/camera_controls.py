"""Exposure and focus of the C920, set over USB.

OpenCV's camera properties do nothing on macOS, so UVC requests go straight to
the camera. Settings stay on the camera until changed.
"""

from __future__ import annotations

import array
import struct

try:
    import usb.backend.libusb1
    import usb.core
    import usb.util
except ImportError as exc:
    raise SystemExit(
        "Camera controls need pyusb. Install it with: "
        "uv pip install --python .venv/bin/python pyusb"
    ) from exc

LIBUSB_PATH = "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib"
C920_ID = (0x046D, 0x08E5)
# UVC requests. wIndex is (unit << 8 | interface); the C920's camera terminal
# is unit 1 on interface 0.
UVC_CAMERA_TERMINAL = 1 << 8
UVC_SET_CUR, UVC_GET_CUR, UVC_GET_MIN, UVC_GET_MAX = 0x01, 0x81, 0x82, 0x83
AE_MODE, AE_MANUAL, AE_AUTO = 0x02, 0x01, 0x08
EXPOSURE_TIME = 0x04
FOCUS_ABSOLUTE, FOCUS_AUTO = 0x06, 0x08
FOCUS_STEP = 5


class CameraControls:
    """Exposure and focus of a C920, set with UVC requests over USB.

    Sent straight to the device's control pipe: macOS holds the camera's
    interfaces while it streams, so pyusb's own ctrl_transfer, which claims
    them first, is refused.
    """

    def __init__(self) -> None:
        self._backend = usb.backend.libusb1.get_backend(
            find_library=lambda _: LIBUSB_PATH
        )
        if self._backend is None:
            raise OSError(f"libusb not found at {LIBUSB_PATH}; brew install libusb")
        for device in self._backend.enumerate_devices():
            descriptor = self._backend.get_device_descriptor(device)
            if (descriptor.idVendor, descriptor.idProduct) == C920_ID:
                self._handle = self._backend.open_device(device)
                break
        else:
            raise OSError("no C920 on USB")
        self.exposure_range = (self._read(UVC_GET_MIN, EXPOSURE_TIME, 4),
                               self._read(UVC_GET_MAX, EXPOSURE_TIME, 4))
        self.focus_range = (self._read(UVC_GET_MIN, FOCUS_ABSOLUTE, 2),
                            self._read(UVC_GET_MAX, FOCUS_ABSOLUTE, 2))

    def _read(self, request: int, selector: int, size: int) -> int:
        buffer = usb.util.create_buffer(size)
        count = self._backend.ctrl_transfer(
            self._handle, 0xA1, request, selector << 8, UVC_CAMERA_TERMINAL, buffer, 500
        )
        return int.from_bytes(bytes(buffer[:count]), "little")

    def _write(self, selector: int, payload: bytes) -> None:
        self._backend.ctrl_transfer(
            self._handle, 0x21, UVC_SET_CUR, selector << 8, UVC_CAMERA_TERMINAL,
            array.array("B", payload), 500
        )

    @property
    def exposure(self) -> int:
        return self._read(UVC_GET_CUR, EXPOSURE_TIME, 4)

    @property
    def focus(self) -> int:
        return FOCUS_STEP * round(self._read(UVC_GET_CUR, FOCUS_ABSOLUTE, 2) / FOCUS_STEP)

    @property
    def auto_exposure(self) -> bool:
        return self._read(UVC_GET_CUR, AE_MODE, 1) != AE_MANUAL

    @property
    def auto_focus(self) -> bool:
        return self._read(UVC_GET_CUR, FOCUS_AUTO, 1) != 0

    def set_auto_exposure(self) -> None:
        self._write(AE_MODE, bytes([AE_AUTO]))

    def set_auto_focus(self) -> None:
        self._write(FOCUS_AUTO, b"\x01")

    def set_exposure(self, value: int) -> None:
        self._write(AE_MODE, bytes([AE_MANUAL]))
        self._write(EXPOSURE_TIME, struct.pack("<I", value))

    def set_focus(self, value: int) -> None:
        self._write(FOCUS_AUTO, b"\x00")
        self._write(FOCUS_ABSOLUTE, struct.pack("<H", value))

    def close(self) -> None:
        self._backend.close_device(self._handle)


def open_controls() -> CameraControls | None:
    """The camera's controls, or None (and why) so callers can go without."""
    try:
        return CameraControls()
    except (OSError, usb.core.USBError) as exc:
        print(f"Exposure and focus controls unavailable: {exc}")
        return None
