# tests the voice panel inside the gui, no mic or robot needed

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 not installed")
class DashboardVoiceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from gui.main import KaraProgramsGUI

        cls.app = QApplication.instance() or QApplication([])
        cls.gui = KaraProgramsGUI()

        # act like the robot server is off
        cls.gui._backend_online = lambda: False

        cls.panel = cls.gui.voice_panel
        cls.panel.speak_box.setChecked(False)

    @classmethod
    def tearDownClass(cls):
        cls.panel.shutdown()
        cls.gui.timer.stop()
        cls.gui.deleteLater()

    def say(self, text):
        self.panel._on_heard(text)
        return self.panel.reply_label.text()

    def test_voice_commands_go_through_dashboard_interlocks(self):
        self.gui.allow_motion.setChecked(False)
        self.assertIn("Turn on robot motion", self.say("home"))
        self.assertIsNone(self.panel.assistant.pending)

        self.gui.allow_motion.setChecked(True)
        self.assertIn("confirm home", self.say("home"))

        # saying confirm skips the popup, but the server check still runs
        reply = self.say("confirm home")
        self.assertTrue(reply.startswith("Could not run: PAROL6 backend not reachable"), reply)
        self.assertIsNone(self.gui.program_process)

        before = self.panel.reply_label.text()
        self.say("the robot looks nice today")
        self.assertEqual(self.panel.reply_label.text(), before)

        # a long reply used up the 8 seconds, the timer should restart once she's done talking
        self.say("park")
        self.panel.assistant.pending.until = time.monotonic() - 1
        self.panel._done_speaking()
        self.assertTrue(self.say("confirm park").startswith("Could not run"))

        self.assertIn("Software stop requested", self.say("stop"))
        self.assertTrue(self.gui.stop_pending)


if __name__ == "__main__":
    unittest.main()
