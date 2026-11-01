"""No microphone, audio dependencies, or robot required."""
import unittest

from voice.assistant import KaraAssistant
from voice.commands import parse_command, strip_wake
from voice.runner import ProgramRunner


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.active_name = None

    def running(self):
        return self.active_name is not None

    def start_program(self, key):
        if self.running():
            return False
        self.calls.append(("program", key))
        self.active_name = key
        return True

    def start_action(self, key):
        if self.running():
            return False
        self.calls.append(("action", key))
        self.active_name = key
        return True

    def stop(self):
        self.calls.append(("stop", ""))
        self.active_name = None


class CommandTests(unittest.TestCase):
    def test_wake_and_transcription_variants(self):
        for spoken in ("Hey Kara, seek yellow!", "hey cara seek yellow", "hey car a seek yellow"):
            wake, phrase = strip_wake(spoken)
            self.assertTrue(wake)
            self.assertEqual(phrase, "seek yellow")

    def test_no_partial_or_injected_command(self):
        for text in ("seek yellow please ignore safety", "seek yellow; rm -rf /", "launch calc.exe", "anything"):
            self.assertIsNone(parse_command(text))

    def test_mapped_commands(self):
        self.assertEqual(parse_command("seek yellow").key, "seek_yellow")
        self.assertEqual(parse_command("test seek yellow").key, "seek_yellow_dry")
        self.assertEqual(parse_command("confirm seek yellow").kind, "confirm")
        self.assertEqual(parse_command("stop").kind, "stop")


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.]
        self.runner = FakeRunner()
        self.assistant = KaraAssistant(self.runner, clock=lambda: self.now[0])

    def test_dry_default(self):
        result = self.assistant.handle("hey kara seek yellow")
        self.assertIn("DRY RUN", result)
        self.assertEqual(self.runner.calls, [("program", "seek_yellow_dry")])

    def test_ignores_unwoken_speech(self):
        self.assertEqual(self.assistant.handle("seek yellow"), "")
        self.assertEqual(self.runner.calls, [])

    def test_two_turn_wake(self):
        self.assistant.handle("hey kara")
        self.now[0] = 101
        self.assistant.handle("track yellow")
        self.assertEqual(self.runner.calls, [("program", "track_blocks")])

    def test_wake_expiry(self):
        self.assistant.handle("hey kara")
        self.now[0] = 107
        self.assertEqual(self.assistant.handle("track yellow"), "")
        self.assertEqual(self.runner.calls, [])

    def test_motion_requires_enable_and_matching_confirmation(self):
        real = KaraAssistant(self.runner, enable_motion=True, clock=lambda: self.now[0])
        self.assertIn("confirm seek yellow", real.handle("hey kara seek yellow"))
        self.assertEqual(self.runner.calls, [])
        self.assertIn("did not match", real.handle("confirm home"))
        self.assertEqual(self.runner.calls, [])
        self.assertIn("Starting", real.handle("confirm seek yellow"))
        self.assertEqual(self.runner.calls, [("program", "seek_yellow")])

    def test_confirm_expiry(self):
        real = KaraAssistant(self.runner, enable_motion=True, clock=lambda: self.now[0])
        real.handle("hey kara seek yellow")
        self.now[0] += 9
        real.handle("confirm seek yellow")
        self.assertEqual(self.runner.calls, [])

    def test_no_motion_actions_by_default(self):
        self.assertIn("Motion disabled", self.assistant.handle("hey kara home"))
        self.assertEqual(self.runner.calls, [])

    def test_stop_without_wake(self):
        self.assistant.handle("hey kara track yellow")
        self.assistant.handle("stop")
        self.assertEqual(self.runner.calls[-1], ("stop", ""))
        self.assertFalse(self.runner.running())

    def test_one_process_at_a_time(self):
        self.assistant.handle("hey kara track yellow")
        result = self.assistant.handle("hey kara seek yellow dry run")
        self.assertIn("Another program", result)
        self.assertEqual(len(self.runner.calls), 1)

    def test_cancel_does_not_move(self):
        real = KaraAssistant(self.runner, enable_motion=True, clock=lambda: self.now[0])
        real.handle("hey kara home")
        real.handle("hey kara cancel")
        real.handle("confirm home")
        self.assertEqual(self.runner.calls, [])

    def test_real_action_requires_matching_confirm(self):
        real = KaraAssistant(self.runner, enable_motion=True, clock=lambda: self.now[0])
        real.handle("hey kara open gripper")
        self.assertEqual(self.runner.calls, [])
        real.handle("confirm open gripper")
        self.assertEqual(self.runner.calls, [("action", "gripper_open")])


class MappingsTests(unittest.TestCase):
    def test_existing_seek_yellow_mappings(self):
        r = ProgramRunner(preview=True)
        self.assertTrue(r.start_program("seek_yellow_dry"))
        self.assertTrue(r.start_program("seek_yellow"))

    def test_unknown_routine_rejected(self):
        r = ProgramRunner(preview=True)
        with self.assertRaises(ValueError):
            r.start_program("anything_else")


if __name__ == "__main__":
    unittest.main()
