"""Stdlib viewer/launcher contracts; no Isaac app, Torch, GPU, or Windows process.

Runtime dependencies below are small test doubles. These tests cannot establish
Windows execution, renderer/video correctness, physics parity, or locomotion.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE_ROOT / "humanoid_rl/scripts"
VIEWER = SCRIPTS / "play_torchscript.py"
LAUNCHER = SCRIPTS / "view_native_windows.cmd"
ROUGH_TASK = "Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0"
STAIRS_TASK = "Isaac-Locomotion-Stairs-PioneerHumanoid-Play-v0"


def load_viewer():
    spec = importlib.util.spec_from_file_location("_native_viewer_source_test", VIEWER)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PACKAGE_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def runtime_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class ViewerArgumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viewer = load_viewer()

    def parse(self, *extra):
        class Launcher:
            def __init__(self, *_args):
                raise AssertionError("Argument parsing must not construct an Isaac application")

            @staticmethod
            def add_app_launcher_args(parser):
                parser.add_argument("--headless", action="store_true")
                parser.add_argument("--enable_cameras", action="store_true")
                parser.add_argument("--device", default="cuda:0")

        isaac = runtime_module("isaaclab", __path__=[])
        app = runtime_module("isaaclab.app", AppLauncher=Launcher)
        isaac.app = app
        argv = [str(VIEWER), "--policy", "policy.pt", "--metadata", "policy.json", *extra]
        with mock.patch.dict(sys.modules, {"isaaclab": isaac, "isaaclab.app": app}), \
                mock.patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
            return self.viewer._parse_args()[0]

    def test_import_does_not_require_runtime_or_launch_application(self):
        unavailable = {name: None for name in ("isaaclab", "isaaclab.app", "torch", "gymnasium", "carb", "omni")}
        with mock.patch.dict(sys.modules, unavailable):
            self.assertTrue(callable(load_viewer().main))

    def test_gui_defaults_watch_indefinitely_without_recording(self):
        args = self.parse()
        self.assertEqual(args.task, ROUGH_TASK)
        self.assertEqual(args.max_steps, 0)
        self.assertFalse(args.headless)
        self.assertFalse(args.video)
        self.assertFalse(args.enable_cameras)
        self.assertEqual(tuple(args.command), (0.5, 0.0, 0.0))

    def test_recording_enables_cameras(self):
        args = self.parse("--video", "--video_seconds", "20")
        self.assertTrue(args.enable_cameras)
        self.assertEqual(args.video_seconds, 20.0)

    def test_flat_override_accepts_rough_task_and_equivalent_alias(self):
        for task in (ROUGH_TASK, ROUGH_TASK.replace("Isaac-Locomotion-", "Isaac-Velocity-")):
            with self.subTest(task=task):
                self.assertTrue(self.parse("--task", task, "--flat_terrain").flat_terrain)

    def test_flat_override_rejects_stairs_task_and_equivalent_alias(self):
        for task in (STAIRS_TASK, STAIRS_TASK.replace("Isaac-Locomotion-", "Isaac-Velocity-")):
            with self.subTest(task=task), self.assertRaises(SystemExit) as error:
                self.parse("--task", task, "--flat_terrain")
            self.assertEqual(error.exception.code, 2)

    def test_rejects_invalid_counts_duration_and_nonfinite_commands(self):
        for options in (
            ("--num_envs", "0"), ("--max_steps", "-1"),
            ("--video_seconds", "0"), ("--video_seconds", "nan"),
            ("--video_seconds", "inf"), ("--command", "nan", "0", "0"),
            ("--command", "0", "inf", "0"),
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit) as error:
                self.parse(*options)
            self.assertEqual(error.exception.code, 2)


class ViewerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.viewer = load_viewer()
        self.args = SimpleNamespace(task=ROUGH_TASK, policy="policy.pt", metadata="policy.json")
        self.manifest = {"policy_sha256": "a" * 64}
        self.app = SimpleNamespace(close=mock.Mock())
        self.launcher = mock.Mock(return_value=SimpleNamespace(app=self.app))
        self.usd = Path("mapped_repo/assets/whole_body_humanoid/usd/whole_body_humanoid.usd")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(self.viewer, "_parse_args", return_value=(self.args, self.launcher)))
        self.metadata = self.stack.enter_context(mock.patch.object(self.viewer, "load_metadata", return_value=self.manifest))
        self.digest = self.stack.enter_context(mock.patch.object(self.viewer, "sha256", return_value="a" * 64))
        self.assets = self.stack.enter_context(mock.patch.object(self.viewer, "validate_assets", return_value=self.usd))
        self.play = self.stack.enter_context(mock.patch.object(self.viewer, "_play"))
        self.order = mock.Mock()
        for name, function in (
            ("metadata", self.metadata), ("hash", self.digest), ("assets", self.assets),
            ("launch", self.launcher), ("play", self.play), ("close", self.app.close),
        ):
            self.order.attach_mock(function, name)

    def test_artifacts_are_checked_before_app_launch_and_app_closes_after_play(self):
        self.viewer.main()
        self.assertEqual([call[0] for call in self.order.mock_calls], ["metadata", "hash", "assets", "launch", "play", "close"])
        self.play.assert_called_once_with(self.args, self.app, Path("policy.pt").absolute(), self.manifest, self.usd)

    def test_manifest_failure_does_not_open_app(self):
        self.metadata.side_effect = ValueError("invalid manifest")
        with self.assertRaisesRegex(ValueError, "invalid manifest"):
            self.viewer.main()
        self.digest.assert_not_called()
        self.launcher.assert_not_called()

    def test_policy_hash_failure_does_not_open_app(self):
        self.digest.return_value = "b" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            self.viewer.main()
        self.assets.assert_not_called()
        self.launcher.assert_not_called()

    def test_asset_failure_does_not_open_app(self):
        self.assets.side_effect = ValueError("changed assets")
        with self.assertRaisesRegex(ValueError, "changed assets"):
            self.viewer.main()
        self.launcher.assert_not_called()

    def test_play_failure_still_closes_app(self):
        self.play.side_effect = RuntimeError("runtime failure")
        with self.assertRaisesRegex(RuntimeError, "runtime failure"):
            self.viewer.main()
        self.app.close.assert_called_once_with()

    def test_app_launcher_mutation_cannot_remove_playback_arguments(self):
        def mutate_launcher_options(options):
            self.assertIsInstance(options, dict)
            self.assertIsNot(options, vars(self.args))
            options.clear()
            return SimpleNamespace(app=self.app)

        self.launcher.side_effect = mutate_launcher_options
        self.viewer.main()
        self.assertEqual(self.args.task, ROUGH_TASK)
        self.assertEqual(self.args.policy, "policy.pt")
        self.play.assert_called_once_with(self.args, self.app, Path("policy.pt").absolute(), self.manifest, self.usd)

    def test_mapped_repository_spelling_is_not_resolved(self):
        alias_root = Path("mapped_repository_alias").absolute()
        source = alias_root / "src/simulation/humanoid_rl/humanoid_rl/scripts/play_torchscript.py"
        with mock.patch.object(self.viewer, "__file__", str(source)), \
                mock.patch.object(Path, "resolve", side_effect=AssertionError("Do not resolve mapped drives")):
            self.viewer.main()
        self.assets.assert_called_once_with(alias_root, self.manifest)


class ViewerKeyboardTests(unittest.TestCase):
    def setUp(self):
        self.viewer = load_viewer()
        self.interface = SimpleNamespace(
            subscribe_to_keyboard_events=mock.Mock(return_value="subscription"),
            unsubscribe_to_keyboard_events=mock.Mock(),
        )
        self.window = SimpleNamespace(get_keyboard=lambda: "keyboard")
        self.input = runtime_module(
            "carb.input", acquire_input_interface=lambda: self.interface,
            KeyboardEventType=SimpleNamespace(KEY_PRESS="press"),
        )
        self.appwindow = runtime_module("omni.appwindow", get_default_app_window=mock.Mock(return_value=self.window))
        self.modules = {
            "carb": runtime_module("carb", __path__=[], input=self.input), "carb.input": self.input,
            "omni": runtime_module("omni", __path__=[], appwindow=self.appwindow), "omni.appwindow": self.appwindow,
        }

    def test_headless_exit_state_needs_no_keyboard_runtime(self):
        with mock.patch.dict(sys.modules, {"carb": None, "omni": None}):
            state, cleanup = self.viewer._exit_key(True)
        self.assertEqual(state, {"requested": False})
        self.assertIsNone(cleanup())

    def test_missing_window_needs_no_subscription(self):
        self.appwindow.get_default_app_window.return_value = None
        with mock.patch.dict(sys.modules, self.modules), redirect_stdout(io.StringIO()):
            state, cleanup = self.viewer._exit_key(False)
        self.assertFalse(state["requested"])
        self.interface.subscribe_to_keyboard_events.assert_not_called()
        self.assertIsNone(cleanup())

    def test_q_and_escape_press_request_exit_and_cleanup_unsubscribes(self):
        with mock.patch.dict(sys.modules, self.modules):
            state, cleanup = self.viewer._exit_key(False)
        callback = self.interface.subscribe_to_keyboard_events.call_args.args[1]
        for kind, name, expected in (("press", "Q", True), ("press", "ESCAPE", True), ("release", "Q", False), ("press", "A", False)):
            with self.subTest(kind=kind, name=name):
                state["requested"] = False
                self.assertTrue(callback(SimpleNamespace(type=kind, input=SimpleNamespace(name=name))))
                self.assertEqual(state["requested"], expected)
        cleanup()
        self.interface.unsubscribe_to_keyboard_events.assert_called_once_with("keyboard", "subscription")


class ViewerRuntimeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.viewer = load_viewer()
        self.args = SimpleNamespace(
            task=ROUGH_TASK, device="cuda:0", num_envs=1, disable_fabric=False, seed=42,
            flat_terrain=False, command=(0.5, 0.0, 0.0), video=False, video_seconds=20.0,
            video_folder="outputs/native_viewer/videos",
        )
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.02, device="cuda:0"), metadata={}, close=mock.Mock())
        self.wrapper = SimpleNamespace(close=mock.Mock())
        self.record = mock.Mock(return_value=self.wrapper)
        self.gym = runtime_module("gymnasium", make=mock.Mock(return_value=self.env), wrappers=SimpleNamespace(RecordVideo=self.record))
        self.policy = SimpleNamespace(eval=lambda: "policy")
        self.torch = runtime_module("torch", jit=SimpleNamespace(load=mock.Mock(return_value=self.policy)))
        utils = runtime_module("isaaclab_tasks.utils", parse_env_cfg=mock.Mock(return_value=SimpleNamespace()))
        modules = {
            "gymnasium": self.gym, "torch": self.torch,
            "humanoid_rl_tasks": runtime_module("humanoid_rl_tasks"),
            "isaaclab_tasks": runtime_module("isaaclab_tasks", __path__=[], utils=utils), "isaaclab_tasks.utils": utils,
        }
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(sys.modules, modules))
        for name in ("validate_env_config", "_fixed_command", "_cached_usd_spawn", "_camera_config"):
            self.stack.enter_context(mock.patch.object(self.viewer, name))
        self.layout = self.stack.enter_context(mock.patch.object(self.viewer, "validate_runtime_layout"))
        self.stack.enter_context(mock.patch.object(self.viewer, "_check_probe", side_effect=RuntimeError("probe failed")))

    def play(self):
        self.viewer._play(self.args, SimpleNamespace(), Path("policy.pt"), {}, Path("robot.usd"))

    def test_runtime_layout_failure_closes_environment(self):
        self.layout.side_effect = RuntimeError("layout failed")
        with self.assertRaisesRegex(RuntimeError, "layout failed"):
            self.play()
        self.env.close.assert_called_once_with()
        self.torch.jit.load.assert_not_called()

    def test_video_wrapper_failure_closes_original_environment(self):
        self.args.video = True
        self.record.side_effect = RuntimeError("wrapper failed")
        with self.assertRaisesRegex(RuntimeError, "wrapper failed"):
            self.play()
        self.env.close.assert_called_once_with()

    def test_video_uses_simulation_duration_and_closes_wrapper_on_probe_failure(self):
        self.args.video = True
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            self.play()
        self.assertEqual(self.env.metadata["render_fps"], 50)
        self.record.assert_called_once()
        options = self.record.call_args.kwargs
        self.assertEqual(options["video_length"], 1000)
        self.assertEqual(options["video_folder"], str(Path(self.args.video_folder).absolute()))
        self.assertTrue(options["step_trigger"](0))
        self.assertFalse(options["step_trigger"](1))
        self.wrapper.close.assert_called_once_with()

    def test_keyboard_unsubscribe_failure_still_closes_environment(self):
        self.args.headless = False
        self.env.reset = mock.Mock(return_value=({"policy": SimpleNamespace(shape=(1, 235))}, {}))
        unsubscribe = mock.Mock(side_effect=RuntimeError("unsubscribe failed"))
        metadata = {"observation_dim": 235, "action_dim": 12, "step_dt": 0.02}
        app = SimpleNamespace(is_running=lambda: False)
        with mock.patch.object(self.viewer, "_check_probe", return_value=0.0), \
                mock.patch.object(self.viewer, "_exit_key", return_value=({"requested": False}, unsubscribe)), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "unsubscribe failed"):
            self.viewer._play(self.args, app, Path("policy.pt"), metadata, Path("robot.usd"))
        unsubscribe.assert_called_once_with()
        self.env.close.assert_called_once_with()


class WindowsLauncherSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_launcher_parent_traversal_reaches_repository_root(self):
        match = re.search(r'^pushd "%~dp0((?:\.\.\\){4}\.\.)"$', self.text, re.MULTILINE)
        self.assertIsNotNone(match)
        root = LAUNCHER.parent
        for component in match.group(1).split("\\"):
            self.assertEqual(component, "..")
            root = root.parent
        self.assertEqual(root, PACKAGE_ROOT.parents[2])

    def test_arguments_pass_unchanged_and_exit_code_is_captured_before_cleanup(self):
        self.assertIn('call "%WATO_ISAACLAB_BAT%" -p -u "%WATO_VIEWER%" %*', self.text)
        self.assertEqual(self.text.count("%*"), 1)
        self.assertNotRegex(self.text, r"(?im)^\s*shift\b")
        self.assertLess(self.text.index('set "WATO_RESULT=%ERRORLEVEL%"'), self.text.rindex("\npopd\n"))
        self.assertLess(self.text.rindex("\npopd\n"), self.text.index("exit /b %WATO_RESULT%"))
        self.assertIn("setlocal EnableExtensions DisableDelayedExpansion", self.text)

    def test_source_paths_and_explicit_installation_input_without_private_dependencies(self):
        self.assertIn("if not defined ISAACLAB_BAT", self.text)
        for source in ("src\\simulation\\humanoid_rl", "src\\simulation\\humanoid_rl_tasks", "src\\pioneer_humanoid"):
            self.assertIn("%WATO_REPO%\\" + source, self.text)
        self.assertIn("\\scripts\\play_torchscript.py", self.text)
        self.assertNotRegex(self.text, r"(?im)^\s*(?:start|pause)\b")
        for private in ("/home/nick", "C:\\Users\\", "outputs\\wato_native_viewer", "pinned_checkpoint", "--latest", "--headless", "vulkan=false"):
            self.assertNotIn(private, self.text)


if __name__ == "__main__":
    unittest.main()
