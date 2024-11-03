import sys
import tempfile
from unittest.mock import patch

from keras.src.testing import test_case
from keras.src.utils import io_utils


class TestIoUtils(test_case.TestCase):
    def test_enable_interactive_logging(self):
        io_utils.enable_interactive_logging()
        self.assertTrue(io_utils.is_interactive_logging_enabled())

    def test_disable_interactive_logging(self):
        io_utils.disable_interactive_logging()
        self.assertFalse(io_utils.is_interactive_logging_enabled())

    def test_set_logging_verbosity_valid(self):
        valid_levels = ["FATAL", "ERROR", "WARNING", "INFO", "DEBUG"]
        for level in valid_levels:
            io_utils.set_logging_verbosity(level)

    def test_set_logging_verbosity_invalid(self):
        with self.assertRaises(ValueError):
            io_utils.set_logging_verbosity("INVALID")

    @patch("builtins.input", side_effect=["y"])
    def test_ask_to_proceed_with_overwrite_yes(self, _):
        self.assertTrue(io_utils.ask_to_proceed_with_overwrite("test_path"))

    @patch("builtins.input", side_effect=["n"])
    def test_ask_to_proceed_with_overwrite_no(self, _):
        self.assertFalse(io_utils.ask_to_proceed_with_overwrite("test_path"))

    @patch("sys.stdout.write")
    def test_print_msg_interactive_with_line_break(self, mock_write):
        io_utils.enable_interactive_logging()
        io_utils.print_msg("Hello", line_break=True)
        mock_write.assert_called_once_with("Hello\n")

    @patch("sys.stdout.write")
    def test_print_msg_interactive_without_line_break(self, mock_write):
        io_utils.enable_interactive_logging()
        io_utils.print_msg("Hello", line_break=False)
        mock_write.assert_called_once_with("Hello")

    @patch("absl.logging.info")
    def test_print_msg_non_interactive(self, mock_logging):
        io_utils.disable_interactive_logging()
        io_utils.print_msg("Hello")
        mock_logging.assert_called_once_with("Hello")

    @patch("builtins.input", side_effect=["invalid", "invalid", "y"])
    def test_ask_to_proceed_with_overwrite_invalid_then_yes(self, _):
        self.assertTrue(io_utils.ask_to_proceed_with_overwrite("test_path"))

    @patch("builtins.input", side_effect=["invalid", "n"])
    def test_ask_to_proceed_with_overwrite_invalid_then_no(self, _):
        self.assertFalse(io_utils.ask_to_proceed_with_overwrite("test_path"))

    def test_print_msg_with_different_encoding(self):
        # https://github.com/keras-team/keras/issues/19386
        io_utils.enable_interactive_logging()
        self.assertTrue(io_utils.is_interactive_logging_enabled())
        ori_stdout = sys.stdout
        with tempfile.TemporaryFile(mode="w", encoding="cp1251") as tmp:
            sys.stdout = tmp
            io_utils.print_msg("━")
        sys.stdout = ori_stdout
