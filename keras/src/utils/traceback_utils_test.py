from keras.src import testing
from keras.src.utils.traceback_utils import disable_traceback_filtering
from keras.src.utils.traceback_utils import enable_traceback_filtering
from keras.src.utils.traceback_utils import filter_traceback
from keras.src.utils.traceback_utils import include_frame
from keras.src.utils.traceback_utils import inject_argument_info_in_traceback
from keras.src.utils.traceback_utils import is_traceback_filtering_enabled


class TracebackFilteringToggleTest(testing.TestCase):
    def test_default_enabled(self):
        enable_traceback_filtering()
        self.assertTrue(is_traceback_filtering_enabled())

    def test_disable(self):
        disable_traceback_filtering()
        self.assertFalse(is_traceback_filtering_enabled())
        # Re-enable for other tests
        enable_traceback_filtering()

    def test_enable_after_disable(self):
        disable_traceback_filtering()
        self.assertFalse(is_traceback_filtering_enabled())
        enable_traceback_filtering()
        self.assertTrue(is_traceback_filtering_enabled())

    def test_toggle_multiple_times(self):
        for _ in range(5):
            enable_traceback_filtering()
            self.assertTrue(is_traceback_filtering_enabled())
            disable_traceback_filtering()
            self.assertFalse(is_traceback_filtering_enabled())
        enable_traceback_filtering()


class IncludeFrameTest(testing.TestCase):
    def test_user_frame_included(self):
        self.assertTrue(include_frame("/home/user/my_project/train.py"))

    def test_keras_frame_excluded(self):
        import os

        keras_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        # A path under the keras source should be excluded
        self.assertFalse(include_frame(keras_path + "/src/layers/dense.py"))

    def test_tensorflow_frame_excluded(self):
        self.assertFalse(
            include_frame("/lib/python3.12/tensorflow/python/eager/execute.py")
        )


class FilterTracebackDecoratorTest(testing.TestCase):
    def test_normal_execution(self):
        @filter_traceback
        def my_func():
            return 42

        self.assertEqual(my_func(), 42)

    def test_exception_still_raised(self):
        @filter_traceback
        def my_func():
            raise ValueError("test error")

        with self.assertRaisesRegex(ValueError, "test error"):
            my_func()

    def test_disabled_filtering_passthrough(self):
        disable_traceback_filtering()
        try:

            @filter_traceback
            def my_func():
                raise ValueError("test")

            with self.assertRaises(ValueError):
                my_func()
        finally:
            enable_traceback_filtering()

    def test_preserves_function_name(self):
        @filter_traceback
        def my_custom_func():
            pass

        self.assertEqual(my_custom_func.__name__, "my_custom_func")


class InjectArgumentInfoTest(testing.TestCase):
    def test_normal_execution(self):
        def my_func(x, y):
            return x + y

        wrapped = inject_argument_info_in_traceback(my_func, "my_func")
        self.assertEqual(wrapped(1, 2), 3)

    def test_preserves_function_name(self):
        def my_func():
            pass

        wrapped = inject_argument_info_in_traceback(my_func)
        self.assertEqual(wrapped.__name__, "my_func")

    def test_exception_with_filtering_disabled(self):
        disable_traceback_filtering()
        try:

            def my_func():
                raise ValueError("err")

            wrapped = inject_argument_info_in_traceback(my_func, "my_func")
            with self.assertRaises(ValueError):
                wrapped()
        finally:
            enable_traceback_filtering()
