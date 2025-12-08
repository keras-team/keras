import collections

from keras.src import backend
from keras.src import testing
from keras.src.utils import tracking


class TrackingTest(testing.TestCase):
    def test_untracking_in_tracked_list(self):
        tracked_variables = []
        tracker = tracking.Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    tracked_variables,
                ),
            }
        )
        v1 = backend.Variable(1.0)
        v2 = backend.Variable(2.0)
        lst = tracking.TrackedList([], tracker)
        lst.append(v1)
        lst.append(float("nan"))
        lst.append(v2)
        lst.append(0)

        self.assertLen(tracked_variables, 2)
        self.assertEqual(tracked_variables[0], v1)
        self.assertEqual(tracked_variables[1], v2)

        lst.remove(v1)
        self.assertLen(lst, 3)
        self.assertLen(tracked_variables, 1)

        lst.remove(v2)
        self.assertLen(lst, 2)
        self.assertLen(tracked_variables, 0)

        lst2 = tracking.TrackedList([], tracker)
        lst2.append(v1)
        lst2.append(float("nan"))
        lst2.append(v2)
        lst2.append(0)

        popped_value = lst2.pop()
        self.assertEqual(popped_value, 0)
        self.assertLen(lst2, 3)
        self.assertLen(tracked_variables, 2)

        lst2.clear()
        self.assertLen(lst2, 0)
        self.assertLen(tracked_variables, 0)

        lst2.append(v1)
        lst2.append(v2)
        del lst2[0]
        self.assertLen(lst2, 1)
        self.assertLen(tracked_variables, 1)

    def test_tuple_tracking(self):
        tracked_variables = []
        tracker = tracking.Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    tracked_variables,
                ),
            }
        )
        v1 = backend.Variable(1.0)
        v2 = backend.Variable(2.0)
        tup = (v1, v2)
        tup = tracker.track(tup)
        self.assertIsInstance(tup, tuple)
        self.assertLen(tracked_variables, 2)
        self.assertEqual(tracked_variables[0], v1)
        self.assertEqual(tracked_variables[1], v2)

    def test_namedtuple_tracking(self):
        tracked_variables = []
        tracker = tracking.Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    tracked_variables,
                ),
            }
        )
        v1 = backend.Variable(1.0)
        v2 = backend.Variable(2.0)
        nt = collections.namedtuple("NT", ["x", "y"])
        tup = nt(x=v1, y=v2)
        tup = tracker.track(tup)
        self.assertIsInstance(tup, tuple)
        self.assertEqual(tup.x, v1)
        self.assertEqual(tup.y, v2)
        self.assertLen(tracked_variables, 2)
        self.assertEqual(tracked_variables[0], v1)
        self.assertEqual(tracked_variables[1], v2)
