import tensorflow as tf

from keras.utils import tracking


class KerasAutoTrackable(tf.__internal__.tracking.AutoTrackable):
    def __setattr__(self, name, value):
        """Support self.foo = trackable syntax."""
        try:
            if getattr(self, name) is value:
                # Short circuit for `self.$x = self.$x`.
                return
        except AttributeError:
            pass

        if getattr(self, "_self_setattr_tracking", True):
            value = sticky_attribute_assignment(
                trackable=self, value=value, name=name)
        super().__setattr__(name, value)

    # def _no_dependency(self, value):
    #     """Override to allow TrackableBase to disable dependency tracking."""
    #     with tracking.DotNotTrackScope():
    #         return value


def sticky_attribute_assignment(trackable, name, value):
    if isinstance(value, tracking.TrackedList):
        value = list(value)
    if isinstance(value, tracking.TrackedDict):
        value = dict(value)
    if isinstance(value, tracking.TrackedSet):
        value = set(value)
    value = tf.__internal__.tracking.wrap(value)
    if not tracking.is_tracking_enabled():
        return value
    if isinstance(value, tf.__internal__.tracking.Trackable):
        trackable._track_trackable(  # pylint: disable=protected-access
            value, name=name,
            # Allow the user to switch the Trackable which is tracked by this
            # name, since assigning a new variable to an attribute has
            # historically been fine (e.g. Adam did this).
            overwrite=True)
    return value