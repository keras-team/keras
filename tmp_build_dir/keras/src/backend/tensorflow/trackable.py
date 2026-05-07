import tensorflow as tf

from keras.src.utils import tracking


class KerasAutoTrackable(tf.__internal__.tracking.AutoTrackable):
    """Manages dependencies on other objects with Keras tracking.

    Similar to TF AutoTrackable, but disabling tracking is based
    on tracking within Keras.

    This serves as an interface between Keras tracking and TF tracking.
    """

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
                trackable=self, value=value, name=name
            )
        super().__setattr__(name, value)


def sticky_attribute_assignment(trackable, name, value):
    """Adds dependencies, called from __setattr__.

    Args:
        trackable: The object to add dependencies to (generally the one having
        an attribute assigned).
        name: The attribute name being assigned.
        value: The value being assigned. Not necessarily a trackable object.

    Returns:
        The value which should be stored in the attribute.
    """
    if isinstance(
        value, (tracking.TrackedList, tracking.TrackedDict, tracking.TrackedSet)
    ) and hasattr(trackable, "_tracked"):
        trackable._tracked.append(name)
    if not tracking.is_tracking_enabled():
        return value
    if isinstance(value, tf.__internal__.tracking.Trackable):
        trackable._track_trackable(  # pylint: disable=protected-access
            value,
            name=name,
            # Allow the user to switch the Trackable which is tracked by this
            # name, since assigning a new variable to an attribute has
            # historically been fine (e.g. Adam did this).
            overwrite=True,
        )
    return value
