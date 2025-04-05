class baseconsole(object):
    def __init__(self):
        pass

    def bell(self):
        raise NotImplementedError

    def pos(self, x=None, y=None):
        """Move or query the window cursor."""
        raise NotImplementedError

    def size(self):
        raise NotImplementedError

    def rectangle(self, rect, attr=None, fill=" "):
        """Fill Rectangle."""
        raise NotImplementedError

    def write_scrolling(self, text, attr=None):
        """write text at current cursor position while watching for scrolling.

        If the window scrolls because you are at the bottom of the screen
        buffer, all positions that you are storing will be shifted by the
        scroll amount. For example, I remember the cursor position of the
        prompt so that I can redraw the line but if the window scrolls,
        the remembered position is off.

        This variant of write tries to keep track of the cursor position
        so that it will know when the screen buffer is scrolled. It
        returns the number of lines that the buffer scrolled.

        """
        raise NotImplementedError

    def getkeypress(self):
        """Return next key press event from the queue, ignoring others."""
        raise NotImplementedError

    def write(self, text):
        raise NotImplementedError

    def page(self, attr=None, fill=" "):
        """Fill the entire screen."""
        raise NotImplementedError

    def isatty(self):
        return True

    def flush(self):
        pass
