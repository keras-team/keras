# -*- coding: ISO-8859-1 -*-


import os
import re
import sys

terminal_escape = re.compile("(\001?\033\\[[0-9;]*m\002?)")
escape_parts = re.compile("\001?\033\\[([0-9;]*)m\002?")


class AnsiState(object):
    def __init__(
        self,
        bold=False,
        inverse=False,
        color="white",
        background="black",
        backgroundbold=False,
    ):
        self.bold = bold
        self.inverse = inverse
        self.color = color
        self.background = background
        self.backgroundbold = backgroundbold

    trtable = {
        "black": 0,
        "red": 4,
        "green": 2,
        "yellow": 6,
        "blue": 1,
        "magenta": 5,
        "cyan": 3,
        "white": 7,
    }
    revtable = dict(zip(trtable.values(), trtable.keys()))

    def get_winattr(self):
        attr = 0
        if self.bold:
            attr |= 0x0008
        if self.backgroundbold:
            attr |= 0x0080
        if self.inverse:
            attr |= 0x4000
        attr |= self.trtable[self.color]
        attr |= self.trtable[self.background] << 4
        return attr

    def set_winattr(self, attr):
        self.bold = bool(attr & 0x0008)
        self.backgroundbold = bool(attr & 0x0080)
        self.inverse = bool(attr & 0x4000)
        self.color = self.revtable[attr & 0x0007]
        self.background = self.revtable[(attr & 0x0070) >> 4]

    winattr = property(get_winattr, set_winattr)

    def __repr__(self):
        return (
            "AnsiState(bold=%s,inverse=%s,color=%9s,"
            "background=%9s,backgroundbold=%s)# 0x%x"
            % (
                self.bold,
                self.inverse,
                '"%s"' % self.color,
                '"%s"' % self.background,
                self.backgroundbold,
                self.winattr,
            )
        )

    def copy(self):
        x = AnsiState()
        x.bold = self.bold
        x.inverse = self.inverse
        x.color = self.color
        x.background = self.background
        x.backgroundbold = self.backgroundbold
        return x


defaultstate = AnsiState(False, False, "white")

trtable = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "white",
}


class AnsiWriter(object):
    def __init__(self, default=defaultstate):
        if isinstance(defaultstate, AnsiState):
            self.defaultstate = default
        else:
            self.defaultstate = AnsiState()
            self.defaultstate.winattr = defaultstate

    def write_color(self, text, attr=None):
        """write text at current cursor position and interpret color escapes.

        return the number of characters written.
        """
        if isinstance(attr, AnsiState):
            defaultstate = attr
        elif attr is None:  # use attribute form initial console
            attr = self.defaultstate.copy()
        else:
            defaultstate = AnsiState()
            defaultstate.winattr = attr
            attr = defaultstate
        chunks = terminal_escape.split(text)
        n = 0  # count the characters we actually write, omitting the escapes
        res = []
        for chunk in chunks:
            m = escape_parts.match(chunk)
            if m:
                parts = m.group(1).split(";")
                if len(parts) == 1 and parts[0] == "0":
                    attr = self.defaultstate.copy()
                    continue
                for part in parts:
                    if part == "0":  # No text attribute
                        attr = self.defaultstate.copy()
                        attr.bold = False
                    elif part == "7":  # switch on reverse
                        attr.inverse = True
                    # switch on bold (i.e. intensify foreground color)
                    elif part == "1":
                        attr.bold = True
                    elif len(part) == 2:
                        if part == "22":  # Normal foreground color
                            attr.bold = False
                        elif "30" <= part <= "37":  # set foreground color
                            attr.color = trtable[int(part) - 30]
                        elif part == "39":  # Default foreground color
                            attr.color = self.defaultstate.color
                        elif "40" <= part <= "47":  # set background color
                            attr.background = trtable[int(part) - 40]
                        elif part == "49":  # Default background color
                            attr.background = self.defaultstate.background
                continue
            n += len(chunk)

            res.append((attr.copy(), chunk))

        return n, res

    def parse_color(self, text, attr=None):
        n, res = self.write_color(text, attr)
        return n, [attr.winattr for attr, text in res]


def write_color(text, attr=None):
    a = AnsiWriter(defaultstate)
    return a.write_color(text, attr)


def write_color_old(text, attr=None):
    """write text at current cursor position and interpret color escapes.

    return the number of characters written.
    """
    res = []
    chunks = terminal_escape.split(text)
    n = 0  # count the characters we actually write, omitting the escapes
    if attr is None:  # use attribute from initial console
        attr = 15
    for chunk in chunks:
        m = escape_parts.match(chunk)
        if m:
            for part in m.group(1).split(";"):
                if part == "0":  # No text attribute
                    attr = 0
                elif part == "7":  # switch on reverse
                    attr |= 0x4000
                # switch on bold (i.e. intensify foreground color)
                if part == "1":
                    attr |= 0x08
                elif len(part) == 2 and "30" <= part <= "37":  # set foreground color
                    part = int(part) - 30
                    # we have to mirror bits
                    attr = (
                        (attr & ~0x07)
                        | ((part & 0x1) << 2)
                        | (part & 0x2)
                        | ((part & 0x4) >> 2)
                    )
                elif len(part) == 2 and "40" <= part <= "47":  # set background color
                    part = int(part) - 40
                    # we have to mirror bits
                    attr = (
                        (attr & ~0x70)
                        | ((part & 0x1) << 6)
                        | ((part & 0x2) << 4)
                        | ((part & 0x4) << 2)
                    )
                # ignore blink, underline and anything we don't understand
            continue
        n += len(chunk)
        if chunk:
            res.append(("0x%x" % attr, chunk))
    return res


# trtable={0:"black",1:"red",2:"green",3:"yellow",4:"blue",5:"magenta",6:"cyan",7:"white"}

if __name__ == "__main__x":
    import pprint

    pprint = pprint.pprint

    s = "\033[0;31mred\033[0;32mgreen\033[0;33myellow\033[0;34mblue\033[0;35mmagenta\033[0;36mcyan\033[0;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))
    s = "\033[1;31mred\033[1;32mgreen\033[1;33myellow\033[1;34mblue\033[1;35mmagenta\033[1;36mcyan\033[1;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))

    s = "\033[0;7;31mred\033[0;7;32mgreen\033[0;7;33myellow\033[0;7;34mblue\033[0;7;35mmagenta\033[0;7;36mcyan\033[0;7;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))
    s = "\033[1;7;31mred\033[1;7;32mgreen\033[1;7;33myellow\033[1;7;34mblue\033[1;7;35mmagenta\033[1;7;36mcyan\033[1;7;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))


if __name__ == "__main__":
    import pprint

    import console

    pprint = pprint.pprint

    c = console.Console()
    c.write_color("dhsjdhs")
    c.write_color("\033[0;32mIn [\033[1;32m1\033[0;32m]:")
    print
    pprint(write_color("\033[0;32mIn [\033[1;32m1\033[0;32m]:"))

if __name__ == "__main__x":
    import pprint

    pprint = pprint.pprint
    s = "\033[0;31mred\033[0;32mgreen\033[0;33myellow\033[0;34mblue\033[0;35mmagenta\033[0;36mcyan\033[0;37mwhite\033[0m"
    pprint(write_color(s))
