import sys
import textwrap

from .py3k_compat import is_callable

rlmain = sys.modules["readline"]
rl = rlmain.rl


def get_doc(rl_):
    methods = [(x, getattr(rl_, x)) for x in dir(rl_) if is_callable(getattr(rl_, x))]
    return [(x, m.__doc__) for x, m in methods if m.__doc__]


def get_rest(rl_):
    q = get_doc(rl_)
    out = []
    for funcname, doc in q:
        out.append(funcname)
        out.append("\n".join(textwrap.wrap(doc, 80, initial_indent="   ")))
        out.append("")
    return out
