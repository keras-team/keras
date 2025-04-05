# The inspect.formatargspec() function was dropped in Python 3.11 but we need
# need it for when constructing signature changing decorators based on result of
# inspect.getargspec() or inspect.getfullargspec(). The code here implements
# inspect.formatargspec() base on Parameter and Signature from inspect module,
# which were added in Python 3.6. Thanks to Cyril Jouve for the implementation.

try:
    from inspect import Parameter, Signature
except ImportError:
    from inspect import formatargspec
else:
    def formatargspec(args, varargs=None, varkw=None, defaults=None,
                      kwonlyargs=(), kwonlydefaults={}, annotations={}):
        if kwonlydefaults is None:
            kwonlydefaults = {}
        ndefaults = len(defaults) if defaults else 0
        parameters = [
            Parameter(
                arg,
                Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults[i] if i >= 0 else Parameter.empty,
                annotation=annotations.get(arg, Parameter.empty),
            ) for i, arg in enumerate(args, ndefaults - len(args))
        ]
        if varargs:
            parameters.append(Parameter(varargs, Parameter.VAR_POSITIONAL))
        parameters.extend(
            Parameter(
                kwonlyarg,
                Parameter.KEYWORD_ONLY,
                default=kwonlydefaults.get(kwonlyarg, Parameter.empty),
                annotation=annotations.get(kwonlyarg, Parameter.empty),
            ) for kwonlyarg in kwonlyargs
        )
        if varkw:
            parameters.append(Parameter(varkw, Parameter.VAR_KEYWORD))
        return_annotation = annotations.get('return', Signature.empty)
        return str(Signature(parameters, return_annotation=return_annotation))