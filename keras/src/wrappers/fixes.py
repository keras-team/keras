import sklearn
from packaging.version import parse as parse_version
from sklearn import get_config

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version < parse_version("1.6"):

    def patched_more_tags(estimator, expected_failed_checks):
        import copy

        from sklearn.utils._tags import _safe_tags

        original_tags = copy.deepcopy(_safe_tags(estimator))

        def patched_more_tags(self):
            original_tags.update({"_xfail_checks": expected_failed_checks})
            return original_tags

        estimator.__class__._more_tags = patched_more_tags
        return estimator

    def parametrize_with_checks(
        estimators,
        *,
        legacy: bool = True,
        expected_failed_checks=None,
    ):
        # legacy is not supported and ignored
        from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: F401, I001

        estimators = [
            patched_more_tags(estimator, expected_failed_checks(estimator))
            for estimator in estimators
        ]

        return parametrize_with_checks(estimators)
else:
    from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: F401, I001


def _validate_data(estimator, *args, **kwargs):
    """Validate the input data.

    wrapper for sklearn.utils.validation.validate_data or
    BaseEstimator._validate_data depending on the scikit-learn version.

    TODO: remove when minimum scikit-learn version is 1.6
    """
    try:
        # scikit-learn >= 1.6
        from sklearn.utils.validation import validate_data

        return validate_data(estimator, *args, **kwargs)
    except ImportError:
        return estimator._validate_data(*args, **kwargs)
    except:
        raise


def type_of_target(y, input_name="", *, raise_unknown=False):
    # fix for raise_unknown which is introduced in scikit-learn 1.6
    from sklearn.utils.multiclass import type_of_target

    def _raise_or_return(target_type):
        """Depending on the value of raise_unknown, either raise an error or
        return 'unknown'.
        """
        if raise_unknown and target_type == "unknown":
            input = input_name if input_name else "data"
            raise ValueError(f"Unknown label type for {input}: {y!r}")
        else:
            return target_type

    target_type = type_of_target(y, input_name=input_name)
    return _raise_or_return(target_type)


def _routing_enabled():
    """Return whether metadata routing is enabled.

    Returns:
        enabled : bool
            Whether metadata routing is enabled. If the config is not set, it
            defaults to False.

    TODO: remove when the config key is no longer available in scikit-learn
    """
    return get_config().get("enable_metadata_routing", False)


def _raise_for_params(params, owner, method):
    """Raise an error if metadata routing is not enabled and params are passed.

    Parameters:
        params : dict
            The metadata passed to a method.
        owner : object
            The object to which the method belongs.
        method : str
            The name of the method, e.g. "fit".

    Raises:
        ValueError
            If metadata routing is not enabled and params are passed.
    """
    caller = (
        f"{owner.__class__.__name__}.{method}"
        if method
        else owner.__class__.__name__
    )
    if not _routing_enabled() and params:
        raise ValueError(
            f"Passing extra keyword arguments to {caller} is only supported if"
            " enable_metadata_routing=True, which you can set using"
            " `sklearn.set_config`. See the User Guide"
            " <https://scikit-learn.org/stable/metadata_routing.html> for more"
            f" details. Extra parameters passed are: {set(params)}"
        )
