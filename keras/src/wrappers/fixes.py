try:
    import sklearn
except ImportError:
    sklearn = None


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
    def _raise_or_return(target_type):
        """Depending on the value of raise_unknown, either raise an error or
        return 'unknown'.
        """
        if raise_unknown and target_type == "unknown":
            input = input_name if input_name else "data"
            raise ValueError(f"Unknown label type for {input}: {y!r}")
        else:
            return target_type

    from sklearn.utils.multiclass import type_of_target as sk_type_of_target

    target_type = sk_type_of_target(y, input_name=input_name)
    return _raise_or_return(target_type)


def _routing_enabled():
    """Return whether metadata routing is enabled.

    Returns:
        enabled : bool
            Whether metadata routing is enabled. If the config is not set, it
            defaults to False.

    TODO: remove when the config key is no longer available in scikit-learn
    """
    return sklearn.get_config().get("enable_metadata_routing", False)


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
