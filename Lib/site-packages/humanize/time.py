"""Time humanizing functions.

These are largely borrowed from Django's `contrib.humanize`.
"""

from __future__ import annotations

from enum import Enum
from functools import total_ordering

from .i18n import _gettext as _
from .i18n import _ngettext
from .number import intcomma

TYPE_CHECKING = False
if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterable
    from typing import Any

__all__ = [
    "naturaldate",
    "naturalday",
    "naturaldelta",
    "naturaltime",
    "precisedelta",
]


@total_ordering
class Unit(Enum):
    MICROSECONDS = 0
    MILLISECONDS = 1
    SECONDS = 2
    MINUTES = 3
    HOURS = 4
    DAYS = 5
    MONTHS = 6
    YEARS = 7

    def __lt__(self, other: Any) -> Any:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


def _now() -> dt.datetime:
    import datetime as dt

    return dt.datetime.now()


def _abs_timedelta(delta: dt.timedelta) -> dt.timedelta:
    """Return an "absolute" value for a timedelta, always representing a time distance.

    Args:
        delta (datetime.timedelta): Input timedelta.

    Returns:
        datetime.timedelta: Absolute timedelta.
    """
    if delta.days < 0:
        now = _now()
        return now - (now + delta)
    return delta


def _date_and_delta(value: Any, *, now: dt.datetime | None = None) -> tuple[Any, Any]:
    """Turn a value into a date and a timedelta which represents how long ago it was.

    If that's not possible, return `(None, value)`.
    """
    import datetime as dt

    if not now:
        now = _now()
    if isinstance(value, dt.datetime):
        date = value
        delta = now - value
    elif isinstance(value, dt.timedelta):
        date = now - value
        delta = value
    else:
        try:
            value = int(value)
            delta = dt.timedelta(seconds=value)
            date = now - delta
        except (ValueError, TypeError):
            return None, value
    return date, _abs_timedelta(delta)


def naturaldelta(
    value: dt.timedelta | float,
    months: bool = True,
    minimum_unit: str = "seconds",
) -> str:
    """Return a natural representation of a timedelta or number of seconds.

    This is similar to `naturaltime`, but does not add tense to the result.

    Args:
        value (datetime.timedelta, int or float): A timedelta or a number of seconds.
        months (bool): If `True`, then a number of months (based on 30.5 days) will be
            used for fuzziness between years.
        minimum_unit (str): The lowest unit that can be used.

    Returns:
        str (str or `value`): A natural representation of the amount of time
            elapsed unless `value` is not datetime.timedelta or cannot be
            converted to int (cannot be float due to 'inf' or 'nan').
            In that case, a `value` is returned unchanged.

    Raises:
        OverflowError: If `value` is too large to convert to datetime.timedelta.

    Examples:
        Compare two timestamps in a custom local timezone::

        ```pycon
        >>> import datetime as dt
        >>> from dateutil.tz import gettz

        >>> berlin = gettz("Europe/Berlin")
        >>> now = dt.datetime.now(tz=berlin)
        >>> later = now + dt.timedelta(minutes=30)

        >>> assert naturaldelta(later - now) == "30 minutes"
        True
        ```

    """
    import datetime as dt

    tmp = Unit[minimum_unit.upper()]
    if tmp not in (Unit.SECONDS, Unit.MILLISECONDS, Unit.MICROSECONDS):
        msg = f"Minimum unit '{minimum_unit}' not supported"
        raise ValueError(msg)
    min_unit = tmp

    if isinstance(value, dt.timedelta):
        delta = value
    else:
        try:
            int(value)  # Explicitly don't support string such as "NaN" or "inf"
            value = float(value)
            delta = dt.timedelta(seconds=value)
        except (ValueError, TypeError):
            return str(value)

    use_months = months

    delta = abs(delta)
    years = delta.days // 365
    days = delta.days % 365
    num_months = int(days // 30.5)

    if not years and days < 1:
        if delta.seconds == 0:
            if min_unit == Unit.MICROSECONDS and delta.microseconds < 1000:
                return (
                    _ngettext("%d microsecond", "%d microseconds", delta.microseconds)
                    % delta.microseconds
                )

            if min_unit == Unit.MILLISECONDS or (
                min_unit == Unit.MICROSECONDS and 1000 <= delta.microseconds < 1_000_000
            ):
                milliseconds = delta.microseconds / 1000
                return (
                    _ngettext("%d millisecond", "%d milliseconds", int(milliseconds))
                    % milliseconds
                )
            return _("a moment")

        if delta.seconds == 1:
            return _("a second")

        if delta.seconds < 60:
            return _ngettext("%d second", "%d seconds", delta.seconds) % delta.seconds

        if 60 <= delta.seconds < 120:
            return _("a minute")

        if 120 <= delta.seconds < 3600:
            minutes = delta.seconds // 60
            return _ngettext("%d minute", "%d minutes", minutes) % minutes

        if 3600 <= delta.seconds < 3600 * 2:
            return _("an hour")

        if 3600 < delta.seconds:
            hours = delta.seconds // 3600
            return _ngettext("%d hour", "%d hours", hours) % hours

    elif years == 0:
        if days == 1:
            return _("a day")

        if not use_months:
            return _ngettext("%d day", "%d days", days) % days

        if not num_months:
            return _ngettext("%d day", "%d days", days) % days

        if num_months == 1:
            return _("a month")

        return _ngettext("%d month", "%d months", num_months) % num_months

    elif years == 1:
        if not num_months and not days:
            return _("a year")

        if not num_months:
            return _ngettext("1 year, %d day", "1 year, %d days", days) % days

        if use_months:
            if num_months == 1:
                return _("1 year, 1 month")

            return (
                _ngettext("1 year, %d month", "1 year, %d months", num_months)
                % num_months
            )

        return _ngettext("1 year, %d day", "1 year, %d days", days) % days

    return _ngettext("%d year", "%d years", years).replace("%d", "%s") % intcomma(years)


def naturaltime(
    value: dt.datetime | dt.timedelta | float,
    future: bool = False,
    months: bool = True,
    minimum_unit: str = "seconds",
    when: dt.datetime | None = None,
) -> str:
    """Return a natural representation of a time in a resolution that makes sense.

    This is more or less compatible with Django's `naturaltime` filter.

    Args:
        value (datetime.datetime, datetime.timedelta, int or float): A `datetime`, a
            `timedelta`, or a number of seconds.
        future (bool): Ignored for `datetime`s and `timedelta`s, where the tense is
            always figured out based on the current time. For integers and floats, the
            return value will be past tense by default, unless future is `True`.
        months (bool): If `True`, then a number of months (based on 30.5 days) will be
            used for fuzziness between years.
        minimum_unit (str): The lowest unit that can be used.
        when (datetime.datetime): Point in time relative to which _value_ is
            interpreted.  Defaults to the current time in the local timezone.

    Returns:
        str: A natural representation of the input in a resolution that makes sense.
    """
    import datetime as dt

    value = _convert_aware_datetime(value)
    when = _convert_aware_datetime(when)

    now = when or _now()

    date, delta = _date_and_delta(value, now=now)
    if date is None:
        return str(value)
    # determine tense by value only if datetime/timedelta were passed
    if isinstance(value, (dt.datetime, dt.timedelta)):
        future = date > now

    ago = _("%s from now") if future else _("%s ago")
    delta = naturaldelta(delta, months, minimum_unit)

    if delta == _("a moment"):
        return _("now")

    return str(ago % delta)


def _convert_aware_datetime(
    value: dt.datetime | dt.timedelta | float | None,
) -> Any:
    """Convert aware datetime to naive datetime and pass through any other type."""
    import datetime as dt

    if isinstance(value, dt.datetime) and value.tzinfo is not None:
        value = dt.datetime.fromtimestamp(value.timestamp())
    return value


def naturalday(value: dt.date | dt.datetime, format: str = "%b %d") -> str:
    """Return a natural day.

    For date values that are tomorrow, today or yesterday compared to
    present day return representing string. Otherwise, return a string
    formatted according to `format`.

    """
    import datetime as dt

    try:
        value = dt.date(value.year, value.month, value.day)
    except AttributeError:
        # Passed value wasn't date-ish
        return str(value)
    except (OverflowError, ValueError):
        # Date arguments out of range
        return str(value)
    delta = value - dt.date.today()

    if delta.days == 0:
        return _("today")

    if delta.days == 1:
        return _("tomorrow")

    if delta.days == -1:
        return _("yesterday")

    return value.strftime(format)


def naturaldate(value: dt.date | dt.datetime) -> str:
    """Like `naturalday`, but append a year for dates more than ~five months away."""
    import datetime as dt

    try:
        value = dt.date(value.year, value.month, value.day)
    except AttributeError:
        # Passed value wasn't date-ish
        return str(value)
    except (OverflowError, ValueError):
        # Date arguments out of range
        return str(value)
    delta = _abs_timedelta(value - dt.date.today())
    if delta.days >= 5 * 365 / 12:
        return naturalday(value, "%b %d %Y")
    return naturalday(value)


def _quotient_and_remainder(
    value: float,
    divisor: float,
    unit: Unit,
    minimum_unit: Unit,
    suppress: Iterable[Unit],
) -> tuple[float, float]:
    """Divide `value` by `divisor` returning the quotient and remainder.

    If `unit` is `minimum_unit`, makes the quotient a float number and the remainder
    will be zero. The rational is that if `unit` is the unit of the quotient, we cannot
    represent the remainder because it would require a unit smaller than the
    `minimum_unit`.

    >>> from humanize.time import _quotient_and_remainder, Unit
    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.DAYS, [])
    (1.5, 0)

    If unit is in `suppress`, the quotient will be zero and the remainder will be the
    initial value. The idea is that if we cannot use `unit`, we are forced to use a
    lower unit so we cannot do the division.

    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS])
    (0, 36)

    In other case return quotient and remainder as `divmod` would do it.

    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [])
    (1, 12)

    """
    if unit == minimum_unit:
        return value / divisor, 0

    if unit in suppress:
        return 0, value

    return divmod(value, divisor)


def _carry(
    value1: float,
    value2: float,
    ratio: float,
    unit: Unit,
    min_unit: Unit,
    suppress: Iterable[Unit],
) -> tuple[float, float]:
    """Return a tuple with two values.

    If the unit is in `suppress`, multiply `value1` by `ratio` and add it to `value2`
    (carry to right). The idea is that if we cannot represent `value1` we need to
    represent it in a lower unit.

    >>> from humanize.time import _carry, Unit
    >>> _carry(2, 6, 24, Unit.DAYS, Unit.SECONDS, [Unit.DAYS])
    (0, 54)

    If the unit is the minimum unit, `value2` is divided by `ratio` and added to
    `value1` (carry to left). We assume that `value2` has a lower unit so we need to
    carry it to `value1`.

    >>> _carry(2, 6, 24, Unit.DAYS, Unit.DAYS, [])
    (2.25, 0)

    Otherwise, just return the same input:

    >>> _carry(2, 6, 24, Unit.DAYS, Unit.SECONDS, [])
    (2, 6)
    """
    if unit == min_unit:
        return value1 + value2 / ratio, 0

    if unit in suppress:
        return 0, value2 + value1 * ratio

    return value1, value2


def _suitable_minimum_unit(min_unit: Unit, suppress: Iterable[Unit]) -> Unit:
    """Return a minimum unit suitable that is not suppressed.

    If not suppressed, return the same unit:

    >>> from humanize.time import _suitable_minimum_unit, Unit
    >>> _suitable_minimum_unit(Unit.HOURS, []).name
    'HOURS'

    But if suppressed, find a unit greater than the original one that is not
    suppressed:

    >>> _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]).name
    'DAYS'

    >>> _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS]).name
    'MONTHS'
    """
    if min_unit in suppress:
        for unit in Unit:
            if unit > min_unit and unit not in suppress:
                return unit

        msg = "Minimum unit is suppressed and no suitable replacement was found"
        raise ValueError(msg)

    return min_unit


def _suppress_lower_units(min_unit: Unit, suppress: Iterable[Unit]) -> set[Unit]:
    """Extend suppressed units (if any) with all units lower than the minimum unit.

    >>> from humanize.time import _suppress_lower_units, Unit
    >>> [x.name for x in sorted(_suppress_lower_units(Unit.SECONDS, [Unit.DAYS]))]
    ['MICROSECONDS', 'MILLISECONDS', 'DAYS']
    """
    suppress = set(suppress)
    for unit in Unit:
        if unit == min_unit:
            break
        suppress.add(unit)

    return suppress


def precisedelta(
    value: dt.timedelta | int | None,
    minimum_unit: str = "seconds",
    suppress: Iterable[str] = (),
    format: str = "%0.2f",
) -> str:
    """Return a precise representation of a timedelta.

    ```pycon
    >>> import datetime as dt
    >>> from humanize.time import precisedelta

    >>> delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    >>> precisedelta(delta)
    '2 days, 1 hour and 33.12 seconds'

    ```

    A custom `format` can be specified to control how the fractional part
    is represented:

    ```pycon
    >>> precisedelta(delta, format="%0.4f")
    '2 days, 1 hour and 33.1230 seconds'

    ```

    Instead, the `minimum_unit` can be changed to have a better resolution;
    the function will still readjust the unit to use the greatest of the
    units that does not lose precision.

    For example setting microseconds but still representing the date with milliseconds:

    ```pycon
    >>> precisedelta(delta, minimum_unit="microseconds")
    '2 days, 1 hour, 33 seconds and 123 milliseconds'

    ```

    If desired, some units can be suppressed: you will not see them represented and the
    time of the other units will be adjusted to keep representing the same timedelta:

    ```pycon
    >>> precisedelta(delta, suppress=['days'])
    '49 hours and 33.12 seconds'

    ```

    Note that microseconds precision is lost if the seconds and all
    the units below are suppressed:

    ```pycon
    >>> delta = dt.timedelta(seconds=90, microseconds=100)
    >>> precisedelta(delta, suppress=['seconds', 'milliseconds', 'microseconds'])
    '1.50 minutes'

    ```

    If the delta is too small to be represented with the minimum unit,
    a value of zero will be returned:

    ```pycon
    >>> delta = dt.timedelta(seconds=1)
    >>> precisedelta(delta, minimum_unit="minutes")
    '0.02 minutes'

    >>> delta = dt.timedelta(seconds=0.1)
    >>> precisedelta(delta, minimum_unit="minutes")
    '0 minutes'

    ```
    """
    date, delta = _date_and_delta(value)
    if date is None:
        return str(value)

    suppress_set = {Unit[s.upper()] for s in suppress}

    # Find a suitable minimum unit (it can be greater the one that the
    # user gave us if it is suppressed).
    min_unit = Unit[minimum_unit.upper()]
    min_unit = _suitable_minimum_unit(min_unit, suppress_set)
    del minimum_unit

    # Expand the suppressed units list/set to include all the units
    # that are below the minimum unit
    suppress_set = _suppress_lower_units(min_unit, suppress_set)

    # handy aliases
    days = delta.days
    secs = delta.seconds
    usecs = delta.microseconds

    MICROSECONDS, MILLISECONDS, SECONDS, MINUTES, HOURS, DAYS, MONTHS, YEARS = list(
        Unit
    )

    # Given DAYS compute YEARS and the remainder of DAYS as follows:
    #   if YEARS is the minimum unit, we cannot use DAYS so
    #   we will use a float for YEARS and 0 for DAYS:
    #       years, days = years/days, 0
    #
    #   if YEARS is suppressed, use DAYS:
    #       years, days = 0, days
    #
    #   otherwise:
    #       years, days = divmod(years, days)
    #
    # The same applies for months, hours, minutes and milliseconds below
    years, days = _quotient_and_remainder(days, 365, YEARS, min_unit, suppress_set)
    months, days = _quotient_and_remainder(days, 30.5, MONTHS, min_unit, suppress_set)

    # If DAYS is not in suppress, we can represent the days but
    # if it is a suppressed unit, we need to carry it to a lower unit,
    # seconds in this case.
    #
    # The same applies for secs and usecs below
    days, secs = _carry(days, secs, 24 * 3600, DAYS, min_unit, suppress_set)

    hours, secs = _quotient_and_remainder(secs, 3600, HOURS, min_unit, suppress_set)
    minutes, secs = _quotient_and_remainder(secs, 60, MINUTES, min_unit, suppress_set)

    secs, usecs = _carry(secs, usecs, 1e6, SECONDS, min_unit, suppress_set)

    msecs, usecs = _quotient_and_remainder(
        usecs, 1000, MILLISECONDS, min_unit, suppress_set
    )

    # if _unused != 0 we had lost some precision
    usecs, _unused = _carry(usecs, 0, 1, MICROSECONDS, min_unit, suppress_set)

    fmts = [
        ("%d year", "%d years", years),
        ("%d month", "%d months", months),
        ("%d day", "%d days", days),
        ("%d hour", "%d hours", hours),
        ("%d minute", "%d minutes", minutes),
        ("%d second", "%d seconds", secs),
        ("%d millisecond", "%d milliseconds", msecs),
        ("%d microsecond", "%d microseconds", usecs),
    ]

    texts: list[str] = []
    for unit, fmt in zip(reversed(Unit), fmts):
        singular_txt, plural_txt, fmt_value = fmt
        if fmt_value > 0 or (not texts and unit == min_unit):
            _fmt_value = 2 if 1 < fmt_value < 2 else int(fmt_value)
            fmt_txt = _ngettext(singular_txt, plural_txt, _fmt_value)
            import math

            if unit == min_unit and math.modf(fmt_value)[0] > 0:
                fmt_txt = fmt_txt.replace("%d", format)
            elif unit == YEARS:
                fmt_txt = fmt_txt.replace("%d", "%s")
                texts.append(fmt_txt % intcomma(fmt_value))
                continue

            texts.append(fmt_txt % fmt_value)

        if unit == min_unit:
            break

    if len(texts) == 1:
        return texts[0]

    head = ", ".join(texts[:-1])
    tail = texts[-1]

    return _("%s and %s") % (head, tail)
