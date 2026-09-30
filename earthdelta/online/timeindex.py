"""Calendar arithmetic for the authorized six-hour 2020 grid."""
from datetime import datetime, timedelta, timezone
from numbers import Integral

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
STEPS = 1464


def checked_index(index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < STEPS:
        raise ValueError('index outside the 2020 grid')
    return int(index)


def index_of(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError('timezone-aware timestamp required')
    seconds = (value.astimezone(timezone.utc) - START).total_seconds()
    if seconds % 21600:
        raise ValueError('timestamp is not six-hour aligned')
    return checked_index(int(seconds // 21600))


def iso_of(index):
    return (START + timedelta(hours=6 * checked_index(index))).isoformat().replace('+00:00', 'Z')


def daily_issues(first, last):
    a, b = index_of(first), index_of(last)
    if a > b or a % 4 or b % 4:
        raise ValueError('ordered daily 00Z endpoints required')
    return tuple(iso_of(i) for i in range(a, b + 1, 4))


def calendar_pairs(indices, days):
    if isinstance(days, bool) or not isinstance(days, Integral) or days <= 0:
        raise ValueError('positive integer calendar lag required')
    values = [checked_index(i) for i in indices]
    if values != sorted(set(values)):
        raise ValueError('unique chronological indices required')
    positions = {v: i for i, v in enumerate(values)}
    return tuple((positions[v - 4 * days], i) for i, v in enumerate(values) if v - 4 * days in positions)
