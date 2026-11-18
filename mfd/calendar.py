"""Toronto Stock Exchange session calendar and decision-cutoff helpers.

The holiday rules below reproduce the TSX's published statutory-holiday schedule
(New Year's Day, Family Day, Good Friday, Victoria Day, Canada Day, Civic Holiday,
Labour Day, Thanksgiving, Christmas, Boxing Day, with weekend substitution).
One-off closures (e.g. the 2022 national day of mourning) are not encoded; when
real price data is loaded, observed sessions in the data take precedence.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, GoodFriday, MO, next_monday,
    next_monday_or_tuesday, DateOffset,
)

from .config import MARKET_TZ, DECISION_CUTOFF_HOUR



class TSXHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=next_monday),
        Holiday("Family Day", month=2, day=1, offset=DateOffset(weekday=MO(3)),
                start_date=dt.datetime(2008, 1, 1)),
        GoodFriday,
        # Victoria Day: Monday preceding May 25
        Holiday("Victoria Day", month=5, day=24, offset=DateOffset(weekday=MO(-1))),
        Holiday("Canada Day", month=7, day=1, observance=next_monday),
        Holiday("Civic Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=10, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday("Christmas", month=12, day=25, observance=next_monday),
        Holiday("Boxing Day", month=12, day=26, observance=next_monday_or_tuesday),
    ]


def tsx_sessions(start, end) -> pd.DatetimeIndex:
    """Business days in [start, end] excluding TSX statutory holidays (naive dates)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    hol = TSXHolidayCalendar().holidays(start - pd.Timedelta(days=7), end + pd.Timedelta(days=7))
    bdays = pd.bdate_range(start, end)
    return bdays[~bdays.isin(hol)]


def last_session_of_month(month: pd.Period, sessions: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    s = sessions if sessions is not None else tsx_sessions(month.start_time, month.end_time)
    s = s[(s >= month.start_time.normalize()) & (s <= month.end_time.normalize())]
    return s.max()


def first_session_after(ts: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First session strictly after the date of `ts` (sessions are naive dates)."""
    d = pd.Timestamp(ts).tz_localize(None).normalize() if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).normalize()
    later = sessions[sessions > d]
    return later.min() if len(later) else None


def decision_cutoff(month: pd.Period, sessions: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    """Tz-aware decision timestamp for a month: 17:00 Toronto on its last session."""
    last = last_session_of_month(month, sessions)
    return pd.Timestamp(last).tz_localize(None).replace(hour=DECISION_CUTOFF_HOUR).tz_localize(MARKET_TZ)


def now_toronto() -> pd.Timestamp:
    return pd.Timestamp.now(tz=MARKET_TZ)


def latest_completed_month(asof: pd.Timestamp | None = None) -> pd.Period:
    """The most recent month whose decision cutoff has passed."""
    asof = asof if asof is not None else now_toronto()
    if asof.tzinfo is None:
        asof = asof.tz_localize(MARKET_TZ)
    m = pd.Period(asof.tz_convert(MARKET_TZ).tz_localize(None), freq="M")
    if decision_cutoff(m) <= asof:
        return m
    return m - 1
