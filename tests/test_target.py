"""Target definition: calendar-day comparison, missing delivery is never 'on time'."""

from __future__ import annotations


def get(wh, order_id, cols):
    return wh.execute(f"SELECT {cols} FROM mart_orders WHERE order_id = ?", [order_id]).fetchone()


def test_delivery_on_promised_day_is_on_time(wh):
    is_late, delay, outcome = get(wh, "o_sameday", "is_late, delay_days, outcome")
    assert is_late is False and delay == 0 and outcome == "on_time"


def test_delivery_next_day_is_late(wh):
    is_late, delay, outcome = get(wh, "o_nextday", "is_late, delay_days, outcome")
    assert is_late is True and delay == 1 and outcome == "late"


def test_missing_delivery_has_no_label(wh):
    for oid in ["o_nodelivery", "o_delivered_nodate"]:
        is_late, delivered, delivery_days, delay = get(wh, oid, "is_late, is_delivered, delivery_days, delay_days")
        assert is_late is None and delivered is False and delivery_days is None and delay is None
    label, eligible, reason = wh.execute(
        "SELECT is_late, ml_eligible, ml_exclusion_reason FROM ml_labels WHERE order_id = 'o_nodelivery'").fetchone()
    assert label is None and eligible is False and reason == "not_delivered"


def test_delivered_status_without_date_is_separate_outcome(wh):
    assert get(wh, "o_delivered_nodate", "outcome")[0] == "delivered_missing_date"


def test_temporal_error_excluded_from_ml(wh):
    assert get(wh, "o_temporal", "ml_exclusion_reason")[0] == "temporal_error"
    assert get(wh, "o_temporal", "flag_delivered_before_approval")[0] is True


def test_no_items_excluded_from_ml(wh):
    # unavailable order without items: excluded as not delivered (first failing rule)
    assert get(wh, "o_noitems", "ml_exclusion_reason")[0] == "not_delivered"


def test_delivery_days_definition(wh):
    days = get(wh, "o_sameday", "delivery_days")[0]
    # purchase 2017-03-01 10:00 -> delivered 2017-03-20 23:59
    assert abs(days - (19 + (13 * 60 + 59) / 1440)) < 1e-6
