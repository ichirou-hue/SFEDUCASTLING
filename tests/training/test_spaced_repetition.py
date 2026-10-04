from datetime import UTC, datetime, timedelta

from backend.services.review_logic import DEFAULT_EASE, next_review_state


NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def test_first_correct_schedules_one_day():
    state = next_review_state(reps=0, ease=DEFAULT_EASE, correct=True, now=NOW)
    assert state["reps"] == 1
    assert state["interval_days"] == 1.0
    assert state["next_review_at"] == NOW + timedelta(days=1)


def test_second_correct_multiplies_interval_by_2_2():
    state = next_review_state(reps=1, ease=2.2, correct=True, now=NOW)
    assert state["reps"] == 2
    assert state["interval_days"] == 2.2
    assert state["next_review_at"] == NOW + timedelta(days=2.2)


def test_third_correct_keeps_geometric_growth():
    state = next_review_state(reps=2, ease=2.2, correct=True, now=NOW)
    assert state["reps"] == 3
    assert state["interval_days"] == 4.84
    assert state["next_review_at"] == NOW + timedelta(days=4.84)


def test_error_resets_reps_and_schedules_one_day():
    state = next_review_state(reps=5, ease=2.2, correct=False, now=NOW)
    assert state["reps"] == 0
    assert state["interval_days"] == 1.0
    assert state["next_review_at"] == NOW + timedelta(days=1)


def test_correct_after_error_starts_new_series_from_one_day():
    state = next_review_state(reps=0, ease=2.2, correct=True, now=NOW)
    assert state["reps"] == 1
    assert state["interval_days"] == 1.0


def test_training_review_timestamps_are_timezone_aware():
    from backend.models.training_review import TrainingReview

    assert TrainingReview.__table__.c.next_review_at.type.timezone is True
    assert TrainingReview.__table__.c.created_at.type.timezone is True
    assert TrainingReview.__table__.c.updated_at.type.timezone is True
