from sdt.realtime.months import (add, between, data_ends, index, month, targets, tuning_end, validation_months)


def test_month_arithmetic():
    assert month(index(202412)) == 202412 and add(202412, 1) == 202501 and add(202501, -1) == 202412
    assert add(202404, 12) == 202504 and between(202411, 202502) == [202411, 202412, 202501, 202502]


def test_data_ends_and_targets():
    ends = data_ends()
    assert len(ends) == 29 and ends[0] == 202404 and ends[-1] == 202608
    assert data_ends(202407) == [202404, 202405, 202406]
    assert targets(202404) == between(202405, 202504) and targets(202608) == [202609]
    assert targets(202404, last_month=202407) == [202405, 202406, 202407]


def test_boosting_is_tuned_every_third_data_end():
    assert [e for e in data_ends() if tuning_end(e) == e] == [202404, 202407, 202410, 202501, 202504, 202507, 202510,
                                                              202601, 202604, 202607]
    assert tuning_end(202406) == 202404 and tuning_end(202608) == 202607


def test_validation_months_are_the_latest_six_from_june_2024():
    assert validation_months(202405) == [] and validation_months(202406) == [202406]
    assert validation_months(202409) == between(202406, 202409)
    assert validation_months(202412) == between(202407, 202412)
