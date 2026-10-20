import numpy as np
import pandas as pd

from trial_conversion_model.api.routes import to_band
from trial_conversion_model.features import add_features


def make_row(**overrides):
    row = {
          "sessions_day1": 0,
          "sessions_day2": 0,
          "sessions_day3": 0,
          "listen_sessions_3d": 0,
          "total_minutes_3d": 0.0,
          "country": "SG",
          "device_type": "ios",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_zero_sessions_gives_zero_not_nan():
    out = add_features(make_row(sessions_3d=0))
    for col in ["day1_share", "listen_share", "avg_session_minutes"]:
        assert not out[col].isna().any(), f"{col} is NaN"
        assert (out[col] == 0).all(), f"{col} should be 0"
        assert not np.isinf(out[col]).any()


def test_nonzero_sessions_computes_share():
    out = add_features(make_row(sessions_day1=2, sessions_day2=1,
                                sessions_day3=1, listen_sessions_3d=1, total_minutes_3d=40.0))
    assert out["day1_share"].iloc[0] == 0.5
    assert out["listen_share"].iloc[0] == 0.25
    assert out["avg_session_minutes"].iloc[0] == 10.0

def test_a_high_probability_gets_the_high_band():
    assert to_band(0.9) == "high"




def test_zero_session_trial_gets_zero_shares_not_nan():
    df = pd.DataFrame([{
        "sessions_day1": 0,
        "sessions_day2": 0,
        "sessions_day3": 0,
        "listen_sessions_3d": 0,
        "total_minutes_3d": 0.0,
        "country": "US",
        "device_type": "iOS",
    }])

    out = add_features(df)

    assert out["day1_share"].iloc[0] == 0
    assert out["listen_share"].iloc[0] == 0
    assert out["avg_session_minutes"].iloc[0] == 0
    # TODO: write this one yourself, in three steps:
    #   1. Build a one-row DataFrame for a trial with nothing in it: zero
    #      sessions on each of the three days, zero listen sessions, and
    #      zero total minutes.
    #   2. Pass it through add_features.
    #   3. Assert that day1_share, listen_share and avg_session_minutes each
    #      come back as 0 rather than a missing value.
    
    