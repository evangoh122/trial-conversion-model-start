from typing import Literal
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """One trial's first-3-day base aggregates, as the caller knows them."""
    sessions_day1: int
    sessions_day2: int
    sessions_day3: int
    listen_sessions_3d: int
    total_minutes_3d: float
    country: str
    device_type: str

    # TODO: declare the seven fields a caller must send, one line each,
    # with a type for each. They are the same seven base aggregates your
    # feature code starts from; features.py names them.

class PredictionResponse(BaseModel):
    """What we send back: a probability and a band a human can act on."""

    # TODO: declare the two fields the caller gets back: the conversion
    # probability, and the low/medium/high band it falls in.
    conversion_probability: float
    conversion_band: Literal["low", "medium", "high"]