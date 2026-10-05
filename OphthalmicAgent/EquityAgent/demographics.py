"""FairVision reliability categories; GDP has a separate documented convention."""
import math


def fairvision_age_group(age):
    try:
        if isinstance(age, bool):
            raise ValueError("Boolean is not an age")
        age = float(age)
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing/invalid FairVision age") from exc
    if not math.isfinite(age) or age < 0:
        raise ValueError("FairVision reliability requires a valid nonnegative age")
    return "younger" if age < 50 else "middle-aged" if age < 70 else "older"
