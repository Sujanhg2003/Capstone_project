
from datetime import datetime


def get_current_datetime() -> dict:
    current_time = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    return {"datetime": current_time}
