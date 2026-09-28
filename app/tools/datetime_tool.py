"""Date and time tool demonstrating external environment access."""

from datetime import datetime
from typing import Dict, Any


def get_current_datetime(timezone_offset_hours: int = 1) -> Dict[str, Any]:
    """Retrieve current system date and time.
    
    Args:
        timezone_offset_hours: Timezone offset in hours (default 1 for CET/Algiers).
    """
    now = datetime.now()
    return {
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p"),
    }
