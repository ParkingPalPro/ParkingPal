from datetime import datetime

def calculate_duration(entrance_time, exit_time):
    try:
        start = datetime.fromisoformat(entrance_time)
        end = datetime.fromisoformat(exit_time)
        duration = (end - start).total_seconds() / 60
        return int(duration)
    except:
        return 0