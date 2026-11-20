"""带时区的实验预算边界：只限制新任务，已经启动的任务正常完成。"""

from datetime import datetime, timezone


def parse_deadline(value):
    if not value:
        return None
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("截止时间必须带时区，例如2026-11-21T12:00:00+08:00")
    return result


def budget_expired(deadline, current=None):
    return deadline is not None and (current or datetime.now(timezone.utc)) >= deadline
