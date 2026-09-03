"""Drift alerting. Log line always; Slack webhook if one is configured."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)


def fire(title: str, body: str, *, severity: str = "warning") -> bool:
    """Emit an alert. Returns True if it reached Slack, False if log-only.

    Never raises: a monitoring system that can take down the thing it monitors
    is worse than no monitoring system.
    """
    line = f"[{severity.upper()}] {title}\n{body}"
    (log.error if severity == "critical" else log.warning)(line)

    if not settings.slack_webhook_url:
        return False
    payload = json.dumps({
        "text": f"*{title}*",
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"FraudPulse: {title}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"```{body}```"}},
        ],
    }).encode()
    req = urllib.request.Request(
        settings.slack_webhook_url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("slack webhook failed (%s); alert was logged only", exc)
        return False
