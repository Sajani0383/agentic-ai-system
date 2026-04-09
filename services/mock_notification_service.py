import json
import os
from copy import deepcopy
from datetime import datetime


class MockNotificationService:
    def __init__(self, storage_path=None):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.storage_path = storage_path or os.path.join(base_dir, "memory", "notification_feed.json")
        self.deliveries = []
        self._load()

    def dispatch(self, notifications, event_context):
        batch = []
        channels = ["mobile_app", "sms_gateway", "campus_signage"]
        for notification in notifications:
            for channel in channels:
                batch.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "channel": channel,
                        "event": event_context.get("name", "Campus Update"),
                        "title": notification.get("title", "Update"),
                        "message": notification.get("message", ""),
                        "level": notification.get("level", "info"),
                        "delivery_status": "queued" if channel == "sms_gateway" else "sent",
                    }
                )
        if batch:
            self.deliveries.extend(batch)
            self.deliveries = self.deliveries[-300:]
            self._save()
        return deepcopy(batch)

    def get_recent_deliveries(self, limit=25):
        return deepcopy(self.deliveries[-limit:])

    def reset(self):
        self.deliveries = []
        self._save()

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return
        self.deliveries = payload.get("deliveries", [])

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as file:
            json.dump({"deliveries": self.deliveries}, file, indent=2)
