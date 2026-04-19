import json
import os
import heapq
import time
import hashlib
import random
from copy import deepcopy
from datetime import datetime

# Assume trace_logger exists; we import optionally or fall back.
try:
    from adk.trace_logger import trace_logger
except ImportError:
    class DummyLogger:
        def log(self, *args, **kwargs): pass
    trace_logger = DummyLogger()

class MockNotificationService:
    LEVEL_PRIORITY = {"critical": 1, "error": 2, "warning": 3, "info": 4}

    def __init__(self, storage_path=None, logger=None):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.storage_path = storage_path or os.path.join(base_dir, "memory", "notification_feed.json")
        self.deliveries = []
        self.queue = []      # Priority queue using heapq
        self.dedup_cache = {} # hash -> expiry_time
        self.last_flush = time.time()
        self.logger = logger or trace_logger
        
        # Rate Limiting
        self.MAX_RATE_PER_SEC = 20
        self.dispatch_timestamps = []
        
        self.dedup_ttl_seconds = 60
        self._load()

    def _validate_schema(self, notification):
        if not isinstance(notification, dict):
            return False, "Notification must be a dict"
        if not notification.get("title") or not isinstance(notification["title"], str):
            return False, "Missing or invalid 'title'"
        if not notification.get("message") or not isinstance(notification["message"], str):
            return False, "Missing or invalid 'message'"
        return True, ""

    def dispatch(self, notifications, event_context):
        batch = []
        channels = ["mobile_app", "sms_gateway", "campus_signage"]
        
        # Filter rate limit out-of-window
        current_time = time.time()
        self.dispatch_timestamps = [t for t in self.dispatch_timestamps if current_time - t < 1.0]

        for notification in notifications:
            valid, err = self._validate_schema(notification)
            if not valid:
                self.logger.log("-", "notification_error", {"error": err, "payload": notification}, level="ERROR")
                continue

            level = notification.get("level", "info").lower()
            priority = self.LEVEL_PRIORITY.get(level, 5)

            for channel in channels:
                # Deduplication logic
                msg_hash = hashlib.md5(f"{notification['title']}_{notification['message']}_{channel}".encode()).hexdigest()
                if msg_hash in self.dedup_cache and self.dedup_cache[msg_hash] > current_time:
                    continue # Skip duplicate
                
                # Rate Limiting
                if len(self.dispatch_timestamps) >= self.MAX_RATE_PER_SEC:
                    self.logger.log("-", "notification_rate_limit", {"channel": channel, "title": notification['title']}, level="WARN")
                    # Push to queue anyway, but it won't be dispatched immediately next cycle
                else:
                    self.dispatch_timestamps.append(current_time)

                self.dedup_cache[msg_hash] = current_time + self.dedup_ttl_seconds

                msg_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "channel": channel,
                    "event": event_context.get("name", "Campus Update"),
                    "title": notification["title"],
                    "message": notification["message"],
                    "level": level,
                    "delivery_status": "queued",
                    "retry_count": 0,
                    "priority": priority,
                    "msg_hash": msg_hash
                }
                
                # Push to heap: format is (priority, timestamp, msg_record)
                heapq.heappush(self.queue, (priority, datetime.utcnow().timestamp(), msg_record))
                batch.append(msg_record)
                self.logger.log("-", "notification_queued", {"channel": channel, "title": msg_record['title']}, level="INFO")

        return deepcopy(batch)

    def process_queue(self):
        """
        Simulate async delivery: iterates over the priority queue, evaluates success, marks as sent or failed.
        Re-queues failures if retry_count < 3.
        """
        if not self.queue:
            return

        current_time = time.time()
        processed = []
        retry_elements = []

        while self.queue:
            priority, ts, msg = heapq.heappop(self.queue)
            
            # Simulate chance of failure. SMS gateway fails 5% of the time.
            drop_rate = 0.05 if msg["channel"] == "sms_gateway" else 0.01
            success = random.random() > drop_rate
            
            if success:
                msg["delivery_status"] = "sent"
                processed.append(msg)
                self.logger.log("-", "notification_sent", {"channel": msg["channel"], "title": msg["title"]}, level="INFO")
            else:
                msg["retry_count"] += 1
                if msg["retry_count"] < 3:
                    msg["delivery_status"] = "queued"  # Will retry
                    retry_elements.append((priority, current_time, msg))
                    self.logger.log("-", "notification_failed_retry", {"channel": msg["channel"], "title": msg["title"], "retry": msg["retry_count"]}, level="WARN")
                else:
                    msg["delivery_status"] = "failed"
                    processed.append(msg)
                    self.logger.log("-", "notification_failed_permanent", {"channel": msg["channel"], "title": msg["title"]}, level="ERROR")

        # Push elements back for retry
        for item in retry_elements:
            heapq.heappush(self.queue, item)

        if processed:
            # We filter out msg_hash and priority for the final stored record
            clean_processed = [
                {k: v for k, v in p.items() if k not in ["msg_hash", "priority"]}
                for p in processed
            ]
            self.deliveries.extend(clean_processed)
            self.deliveries = self.deliveries[-300:]
            
            # Smart flush to disk
            self.flush()

    def get_recent_deliveries(self, limit=25, channel=None, level=None, status=None):
        filtered = []
        for delivery in reversed(self.deliveries):
            if channel and delivery.get("channel") != channel:
                continue
            if level and delivery.get("level") != level:
                continue
            if status and delivery.get("delivery_status") != status:
                continue
            
            filtered.append(delivery)
            if len(filtered) >= limit:
                break
        
        filtered.reverse()
        return deepcopy(filtered)

    def flush(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as file:
            json.dump({"deliveries": self.deliveries}, file, indent=2)

    def reset(self):
        self.deliveries = []
        self.queue = []
        self.dedup_cache = {}
        self.dispatch_timestamps = []
        self.flush()

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return
        self.deliveries = payload.get("deliveries", [])
