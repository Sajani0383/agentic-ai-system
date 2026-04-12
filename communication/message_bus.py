import asyncio
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime


class MessageBus:
    DEFAULT_TOPICS = {
        "monitoring",
        "planning",
        "critic",
        "execution",
        "policy",
        "reward",
        "system",
        "broadcast",
    }

    def __init__(self, max_messages=500, allowed_topics=None):
        self.subscribers = defaultdict(list)
        self.messages = deque(maxlen=max(1, int(max_messages)))
        self.allowed_topics = set(allowed_topics or self.DEFAULT_TOPICS)
        self.delivery_errors = []

    def subscribe(self, topic, agent):
        topic = self._validate_topic(topic)
        self._validate_agent(agent)
        if agent not in self.subscribers[topic]:
            self.subscribers[topic].append(agent)
        return True

    def unsubscribe(self, topic, agent):
        topic = self._validate_topic(topic)
        subscribers = self.subscribers.get(topic, [])
        if agent in subscribers:
            subscribers.remove(agent)
            if not subscribers and topic in self.subscribers:
                del self.subscribers[topic]
            return True
        return False

    def publish(
        self,
        topic,
        message,
        sender="system",
        message_type="event",
        priority="normal",
        target_agents=None,
        metadata=None,
    ):
        topic = self._validate_topic(topic)
        payload = self._build_message(
            topic=topic,
            message=message,
            sender=sender,
            message_type=message_type,
            priority=priority,
            metadata=metadata,
        )
        self.messages.append(payload)
        recipients = self._select_recipients(topic, target_agents)
        return self._deliver(payload, recipients)

    async def publish_async(
        self,
        topic,
        message,
        sender="system",
        message_type="event",
        priority="normal",
        target_agents=None,
        metadata=None,
    ):
        topic = self._validate_topic(topic)
        payload = self._build_message(
            topic=topic,
            message=message,
            sender=sender,
            message_type=message_type,
            priority=priority,
            metadata=metadata,
        )
        self.messages.append(payload)
        recipients = self._select_recipients(topic, target_agents)
        return await self._deliver_async(payload, recipients)

    def get_messages(self, topic=None, priority=None, limit=None):
        messages = list(self.messages)
        if topic is not None:
            messages = [message for message in messages if message.get("topic") == topic]
        if priority is not None:
            messages = [message for message in messages if message.get("priority") == priority]
        if limit is not None:
            messages = messages[-max(0, int(limit)) :]
        return deepcopy(messages)

    def get_delivery_errors(self, limit=None):
        errors = self.delivery_errors[-max(0, int(limit)) :] if limit is not None else self.delivery_errors
        return deepcopy(errors)

    def _build_message(self, topic, message, sender, message_type, priority, metadata):
        return {
            "topic": topic,
            "message": deepcopy(message),
            "sender": str(sender or "system"),
            "type": str(message_type or "event"),
            "priority": self._normalize_priority(priority),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": deepcopy(metadata) if isinstance(metadata, dict) else {},
        }

    def _select_recipients(self, topic, target_agents):
        recipients = list(self.subscribers.get(topic, []))
        if topic == "broadcast":
            for agents in self.subscribers.values():
                for agent in agents:
                    if agent not in recipients:
                        recipients.append(agent)
        if not target_agents:
            return recipients
        target_set = set(target_agents)
        return [agent for agent in recipients if self._agent_name(agent) in target_set or agent in target_set]

    def _deliver(self, payload, recipients):
        deliveries = []
        for agent in recipients:
            try:
                agent.receive(payload["topic"], deepcopy(payload))
                deliveries.append({"agent": self._agent_name(agent), "status": "delivered"})
            except Exception as exc:
                error = {
                    "agent": self._agent_name(agent),
                    "topic": payload["topic"],
                    "error": str(exc),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                self.delivery_errors.append(error)
                self.delivery_errors = self.delivery_errors[-100:]
                deliveries.append({"agent": self._agent_name(agent), "status": "failed", "error": str(exc)})
        return {
            "published": True,
            "topic": payload["topic"],
            "recipients": deliveries,
            "message": deepcopy(payload),
        }

    async def _deliver_async(self, payload, recipients):
        async def send(agent):
            try:
                receive = getattr(agent, "receive")
                if asyncio.iscoroutinefunction(receive):
                    await receive(payload["topic"], deepcopy(payload))
                else:
                    receive(payload["topic"], deepcopy(payload))
                return {"agent": self._agent_name(agent), "status": "delivered"}
            except Exception as exc:
                error = {
                    "agent": self._agent_name(agent),
                    "topic": payload["topic"],
                    "error": str(exc),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                self.delivery_errors.append(error)
                self.delivery_errors = self.delivery_errors[-100:]
                return {"agent": self._agent_name(agent), "status": "failed", "error": str(exc)}

        deliveries = await asyncio.gather(*(send(agent) for agent in recipients)) if recipients else []
        return {
            "published": True,
            "topic": payload["topic"],
            "recipients": deliveries,
            "message": deepcopy(payload),
        }

    def _validate_topic(self, topic):
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")
        normalized = topic.strip()
        if self.allowed_topics and normalized not in self.allowed_topics:
            raise ValueError(f"Unsupported topic: {normalized}")
        return normalized

    def _validate_agent(self, agent):
        if not hasattr(agent, "receive") or not callable(getattr(agent, "receive")):
            raise TypeError("Subscribed agent must implement a callable receive(topic, message) method.")

    def _normalize_priority(self, priority):
        normalized = str(priority or "normal").lower()
        if normalized not in {"low", "normal", "high", "critical"}:
            return "normal"
        return normalized

    def _agent_name(self, agent):
        return getattr(agent, "name", agent.__class__.__name__)
