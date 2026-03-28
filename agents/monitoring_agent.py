class MonitoringAgent:
    def observe(self, source):
        if hasattr(source, "get_state"):
            return source.get_state()
        return source
