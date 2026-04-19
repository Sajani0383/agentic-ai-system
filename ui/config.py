# UI Palette Configuration
PALETTE = {
    "bg": "#081018",
    "panel": "#101b2b",
    "border": "#20334b",
    "text": "#eef4ff",
    "muted": "#95a6bd",
    "blue": "#4da3ff",
    "green": "#4bd38a",
    "gold": "#d8c86e",
    "coral": "#ff746c",
    "cyan": "#6edff6",
    "violet": "#9f8cff",
}

# Baseline Supported Simulation Scenarios
SCENARIOS = [
    "Auto Schedule",
    "Normal Day",
    "Class Changeover",
    "Exam Rush",
    "Sports Event",
    "Fest Night",
    "Emergency Spillover",
]

# Business Thresholds for UI Renders
THRESHOLDS = {
    "utilization": {
        "critical": 90,
        "warning": 80,
        "active": 65
    },
    "free_slots": {
        "preferred": 20,
        "overflow": 10
    }
}

# Base UI strings
STRINGS = {
    "title": "SRM Agentic Parking Command Center",
    "subtitle": "Event-aware campus parking simulation with demand prediction, dynamic space allocation, measurable performance outcomes, and proactive parking recommendations.",
    "recommendations": {
        "preferred": "Preferred",
        "overflow": "Overflow",
        "avoid": "Avoid"
    }
}
