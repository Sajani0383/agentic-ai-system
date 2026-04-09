import random
from copy import deepcopy
from datetime import datetime

from ml.predict import predict_demand


class ParkingEnvironment:
    def __init__(self, zones=None):
        self.default_zones = [
            "Academic Block",
            "Library",
            "Innovation Lab",
            "Hostel Hub",
            "Stadium",
        ]
        self.zones = zones or list(self.default_zones)
        self.zone_map = {zone: index for index, zone in enumerate(self.zones)}
        self.history = []
        self.step_count = 0
        self.day_index = 0
        self.simulated_hour = 8
        self.last_transition = {}
        self.scenario_mode = "Auto Schedule"
        self.event_catalog = self._build_event_catalog()
        self.reset()

    def reset(self):
        self.step_count = 0
        self.day_index = 0
        self.simulated_hour = 8
        self.state = {
            zone: {
                "total_slots": random.randint(90, 140),
                "occupied": random.randint(25, 55),
                "entry": 0,
                "exit": 0,
            }
            for zone in self.zones
        }
        self.history = [self.get_state()]
        self.last_transition = {
            "step": 0,
            "applied_action": {"action": "none"},
            "transferred": 0,
            "totals": {"entries": 0, "exits": 0, "occupancy_change": 0, "denied_entries": 0},
            "zones": [],
            "event_context": self.get_event_context(),
            "notifications": [],
            "kpis": {},
        }
        return self.get_state()

    def step(self, action=None):
        previous_state = self.get_state()
        self.step_count += 1
        self._advance_time()
        event_context = self.get_event_context()

        transfer_report = {"moved": 0, "from": None, "to": None, "requested": 0}
        if action:
            transfer_report = self.apply_action(action)

        total_denied_entries = 0
        for zone in self.zones:
            zone_id = self.zone_map[zone]
            base_demand = max(0, predict_demand(self.simulated_hour, self.day_index + 1, zone_id, 0))
            event_multiplier = event_context["zone_multipliers"].get(zone, 1.0)
            time_multiplier = self._time_multiplier(zone)
            demand = int(base_demand * event_multiplier * time_multiplier)

            total = self.state[zone]["total_slots"]
            occupied = self.state[zone]["occupied"]
            occupancy_ratio = occupied / total if total else 0
            free_capacity = max(0, total - occupied)

            if occupancy_ratio > 0.88:
                raw_entry = int(demand * random.uniform(0.08, 0.16))
            elif occupancy_ratio > 0.72:
                raw_entry = int(demand * random.uniform(0.14, 0.24))
            else:
                raw_entry = int(demand * random.uniform(0.20, 0.36))

            event_exit_bias = 0.02 if zone == event_context["focus_zone"] else 0.0
            if occupancy_ratio > 0.78:
                exit_count = int(occupied * random.uniform(0.18, 0.30 + event_exit_bias))
            else:
                exit_count = int(occupied * random.uniform(0.08, 0.16 + event_exit_bias))

            entry = min(max(0, raw_entry), max(0, free_capacity + exit_count))
            denied_entries = max(0, raw_entry - entry)
            total_denied_entries += denied_entries
            exit_count = min(max(0, exit_count), occupied)

            self.state[zone]["entry"] = max(0, entry)
            self.state[zone]["exit"] = max(0, exit_count)
            self.state[zone]["occupied"] += self.state[zone]["entry"] - self.state[zone]["exit"]
            self.state[zone]["occupied"] = max(0, min(self.state[zone]["occupied"], total))

        new_state = self.get_state()
        reward = self._calculate_reward(previous_state, new_state, action)
        notifications = self._build_notifications(new_state, event_context, transfer_report, total_denied_entries)
        kpis = self._build_kpis(previous_state, new_state, transfer_report, total_denied_entries)
        self.last_transition = self._build_transition_report(
            previous_state,
            new_state,
            action,
            transfer_report,
            datetime.now().isoformat(),
            event_context,
            notifications,
            kpis,
            total_denied_entries,
        )

        self.history.append(new_state)
        if len(self.history) > 80:
            self.history.pop(0)

        return new_state, reward

    def set_scenario_mode(self, scenario_mode):
        self.scenario_mode = scenario_mode if scenario_mode in self.event_catalog else "Auto Schedule"

    def get_scenario_mode(self):
        return self.scenario_mode

    def get_event_context(self):
        event = self.event_catalog.get(self.scenario_mode) or self._get_auto_scheduled_event()
        zone_multipliers = {zone: 1.0 for zone in self.zones}
        zone_multipliers.update(event.get("zone_multipliers", {}))
        focus_zone = event["focus_zone"] if event["focus_zone"] in self.zones else self.zones[0]
        recommended_zone = event["recommended_zone"] if event["recommended_zone"] in self.zones else self.zones[min(1, len(self.zones) - 1)]
        return {
            "name": event["name"],
            "severity": event["severity"],
            "description": event["description"],
            "focus_zone": focus_zone,
            "recommended_zone": recommended_zone,
            "allocation_strategy": event["allocation_strategy"],
            "zone_multipliers": zone_multipliers,
            "user_advisory": event["user_advisory"],
            "time_window": f"{self.simulated_hour:02d}:00 - {(self.simulated_hour + 1) % 24:02d}:00",
        }

    def apply_action(self, action):
        if action.get("action") != "redirect":
            return {"moved": 0, "from": None, "to": None, "requested": 0}

        from_zone = action.get("from")
        to_zone = action.get("to")
        vehicles = int(action.get("vehicles", 0))

        if from_zone not in self.state or to_zone not in self.state or from_zone == to_zone:
            return {"moved": 0, "from": from_zone, "to": to_zone, "requested": vehicles}

        available_to_move = min(vehicles, self.state[from_zone]["occupied"])
        free_capacity = self.state[to_zone]["total_slots"] - self.state[to_zone]["occupied"]
        transfer = max(0, min(available_to_move, free_capacity))

        if transfer == 0:
            return {"moved": 0, "from": from_zone, "to": to_zone, "requested": vehicles}

        self.state[from_zone]["occupied"] -= transfer
        self.state[to_zone]["occupied"] += transfer
        return {"moved": transfer, "from": from_zone, "to": to_zone, "requested": vehicles}

    def get_state(self):
        return {
            zone: {
                "total_slots": self.state[zone]["total_slots"],
                "occupied": self.state[zone]["occupied"],
                "free_slots": self.state[zone]["total_slots"] - self.state[zone]["occupied"],
                "entry": self.state[zone]["entry"],
                "exit": self.state[zone]["exit"],
            }
            for zone in self.zones
        }

    def get_trend(self):
        return self.history

    def get_last_transition(self):
        return self.last_transition

    def export_snapshot(self):
        return {
            "zones": deepcopy(self.zones),
            "state": deepcopy(self.state),
            "history": deepcopy(self.history),
            "step_count": self.step_count,
            "last_transition": deepcopy(self.last_transition),
            "day_index": self.day_index,
            "simulated_hour": self.simulated_hour,
            "scenario_mode": self.scenario_mode,
        }

    def restore_default_layout(self):
        self.zones = list(self.default_zones)
        self.zone_map = {zone: index for index, zone in enumerate(self.zones)}
        self.scenario_mode = "Auto Schedule"
        self.reset()

    def load_snapshot(self, snapshot):
        self.zones = snapshot.get("zones", self.zones)
        self.zone_map = {zone: index for index, zone in enumerate(self.zones)}
        self.state = deepcopy(snapshot.get("state", self.state))
        self.history = deepcopy(snapshot.get("history", [self.get_state()]))
        self.step_count = snapshot.get("step_count", 0)
        self.day_index = snapshot.get("day_index", 0)
        self.simulated_hour = snapshot.get("simulated_hour", 8)
        self.scenario_mode = snapshot.get("scenario_mode", "Auto Schedule")
        self.last_transition = deepcopy(
            snapshot.get(
                "last_transition",
                {
                    "step": self.step_count,
                    "applied_action": {"action": "none"},
                    "transferred": 0,
                    "totals": {"entries": 0, "exits": 0, "occupancy_change": 0, "denied_entries": 0},
                    "zones": [],
                    "event_context": self.get_event_context(),
                    "notifications": [],
                    "kpis": {},
                },
            )
        )

    def _advance_time(self):
        self.simulated_hour += 1
        if self.simulated_hour > 20:
            self.simulated_hour = 8
            self.day_index = (self.day_index + 1) % 7

    def _time_multiplier(self, zone):
        base = 1.0
        if 8 <= self.simulated_hour <= 10:
            base += 0.18 if zone in {"Academic Block", "Library"} else 0.05
        elif 12 <= self.simulated_hour <= 14:
            base += 0.10
        elif 17 <= self.simulated_hour <= 19:
            base += 0.20 if zone in {"Hostel Hub", "Stadium"} else 0.08
        return base

    def _get_auto_scheduled_event(self):
        slot = self.step_count % 6
        if slot == 1:
            return self.event_catalog["Class Changeover"]
        if slot == 2:
            return self.event_catalog["Exam Rush"]
        if slot == 3:
            return self.event_catalog["Sports Event"]
        if slot == 4:
            return self.event_catalog["Fest Night"]
        if slot == 5:
            return self.event_catalog["Emergency Spillover"]
        return self.event_catalog["Normal Day"]

    def _build_event_catalog(self):
        return {
            "Auto Schedule": {"name": "Auto Schedule", "severity": "adaptive", "description": "Automatic campus demand schedule.", "focus_zone": "Academic Block", "recommended_zone": "Library", "allocation_strategy": "Adaptive overflow routing", "zone_multipliers": {}, "user_advisory": "The system is automatically selecting the current campus event profile."},
            "Normal Day": {"name": "Normal Day", "severity": "low", "description": "Baseline weekday parking demand across campus.", "focus_zone": "Academic Block", "recommended_zone": "Library", "allocation_strategy": "Balanced utilisation", "zone_multipliers": {"Academic Block": 1.1, "Library": 1.0, "Innovation Lab": 1.0, "Hostel Hub": 0.95, "Stadium": 0.85}, "user_advisory": "Use the nearest academic parking zone. Overflow demand is currently low."},
            "Class Changeover": {"name": "Class Changeover", "severity": "medium", "description": "Large inter-building movement between academic blocks during class transitions.", "focus_zone": "Academic Block", "recommended_zone": "Innovation Lab", "allocation_strategy": "Rapid overflow routing", "zone_multipliers": {"Academic Block": 1.45, "Library": 1.1, "Innovation Lab": 1.2, "Hostel Hub": 0.9, "Stadium": 0.8}, "user_advisory": "Drivers should prefer Innovation Lab parking to reduce queues near the Academic Block."},
            "Exam Rush": {"name": "Exam Rush", "severity": "high", "description": "Exam sessions create sharp demand near study areas and academic halls.", "focus_zone": "Library", "recommended_zone": "Hostel Hub", "allocation_strategy": "Demand smoothing", "zone_multipliers": {"Academic Block": 1.3, "Library": 1.55, "Innovation Lab": 1.25, "Hostel Hub": 1.05, "Stadium": 0.75}, "user_advisory": "Visitors should be redirected early to Hostel Hub and satellite lots during exam peaks."},
            "Sports Event": {"name": "Sports Event", "severity": "high", "description": "A campus sports event drives heavy incoming flow toward the Stadium.", "focus_zone": "Stadium", "recommended_zone": "Hostel Hub", "allocation_strategy": "Event-priority routing", "zone_multipliers": {"Academic Block": 0.95, "Library": 0.85, "Innovation Lab": 1.0, "Hostel Hub": 1.2, "Stadium": 1.75}, "user_advisory": "Use Hostel Hub overflow parking and walk or shuttle to the Stadium."},
            "Fest Night": {"name": "Fest Night", "severity": "critical", "description": "Cultural fest traffic pushes both event and nearby residential zones into congestion.", "focus_zone": "Stadium", "recommended_zone": "Innovation Lab", "allocation_strategy": "Festival spillover containment", "zone_multipliers": {"Academic Block": 1.0, "Library": 0.8, "Innovation Lab": 1.3, "Hostel Hub": 1.4, "Stadium": 1.9}, "user_advisory": "Advance parking alerts should send users to Innovation Lab overflow before they reach the Stadium."},
            "Emergency Spillover": {"name": "Emergency Spillover", "severity": "critical", "description": "A sudden closure or blockage forces fast redistribution across the network.", "focus_zone": "Academic Block", "recommended_zone": "Library", "allocation_strategy": "Protective rerouting", "zone_multipliers": {"Academic Block": 1.6, "Library": 1.25, "Innovation Lab": 1.15, "Hostel Hub": 1.0, "Stadium": 0.9}, "user_advisory": "The system should notify drivers immediately and reroute them away from the affected zone."},
        }

    def _build_notifications(self, state, event_context, transfer_report, denied_entries):
        notifications = [
            {
                "level": "info",
                "title": f"{event_context['name']} active",
                "message": event_context["user_advisory"],
            }
        ]
        focus_zone = event_context["focus_zone"]
        recommended_zone = event_context["recommended_zone"]
        if state[focus_zone]["free_slots"] < 12:
            notifications.append(
                {
                    "level": "warning",
                    "title": f"Congestion risk at {focus_zone}",
                    "message": f"Recommend sending incoming drivers to {recommended_zone} to protect event access.",
                }
            )
        if denied_entries > 0:
            notifications.append(
                {
                    "level": "error",
                    "title": "Overflow pressure detected",
                    "message": f"{denied_entries} simulated vehicles could not enter first-choice zones this cycle.",
                }
            )
        if transfer_report.get("moved", 0) > 0:
            notifications.append(
                {
                    "level": "success",
                    "title": "Dynamic allocation applied",
                    "message": f"{transfer_report['moved']} vehicles were reallocated from {transfer_report['from']} to {transfer_report['to']}.",
                }
            )
        return notifications

    def _build_kpis(self, previous_state, new_state, transfer_report, denied_entries):
        total_capacity = sum(zone["total_slots"] for zone in new_state.values())
        total_occupied = sum(zone["occupied"] for zone in new_state.values())
        utilisation = round((total_occupied / total_capacity) * 100, 2) if total_capacity else 0.0
        search_time = round(1.8 + denied_entries * 0.12 + max(0, utilisation - 72) * 0.08, 2)
        requested = transfer_report.get("requested", 0)
        moved = transfer_report.get("moved", 0)
        allocation_success = round((moved / requested) * 100, 2) if requested else 100.0
        free_slots = [zone["free_slots"] for zone in new_state.values()]
        balance_index = round(max(free_slots) - min(free_slots), 2) if free_slots else 0.0
        return {
            "space_utilisation_pct": utilisation,
            "estimated_search_time_min": search_time,
            "allocation_success_pct": allocation_success,
            "congestion_hotspots": sum(1 for zone in new_state.values() if zone["free_slots"] < 10),
            "balance_index": balance_index,
        }

    def _calculate_reward(self, previous_state, new_state, action):
        previous_pressure = sum(max(0, 10 - zone["free_slots"]) for zone in previous_state.values())
        new_pressure = sum(max(0, 10 - zone["free_slots"]) for zone in new_state.values())
        balance_bonus = -(
            max(zone["occupied"] for zone in new_state.values())
            - min(zone["occupied"] for zone in new_state.values())
        ) / 20
        action_bonus = 1 if action and action.get("action") == "redirect" else 0
        return round((previous_pressure - new_pressure) + balance_bonus + action_bonus, 2)

    def _build_transition_report(
        self,
        previous_state,
        new_state,
        action,
        transfer_report,
        timestamp,
        event_context,
        notifications,
        kpis,
        denied_entries,
    ):
        zone_reports = []
        total_entries = 0
        total_exits = 0
        total_change = 0

        for zone in self.zones:
            before = previous_state[zone]
            after = new_state[zone]
            occupancy_change = after["occupied"] - before["occupied"]
            total_entries += after["entry"]
            total_exits += after["exit"]
            total_change += occupancy_change
            zone_reports.append(
                {
                    "zone": zone,
                    "occupied_before": before["occupied"],
                    "occupied_after": after["occupied"],
                    "free_before": before["free_slots"],
                    "free_after": after["free_slots"],
                    "entry": after["entry"],
                    "exit": after["exit"],
                    "occupancy_change": occupancy_change,
                }
            )

        applied_action = action or {"action": "none"}
        return {
            "step": self.step_count,
            "timestamp": timestamp,
            "simulated_hour": self.simulated_hour,
            "day_index": self.day_index,
            "applied_action": applied_action,
            "transferred": transfer_report.get("moved", 0),
            "transfer_detail": transfer_report,
            "totals": {
                "entries": total_entries,
                "exits": total_exits,
                "occupancy_change": total_change,
                "denied_entries": denied_entries,
            },
            "zones": zone_reports,
            "event_context": event_context,
            "notifications": notifications,
            "kpis": kpis,
        }
