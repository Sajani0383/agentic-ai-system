import random
from copy import deepcopy
from datetime import datetime

from ml.predict import predict_demand


class ParkingEnvironment:
    DEFAULT_CONFIG = {
        "history_limit": 80,
        "initial_capacity_min": 90,
        "initial_capacity_max": 140,
        "initial_occupied_min": 25,
        "initial_occupied_max": 55,
        "morning_primary_boost": 0.18,
        "morning_secondary_boost": 0.05,
        "lunch_boost": 0.10,
        "evening_primary_boost": 0.20,
        "evening_secondary_boost": 0.08,
        "weather_trigger_probability": 0.32,
        "high_severity_block_probability": 0.22,
        "fallback_block_probability": 0.08,
        "vip_probability": 0.18,
        "vip_reserved_min": 2,
        "vip_reserved_max": 6,
        "rain_multiplier": 1.08,
        "blocked_entry_multiplier": 0.18,
        "blocked_exit_multiplier": 1.25,
        "queue_warning_threshold": 3,
        "congestion_alert_threshold": 12,
        "congestion_hotspot_threshold": 10,
        "search_time_base": 1.8,
        "search_time_denied_weight": 0.12,
        "search_time_utilisation_threshold": 72.0,
        "search_time_utilisation_weight": 0.08,
        "search_time_queue_weight": 0.3,
        "search_time_blocked_penalty": 0.4,
    }

    def __init__(self, zones=None, seed=None, config=None):
        self.default_zones = [
            "Academic Block",
            "Library",
            "Innovation Lab",
            "Hostel Hub",
            "Stadium",
        ]
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.zones = zones or list(self.default_zones)
        self._validate_zones(self.zones)
        self.zone_map = {zone: index for index, zone in enumerate(self.zones)}
        self.history = []
        self.step_count = 0
        self.day_index = 0
        self.simulated_hour = 8
        self.last_transition = {}
        self.active_dynamic_signals = {}
        self.scenario_mode = "Auto Schedule"
        self.event_catalog = self._build_event_catalog()
        self.reset()

    def reset(self):
        self.step_count = 0
        self.day_index = 0
        self.simulated_hour = 8
        self.active_dynamic_signals = {}
        self.state = {
            zone: {
                "total_slots": self.rng.randint(
                    self.config["initial_capacity_min"],
                    self.config["initial_capacity_max"],
                ),
                "occupied": self.rng.randint(
                    self.config["initial_occupied_min"],
                    self.config["initial_occupied_max"],
                ),
                "entry": 0,
                "exit": 0,
            }
            for zone in self.zones
        }
        self._validate_internal_state()
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
            "environment_score": 0.0,
            "step_breakdown": self.explain_step_model(),
        }
        return self.get_state()

    def step(self, action=None):
        previous_state = self.get_state()
        self._validate_action(action)
        self.step_count += 1
        self._advance_time()
        event_context = self.get_event_context()
        dynamic_signals = self._build_dynamic_signals(event_context)
        self.active_dynamic_signals = dynamic_signals

        zone_flow_plan = self._build_zone_flow_plan(event_context, dynamic_signals)
        transfer_report = self._plan_redirect(action, zone_flow_plan)
        self._apply_redirect_to_flow_plan(zone_flow_plan, transfer_report)
        new_state, total_denied_entries = self._apply_zone_flow_plan(zone_flow_plan)

        notifications = self._build_notifications(
            new_state,
            event_context,
            transfer_report,
            total_denied_entries,
            dynamic_signals,
        )
        kpis = self._build_kpis(
            previous_state,
            new_state,
            transfer_report,
            total_denied_entries,
            dynamic_signals,
        )
        environment_score = self._build_environment_score(
            previous_state,
            new_state,
            kpis,
            transfer_report,
            total_denied_entries,
        )
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
            dynamic_signals,
            environment_score,
        )

        self.history.append(new_state)
        if len(self.history) > self.config["history_limit"]:
            self.history.pop(0)

        return new_state, environment_score

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
            return {"moved": 0, "from": None, "to": None, "requested": 0, "status": "success"}

        from_zone = action.get("from")
        to_zone = action.get("to")
        vehicles = int(action.get("vehicles", 0))

        if from_zone not in self.state or to_zone not in self.state or from_zone == to_zone:
            return {"moved": 0, "from": from_zone, "to": to_zone, "requested": vehicles, "status": "invalid"}

        # Simulated realism: 15% chance of a real-world execution block
        if self.rng.random() < 0.15:
            reason = self.rng.choice(["Network latency timeout", "Security barrier unresponsive", "Dynamic signage offline"])
            return {"moved": 0, "from": from_zone, "to": to_zone, "requested": vehicles, "status": "failed", "reason": reason}

        source_redirect_capacity = max(
            self.state[from_zone]["entry"],
            max(0, self.config["congestion_hotspot_threshold"] + 2 - self.get_state()[from_zone]["free_slots"]),
        )
        free_capacity = self.state[to_zone]["total_slots"] - self.state[to_zone]["occupied"]
        transfer = max(0, min(vehicles, source_redirect_capacity, free_capacity))
        return {
            "moved": transfer,
            "from": from_zone,
            "to": to_zone,
            "requested": vehicles,
            "mode": "incoming_reroute",
            "status": "success"
        }

    def explain_step_model(self):
        return [
            "1. Advance simulated time and load the current event profile.",
            "2. Build operational signals such as weather, queues, blockages, and reserved spaces.",
            "3. Estimate entry and exit flow for each zone using demand, event pressure, and time-of-day multipliers.",
            "4. Apply redirect decisions by rerouting incoming arrivals across zones.",
            "5. Update occupancy, compute KPIs, and emit notifications plus a transition report.",
        ]

    def get_environment_summary(self):
        return {
            "environment_type": "dynamic event-driven stochastic parking simulation",
            "step_breakdown": self.explain_step_model(),
            "config": deepcopy(self.config),
            "seed": self.seed,
        }

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

    def get_operational_signals(self):
        return deepcopy(self.active_dynamic_signals)

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
            "seed": self.seed,
            "config": deepcopy(self.config),
            "active_dynamic_signals": deepcopy(self.active_dynamic_signals),
        }

    def restore_default_layout(self):
        self.zones = list(self.default_zones)
        self.zone_map = {zone: index for index, zone in enumerate(self.zones)}
        self.scenario_mode = "Auto Schedule"
        self.reset()

    def load_snapshot(self, snapshot):
        self.zones = snapshot.get("zones", self.zones)
        self._validate_zones(self.zones)
        self.zone_map = {zone: index for index, zone in enumerate(self.zones)}
        self.config.update(snapshot.get("config", {}))
        self.state = deepcopy(snapshot.get("state", self.state))
        self._validate_internal_state()
        self.history = deepcopy(snapshot.get("history", [self.get_state()]))
        self.step_count = snapshot.get("step_count", 0)
        self.day_index = snapshot.get("day_index", 0)
        self.simulated_hour = snapshot.get("simulated_hour", 8)
        self.scenario_mode = snapshot.get("scenario_mode", "Auto Schedule")
        self.active_dynamic_signals = deepcopy(snapshot.get("active_dynamic_signals", {}))
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
                    "environment_score": 0.0,
                    "step_breakdown": self.explain_step_model(),
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
            base += self.config["morning_primary_boost"] if zone in {"Academic Block", "Library"} else self.config["morning_secondary_boost"]
        elif 12 <= self.simulated_hour <= 14:
            base += self.config["lunch_boost"]
        elif 17 <= self.simulated_hour <= 19:
            base += self.config["evening_primary_boost"] if zone in {"Hostel Hub", "Stadium"} else self.config["evening_secondary_boost"]
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

    def _build_zone_flow_plan(self, event_context, dynamic_signals):
        zone_flow_plan = {}
        for zone in self.zones:
            zone_id = self.zone_map[zone]
            base_demand = max(0, predict_demand(self.simulated_hour, self.day_index + 1, zone_id, 0))
            event_multiplier = event_context["zone_multipliers"].get(zone, 1.0)
            time_multiplier = self._time_multiplier(zone)
            signal_multiplier = self._dynamic_signal_multiplier(zone, dynamic_signals, event_context)
            demand = int(base_demand * event_multiplier * time_multiplier * signal_multiplier)
            zone_flow_plan[zone] = self._estimate_zone_flow(zone, demand, event_context, dynamic_signals)
        return zone_flow_plan

    def _estimate_zone_flow(self, zone, demand, event_context, dynamic_signals):
        total = self.state[zone]["total_slots"]
        occupied = self.state[zone]["occupied"]
        occupancy_ratio = occupied / total if total else 0

        if occupancy_ratio > 0.88:
            raw_entry = int(demand * self.rng.uniform(0.08, 0.16))
        elif occupancy_ratio > 0.72:
            raw_entry = int(demand * self.rng.uniform(0.14, 0.24))
        else:
            raw_entry = int(demand * self.rng.uniform(0.20, 0.36))

        event_exit_bias = 0.02 if zone == event_context["focus_zone"] else 0.0
        if occupancy_ratio > 0.78:
            exit_count = int(occupied * self.rng.uniform(0.18, 0.30 + event_exit_bias))
        else:
            exit_count = int(occupied * self.rng.uniform(0.08, 0.16 + event_exit_bias))

        if dynamic_signals.get("blocked_zone") == zone:
            raw_entry = int(raw_entry * self.config["blocked_entry_multiplier"])
            exit_count = min(occupied, int(exit_count * self.config["blocked_exit_multiplier"]) + 2)
        if dynamic_signals.get("vip_reserve_zone") == zone:
            raw_entry = max(0, raw_entry - dynamic_signals.get("vip_reserved_slots", 0))
        if dynamic_signals.get("weather") == "Rain Surge" and zone in {
            event_context["focus_zone"],
            event_context["recommended_zone"],
        }:
            raw_entry = int(raw_entry * self.config["rain_multiplier"])

        return {
            "raw_entry": max(0, raw_entry),
            "exit_count": min(max(0, exit_count), occupied),
            "occupied": occupied,
            "total_slots": total,
        }

    def _apply_redirect_to_flow_plan(self, zone_flow_plan, transfer_report):
        if transfer_report["moved"] <= 0:
            return
        source_zone = transfer_report["from"]
        destination_zone = transfer_report["to"]
        zone_flow_plan[source_zone]["raw_entry"] = max(0, zone_flow_plan[source_zone]["raw_entry"] - transfer_report["moved"])
        zone_flow_plan[destination_zone]["raw_entry"] += transfer_report["moved"]

    def _apply_zone_flow_plan(self, zone_flow_plan):
        total_denied_entries = 0
        for zone in self.zones:
            zone_plan = zone_flow_plan[zone]
            total = self.state[zone]["total_slots"]
            occupied = self.state[zone]["occupied"]
            free_capacity = max(0, total - occupied)
            raw_entry = zone_plan["raw_entry"]
            exit_count = zone_plan["exit_count"]
            entry = min(raw_entry, max(0, free_capacity + exit_count))
            denied_entries = max(0, raw_entry - entry)
            total_denied_entries += denied_entries

            self.state[zone]["entry"] = max(0, entry)
            self.state[zone]["exit"] = max(0, exit_count)
            self.state[zone]["occupied"] += self.state[zone]["entry"] - self.state[zone]["exit"]
            self.state[zone]["occupied"] = max(0, min(self.state[zone]["occupied"], total))

        self._validate_internal_state()
        return self.get_state(), total_denied_entries

    def _build_dynamic_signals(self, event_context):
        weather = "Clear"
        if self.simulated_hour in {8, 9, 17, 18} and self.rng.random() < self.config["weather_trigger_probability"]:
            weather = "Rain Surge"

        queue_length = max(
            0,
            int(
                self.rng.randint(0, 3)
                + (2 if event_context.get("severity") in {"high", "critical"} else 0)
                + (1 if weather == "Rain Surge" else 0)
            ),
        )
        blocked_zone = None
        if event_context.get("severity") in {"high", "critical"} and self.rng.random() < self.config["high_severity_block_probability"]:
            blocked_zone = event_context.get("focus_zone")
        elif self.rng.random() < self.config["fallback_block_probability"]:
            blocked_zone = self.zones[self.step_count % len(self.zones)]

        vip_reserve_zone = None
        vip_reserved_slots = 0
        if self.rng.random() < self.config["vip_probability"]:
            vip_reserve_zone = event_context.get("recommended_zone")
            vip_reserved_slots = self.rng.randint(
                self.config["vip_reserved_min"],
                self.config["vip_reserved_max"],
            )

        patrol_mode = "Enhanced" if event_context.get("severity") in {"high", "critical"} else "Normal"
        if queue_length >= 5:
            patrol_mode = "Escalated"

        return {
            "weather": weather,
            "queue_length": queue_length,
            "blocked_zone": blocked_zone,
            "vip_reserve_zone": vip_reserve_zone,
            "vip_reserved_slots": vip_reserved_slots,
            "patrol_mode": patrol_mode,
        }

    def _dynamic_signal_multiplier(self, zone, dynamic_signals, event_context):
        multiplier = 1.0
        if dynamic_signals.get("weather") == "Rain Surge":
            multiplier += 0.12
        if dynamic_signals.get("queue_length", 0) >= self.config["queue_warning_threshold"] and zone == event_context.get("recommended_zone"):
            multiplier += 0.18
        if dynamic_signals.get("blocked_zone") == zone:
            multiplier *= 0.55
        if dynamic_signals.get("vip_reserve_zone") == zone:
            multiplier *= 0.9
        return max(0.35, multiplier)

    def _build_notifications(self, state, event_context, transfer_report, denied_entries, dynamic_signals):
        notifications = [
            {
                "level": "info",
                "title": f"{event_context['name']} active",
                "message": event_context["user_advisory"],
            }
        ]
        focus_zone = event_context["focus_zone"]
        recommended_zone = event_context["recommended_zone"]
        if state[focus_zone]["free_slots"] < self.config["congestion_alert_threshold"]:
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
                    "message": f"{transfer_report['moved']} vehicles were rerouted from {transfer_report['from']} to {transfer_report['to']}.",
                }
            )
        if dynamic_signals.get("queue_length", 0) >= self.config["queue_warning_threshold"]:
            notifications.append(
                {
                    "level": "warning",
                    "title": "Entry queue building",
                    "message": f"Simulated gate queue reached {dynamic_signals['queue_length']} vehicles. Pre-arrival guidance should intensify.",
                }
            )
        if dynamic_signals.get("blocked_zone"):
            notifications.append(
                {
                    "level": "error",
                    "title": f"Temporary disruption at {dynamic_signals['blocked_zone']}",
                    "message": "A simulated blockage is limiting inflow, so the agent should protect the rest of the network.",
                }
            )
        if dynamic_signals.get("vip_reserved_slots", 0) > 0:
            notifications.append(
                {
                    "level": "info",
                    "title": "Reserved parking window active",
                    "message": f"{dynamic_signals['vip_reserved_slots']} slots are being held in {dynamic_signals['vip_reserve_zone']} for a simulated priority cohort.",
                }
            )
        return notifications

    def _build_kpis(self, previous_state, new_state, transfer_report, denied_entries, dynamic_signals):
        total_capacity = sum(zone["total_slots"] for zone in new_state.values())
        total_occupied = sum(zone["occupied"] for zone in new_state.values())
        utilisation = round((total_occupied / total_capacity) * 100, 2) if total_capacity else 0.0
        search_time = round(
            self.config["search_time_base"]
            + denied_entries * self.config["search_time_denied_weight"]
            + max(0, utilisation - self.config["search_time_utilisation_threshold"]) * self.config["search_time_utilisation_weight"]
            + dynamic_signals.get("queue_length", 0) * self.config["search_time_queue_weight"]
            + (self.config["search_time_blocked_penalty"] if dynamic_signals.get("blocked_zone") else 0.0),
            2,
        )
        requested = transfer_report.get("requested", 0)
        moved = transfer_report.get("moved", 0)
        allocation_success = round((moved / requested) * 100, 2) if requested else 100.0
        free_slots = [zone["free_slots"] for zone in new_state.values()]
        balance_index = round(max(free_slots) - min(free_slots), 2) if free_slots else 0.0
        return {
            "space_utilisation_pct": utilisation,
            "estimated_search_time_min": search_time,
            "allocation_success_pct": allocation_success,
            "congestion_hotspots": sum(1 for zone in new_state.values() if zone["free_slots"] < self.config["congestion_hotspot_threshold"]),
            "balance_index": balance_index,
            "queue_length": dynamic_signals.get("queue_length", 0),
            "resilience_score": round(
                max(0.0, 100.0 - search_time * 8 - denied_entries * 1.5 - dynamic_signals.get("queue_length", 0) * 4),
                2,
            ),
        }

    def _build_environment_score(self, previous_state, new_state, kpis, transfer_report, denied_entries):
        prev_free = sum(zone["free_slots"] for zone in previous_state.values()) / max(1, len(previous_state))
        new_free = sum(zone["free_slots"] for zone in new_state.values()) / max(1, len(new_state))
        balance_bonus = max(0.0, 1.0 - kpis.get("balance_index", 0.0) / 100)
        redirect_value = 0.2 if transfer_report.get("moved", 0) > 0 else 0.0
        score = (
            (new_free - prev_free) / 10
            + balance_bonus
            + redirect_value
            - denied_entries * 0.08
            - max(0.0, kpis.get("estimated_search_time_min", 0.0) - 2.0) * 0.1
        )
        return round(score, 3)

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
        dynamic_signals,
        environment_score,
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
            "dynamic_signals": dynamic_signals,
            "notifications": notifications,
            "kpis": kpis,
            "environment_score": environment_score,
            "step_breakdown": self.explain_step_model(),
        }

    def _plan_redirect(self, action, zone_flow_plan):
        if not action or action.get("action") != "redirect":
            return {"moved": 0, "from": None, "to": None, "requested": 0, "mode": "incoming_reroute", "status": "success"}

        from_zone = action.get("from")
        to_zone = action.get("to")
        requested = max(0, int(action.get("vehicles", 0) or 0))
        if from_zone not in zone_flow_plan or to_zone not in zone_flow_plan or from_zone == to_zone:
            return {"moved": 0, "from": from_zone, "to": to_zone, "requested": requested, "mode": "incoming_reroute", "status": "invalid"}

        # Simulated realism: 15% chance of a real-world execution block during high intensity simulations
        if self.rng.random() < 0.15:
            reason = self.rng.choice(["Network latency timeout", "Security barrier unresponsive", "Dynamic signage offline"])
            return {"moved": 0, "from": from_zone, "to": to_zone, "requested": requested, "mode": "incoming_reroute", "status": "failed", "failure_reason": reason}

        source_plan = zone_flow_plan[from_zone]
        destination_plan = zone_flow_plan[to_zone]
        destination_capacity = max(
            0,
            (destination_plan["total_slots"] - destination_plan["occupied"]) + destination_plan["exit_count"],
        )
        moved = max(0, min(requested, source_plan["raw_entry"], destination_capacity))
        return {
            "moved": moved,
            "from": from_zone,
            "to": to_zone,
            "requested": requested,
            "mode": "incoming_reroute",
            "status": "success"
        }

    def _validate_zones(self, zones):
        if not zones or not isinstance(zones, list):
            raise ValueError("zones must be a non-empty list")
        if len(set(zones)) != len(zones):
            raise ValueError("zones must be unique")

    def _validate_action(self, action):
        if action is None:
            return
        if not isinstance(action, dict):
            raise ValueError("action must be a dict or None")
        if action.get("action") == "redirect":
            if action.get("from") not in self.zones or action.get("to") not in self.zones:
                raise ValueError("redirect action zones must exist in the environment")

    def _validate_internal_state(self):
        for zone in self.zones:
            payload = self.state.get(zone)
            if not isinstance(payload, dict):
                raise ValueError(f"Missing state payload for zone '{zone}'")
            total = int(payload.get("total_slots", 0))
            occupied = int(payload.get("occupied", 0))
            if total <= 0:
                raise ValueError(f"Zone '{zone}' has invalid total_slots")
            if occupied < 0 or occupied > total:
                raise ValueError(f"Zone '{zone}' has invalid occupied count")
