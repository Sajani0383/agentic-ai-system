import math
from copy import deepcopy


class BayesianAgent:
    def __init__(self, congestion_threshold=0.78, smoothing=0.15):
        self.congestion_threshold = congestion_threshold
        self.smoothing = smoothing
        self.beliefs = {}
        self.evidence_history = []

    def infer(self, state):
        self._ensure_beliefs(state)
        evidence_by_zone = {
            zone: self._extract_evidence(data)
            for zone, data in state.items()
        }
        likelihoods = {
            zone: self._likelihood_of_congestion(evidence)
            for zone, evidence in evidence_by_zone.items()
        }
        priors = {
            zone: self.beliefs.get(zone, 0.5)
            for zone in state
        }
        raw_posteriors = {
            zone: self._posterior(priors[zone], likelihoods[zone])
            for zone in state
        }
        normalized_posteriors = self._normalize(raw_posteriors)
        marginal_evidence = {
            zone: self._marginal_evidence(priors[zone], likelihoods[zone])
            for zone in state
        }

        self._update_beliefs(raw_posteriors)
        self._record_evidence(evidence_by_zone, priors, likelihoods, raw_posteriors)

        most_crowded = max(normalized_posteriors, key=normalized_posteriors.get)
        best_zone = min(normalized_posteriors, key=normalized_posteriors.get)
        free_values = [zone["free_slots"] for zone in state.values()]
        spread = max(free_values) - min(free_values) if free_values else 0
        entropy = self._entropy(normalized_posteriors)
        confidence = round(max(raw_posteriors.values()) if raw_posteriors else 0.0, 2)

        return {
            "confidence": confidence,
            "most_crowded": most_crowded,
            "best_zone": best_zone,
            "spread": spread,
            "priors": self._round_dict(priors),
            "likelihoods": self._round_dict(likelihoods),
            "marginal_evidence": self._round_dict(marginal_evidence),
            "posteriors": self._round_dict(raw_posteriors),
            "normalized_posteriors": self._round_dict(normalized_posteriors),
            "uncertainty": {
                "entropy": round(entropy, 3),
                "confidence_gap": self._confidence_gap(normalized_posteriors),
            },
            "beliefs": self._round_dict(self.beliefs),
            "bayes_rule": "P(congested|evidence) = P(evidence|congested) * P(congested) / P(evidence)",
            "evidence": evidence_by_zone,
        }

    def get_beliefs(self):
        return deepcopy(self.beliefs)

    def reset_beliefs(self):
        self.beliefs = {}
        self.evidence_history = []

    def _ensure_beliefs(self, state):
        for zone, data in state.items():
            if zone not in self.beliefs:
                occupancy = self._occupancy_ratio(data)
                self.beliefs[zone] = min(0.85, max(0.15, occupancy))

    def _extract_evidence(self, data):
        occupancy = self._occupancy_ratio(data)
        free_ratio = data.get("free_slots", 0) / max(1, data.get("total_slots", 1))
        inflow_pressure = max(0, data.get("entry", 0) - data.get("exit", 0)) / max(1, data.get("total_slots", 1))
        scarce_capacity = 1.0 - free_ratio
        return {
            "occupancy_ratio": round(occupancy, 4),
            "free_ratio": round(free_ratio, 4),
            "inflow_pressure": round(min(1.0, inflow_pressure), 4),
            "scarce_capacity": round(min(1.0, scarce_capacity), 4),
        }

    def _likelihood_of_congestion(self, evidence):
        occupancy_signal = self._sigmoid((evidence["occupancy_ratio"] - self.congestion_threshold) * 9.0)
        scarcity_signal = self._sigmoid((evidence["scarce_capacity"] - self.congestion_threshold) * 7.0)
        inflow_signal = self._sigmoid((evidence["inflow_pressure"] - 0.03) * 18.0)
        likelihood = 0.58 * occupancy_signal + 0.27 * scarcity_signal + 0.15 * inflow_signal
        return min(0.99, max(0.01, likelihood))

    def _posterior(self, prior, likelihood):
        evidence_probability = self._marginal_evidence(prior, likelihood)
        if evidence_probability <= 0:
            return prior
        return (likelihood * prior) / evidence_probability

    def _marginal_evidence(self, prior, likelihood):
        likelihood_given_not_congested = max(0.01, min(0.99, 1.0 - likelihood))
        return likelihood * prior + likelihood_given_not_congested * (1.0 - prior)

    def _normalize(self, probabilities):
        total = sum(probabilities.values())
        if total <= 0:
            return {key: 0.0 for key in probabilities}
        return {key: value / total for key, value in probabilities.items()}

    def _update_beliefs(self, posteriors):
        for zone, posterior in posteriors.items():
            previous = self.beliefs.get(zone, 0.5)
            updated = (1 - self.smoothing) * previous + self.smoothing * posterior
            self.beliefs[zone] = min(0.99, max(0.01, updated))

    def _record_evidence(self, evidence_by_zone, priors, likelihoods, posteriors):
        self.evidence_history.append(
            {
                "evidence": deepcopy(evidence_by_zone),
                "priors": deepcopy(priors),
                "likelihoods": deepcopy(likelihoods),
                "posteriors": deepcopy(posteriors),
            }
        )
        self.evidence_history = self.evidence_history[-50:]

    def _occupancy_ratio(self, data):
        total_slots = max(1, data.get("total_slots", 1))
        return data.get("occupied", 0) / total_slots

    def _entropy(self, probabilities):
        return -sum(
            probability * math.log(probability, 2)
            for probability in probabilities.values()
            if probability > 0
        )

    def _confidence_gap(self, probabilities):
        values = sorted(probabilities.values(), reverse=True)
        if len(values) < 2:
            return round(values[0], 3) if values else 0.0
        return round(values[0] - values[1], 3)

    def _sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

    def _round_dict(self, values):
        return {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in values.items()
        }
