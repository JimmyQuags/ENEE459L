from __future__ import annotations


import statistics
from typing import Any


STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12


def unknown(source: str, why: str) -> dict[str, Any]:
	return {"value": None, "source": source, "status": "unknown", "detail": why}


def measured(value: Any, source: str, **extra: Any) -> dict[str, Any]:
	return {"value": value, "source": source, "status": "ok", **extra}


def is_stationary(samples: list[float]) -> dict[str, Any]:
	if len(samples) < MIN_SAMPLES_FOR_STATIONARITY:
		return unknown("stationarity", "too few samples to divide into thirds")

	overall_median = statistics.median(samples)
	if overall_median <= 0:
		return unknown("stationarity", "median is not positive")

	third_size = len(samples) // 3
	first_third_median = statistics.median(samples[:third_size])
	last_third_median = statistics.median(samples[-third_size:])
	drift = last_third_median - first_third_median
	relative_drift = abs(drift) / overall_median

	if drift > 0:
		direction = "slower"
	elif drift < 0:
		direction = "faster"
	else:
		direction = "flat"

	return measured(
		relative_drift <= STATIONARITY_TOL,
		"stationarity",
		first_third_median_ms=round(first_third_median, 4),
		last_third_median_ms=round(last_third_median, 4),
		drift_ms=round(drift, 4),
		drift_relative=round(relative_drift, 4),
		direction=direction,
		tolerance=STATIONARITY_TOL,
	)
