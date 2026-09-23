from __future__ import annotations


import statistics
from pathlib import Path
from typing import Any


from bench import Bench, measured, read_first, read_text, unknown


import json


# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5


# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5


# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)


# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0


# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10


# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20


# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12


THERMAL_ZONES = "sys/devices/virtual/thermal"


POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)


GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)


CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"




# ===========================================================================
# 1. The loop (START EDITING AFTER HERE)
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    elapsed_times = []


    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        elapsed_times.append((end - start) / 1_000_000.0)


    return elapsed_times




def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("warmup boundary", "too few samples")


    settled_rate = statistics.median(samples[len(samples) // 2 :])
    if settled_rate <= 0:
        return unknown("warmup boundary", "settled median is not positive")


    threshold = settled_rate * (1 + WARMUP_TOL)
    discarded = 0
    for sample in samples:
        if sample <= threshold:
            break
        discarded += 1


    return measured(
        discarded,
        "leading prefix above (1 + 0.5) x median of the run's second half",
        settled_rate_ms=round(settled_rate, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )






def summarize(samples: list[float]) -> dict[str, Any]:
    metric_keys = ("mean", "std", "min", "max", "p50", "p95", "p99")
    if not samples:
        return {"n": 0, **{key: None for key in metric_keys}}


    ordered = sorted(samples)
    count = len(ordered)


    def percentile(percent: int) -> float:
        position = (count - 1) * percent / 100
        index = int(position)
        if index == count - 1:
            return ordered[index]
        fraction = position - index
        return ordered[index] + fraction * (ordered[index + 1] - ordered[index])


    return {
        "n": count,
        "mean": round(statistics.fmean(ordered), 4),
        "std": round(statistics.stdev(ordered), 4) if count > 2 else 0.0,
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "p50": round(percentile(50), 4),
        "p95": round(percentile(95), 4),
        "p99": round(percentile(99), 4),
    }


def is_multimodal(samples: list[float]) -> dict[str, Any]:
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown("multimodal distribution", "not enough samples")


    ordered = sorted(samples)
    trim = len(ordered) // 20
    trimmed = ordered[trim : len(ordered) - trim]
    gaps = [right - left for left, right in zip(trimmed, trimmed[1:])]
    typical_gap = statistics.median(gaps)
    if typical_gap <= 0:
        return unknown("trimmed adjacent gaps", "timer resolution is too coarse")


    widest_index = max(range(len(gaps)), key=gaps.__getitem__)
    widest_gap = gaps[widest_index]
    ratio = widest_gap / typical_gap
    split_value = trimmed[widest_index]
    left = [sample for sample in ordered if sample <= split_value]
    right = [sample for sample in ordered if sample > split_value]
    left_share = len(left) / len(ordered)
    right_share = len(right) / len(ordered)


    return measured(
        ratio >= MULTIMODAL_GAP_RATIO
        and left_share >= MIN_MODE_FRACTION
        and right_share >= MIN_MODE_FRACTION,
        "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 5),
        typical_gap_ms=round(typical_gap, 5),
        modes=[
            {
                "n": len(left),
                "share": round(left_share, 2),
                "median_ms": round(statistics.median(left), 4),
            },
            {
                "n": len(right),
                "share": round(right_share, 2),
                "median_ms": round(statistics.median(right), 4),
            },
        ],
    )


# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================




def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0:
        detail = result.error or f"command exited with status {result.returncode}"
        return unknown("nvpmodel -q", detail)


    lines = result.stdout.splitlines()
    mode_name: str | None = None
    mode_index: int | None = None
    for index, line in enumerate(lines):
        if "NV Power Mode:" not in line:
            continue
        mode_name = line.split("NV Power Mode:", 1)[1].strip()
        if mode_name.startswith("MODE_"):
            mode_name = mode_name[5:]
        if index + 1 < len(lines):
            following = lines[index + 1].strip()
            try:
                mode_index = int(following)
            except ValueError:
                mode_index = None
        break


    if not mode_name or mode_index is None:
        return unknown("nvpmodel -q", "unable to parse active power mode and index")


    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    clock_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    if not minimum or not maximum:
        jetson_clocks = None
        clock_finding = unknown(clock_source, "could not read both CPU frequency limits")
    else:
        jetson_clocks = minimum == maximum
        clock_finding = measured(
            f"scaling_min_freq={minimum}, scaling_max_freq={maximum}",
            clock_source,
        )


    return measured(
        mode_name,
        "nvpmodel -q",
        mode_index=mode_index,
        jetson_clocks=jetson_clocks,
        jetson_clocks_source=clock_finding,
    )






def probe_telemetry(bench: Bench) -> dict[str, Any]:
    thermal_source = f"{THERMAL_ZONES}/*/temp"
    thermal_root = Path(bench.telemetry) / THERMAL_ZONES
    temperatures: list[tuple[float, str]] = []
    try:
        zone_paths = sorted(path for path in thermal_root.iterdir() if path.is_dir())
    except OSError:
        zone_paths = []


    for zone_path in zone_paths:
        raw_temp = read_text(bench.telemetry, f"{THERMAL_ZONES}/{zone_path.name}/temp")
        if raw_temp is None:
            continue
        try:
            raw_temp_millidegrees = float(raw_temp)
        except ValueError:
            continue
        if raw_temp_millidegrees <= -1000:
            continue
        temp_c = raw_temp_millidegrees / 1000.0
        zone_name = read_text(
            bench.telemetry, f"{THERMAL_ZONES}/{zone_path.name}/type"
        ) or zone_path.name
        temperatures.append((temp_c, zone_name))


    if temperatures:
        peak_temperature, peak_zone = max(temperatures)
        temperature_finding = measured(
            round(peak_temperature, 2),
            thermal_source,
            zone=peak_zone,
            zones_read=len(temperatures),
        )
    else:
        temperature_finding = unknown(thermal_source, "no valid thermal zone entries")


    power_source = " | ".join(POWER_RAIL_CANDIDATES)
    power_result = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_result is None:
        power_finding = unknown(
            power_source, "none of the documented INA3221 rail paths could be read"
        )
    else:
        power_path, raw_power = power_result
        try:
            power_finding = measured(int(raw_power), power_path)
        except ValueError:
            power_finding = unknown(power_path, "INA3221 power value is not an integer")


    gpu_source = " | ".join(GPU_LOAD_CANDIDATES)
    gpu_result = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_result is None:
        gpu_finding = unknown(gpu_source, "none of the documented GPU load paths could be read")
    else:
        gpu_path, raw_load = gpu_result
        try:
            gpu_finding = measured(
                round(int(raw_load) / 10.0, 1),
                gpu_path,
                units="per-mille / 10",
            )
        except ValueError:
            gpu_finding = unknown(gpu_path, "GPU load value is not an integer")


    return {
        "temperature_c": temperature_finding,
        "power_mw": power_finding,
        "gpu_utilization_percent": gpu_finding,
    }


# for debugging - uncomment the following lines for debugging.
#if __name__ == "__main__":
    #env = Bench.real()
    #out = find_warmup_boundary(samples)
    #print(out)


# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()


    # get your samples
    samples = run_timed_iterations(env, repeats=100)


    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }


    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)


    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
