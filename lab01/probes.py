"""Probes — read what the machine says about itself.

INSTRUCTOR SOLUTION. Do not distribute. The student copy of this file has the
body of every function below replaced by `raise NotImplementedError`.

Every probe takes a `root` argument and reads nothing outside it. That is not
decoration: it is what makes this lab gradeable without twenty boards on a
desk, and it is the reason the test suite can present a fake SD-booted machine
and check that the student's code notices. Code that hardcodes "/" cannot be
tested, and a measurement you cannot test is a measurement you cannot trust —
which is the whole argument of Lecture 01, applied to the student's own code.

Each probe returns a dict with, at minimum, a `value` and a `source` key. The
`source` is the path or command the value came from. A number without its
provenance is not evidence, so the report format refuses to carry one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
import pdb
import json

# ---------------------------------------------------------------------------
# Small helpers. These are given to students; the exercise is the probes.
# ---------------------------------------------------------------------------


def read_text(root: Path, rel: str) -> str | None:
    """Read `root/rel`, returning None if it is missing or unreadable.

    Missing is a normal outcome here, not an error: a devkit with no NVMe
    genuinely has no /sys/block/nvme0n1, and the report needs to say so rather
    than crash.
    """
    p = Path(root) / rel.lstrip("/")
    try:
        return p.read_text(errors="replace").strip("\x00").strip()
    except (OSError, UnicodeDecodeError):
        return None


def run(cmd: list[str]) -> str | None:
    """Run a command, returning stdout, or None if it is absent or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def unknown(source: str, why: str) -> dict[str, Any]:
    """The value this lab returns when it cannot determine something.

    Note what this is not: it is not None threaded through the report, and it
    is not a plausible default. It is an explicit record that the probe ran and
    failed, carrying the reason. Assignment 1's rubric gives credit for these.
    """
    return {"value": None, "source": source, "status": "unknown", "detail": why}

# LnkSta/LnkCap lines look like:
#   LnkSta: Speed 8GT/s, Width x4, TrErr- Train- SlotClk+ DLActive- ...
#   LnkCap: Port #0, Speed 16GT/s, Width x4, ASPM L1, Exit Latency L1 <64us
_SPEED_RE = re.compile(r"Speed\s+([\d.]+)GT/s")
_WIDTH_RE = re.compile(r"Width\s+x(\d+)")

# PCIe generation by per-lane transfer rate. Gen3 is 8 GT/s; the Orin Nano
# devkit's M.2 Key-M slot is wired Gen3 x4, so a Gen4 drive reporting 16 GT/s
# capability and 8 GT/s status is behaving correctly, not underperforming.
#
# Keyed by float, not by the string lspci printed. Keying by string means
# deciding whether "8", "8.0" and "08" are the same rate, and the obvious
# normalisation — stripping trailing zeros and dots — silently turns 20 into 2.
_GEN_BY_GTS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


def _parse_link_line(line: str) -> dict[str, Any]:
    speed = _SPEED_RE.search(line)
    width = _WIDTH_RE.search(line)
    gts = float(speed.group(1)) if speed else None
    return {
        "raw": line.strip(),
        "gts": gts,
        "width": int(width.group(1)) if width else None,
        "gen": _GEN_BY_GTS.get(gts) if gts is not None else None,
    }

def generate_interpretation_string(neg_speed, cap_speed):
    if cap_speed > neg_speed:
        interpretation = (
            f"drive capable of Gen{capability['gen']}, link running at "
            f"Gen{negotiated['gen']} — expected on this carrier board, "
            "whose M.2 Key-M slot is wired Gen3 x4"
        )
    else:
        interpretation = (
            f"link running at its full capability, Gen{negotiated['gen']} "
            f"x{negotiated['width']}"
        )
    return interpretation
    


# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

## An example code.
def probe_module_model(root: Path = Path("/")) -> dict[str, Any]:
    """Which board is this?

    The device tree model string is the most trustworthy identity on a Jetson —
    it comes from the hardware description the bootloader handed the kernel,
    not from anything installed afterwards.
    """

    # step 1: Read the raw null-terminated text from /proc/device-tree/model
    src = "/proc/device-tree/model"
    raw = read_text(root, src)

    # if unable to read, return an empty dictionary by calling unknown().
    if not raw:
        return unknown(src, "device tree model node absent — not a Jetson, or /proc not mounted")
    
    # step 2: strip null bytes and whitespace from raw string
    raw = raw.rstrip("\x00").strip()
    return {"value": raw, "source": src, "status": "ok"}



## todo by students START HERE
# Probe 1
def probe_memory_total_kb(root: Path = Path("/")) -> dict[str, Any]:
    """How much memory is there, in kB, as the kernel counts it?

    This will read a little under 8 GB on an 8 GB board. That gap is not a
    fault: the carveout for the GPU and other hardware is taken before Linux
    ever sees the pool. Students are expected to notice and to explain it in
    their report rather than round it up.
    """

    # My Code:
    src = "/proc/meminfo"
    raw = read_text(root, src)

    if not raw:
        return unknown(src, "meminfo node absent — /proc not mounted")

    # /proc/meminfo typically starts with: MemTotal: 8012344 kB
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", raw)
    if not m:
        return unknown(src, f"unable to parse MemTotal from contents of {src}")

    
    return {"value": int(m.group(1)), "source": src, "status": "ok"}

# Probe 2
def probe_root_source(root: Path = Path("/")) -> dict[str, Any]:
    """What device is the root filesystem actually mounted from?

    This is the probe the lab is built around. A unit that boots from the SD
    card works, boots, and passes every casual inspection — and then runs the
    semester's benchmarks against a card an order of magnitude slower than the
    NVMe sitting unused in the slot. The failure is silent, which is exactly
    why it has to be a command rather than an assumption.

    /proc/mounts is preferred over `findmnt` because it needs no external
    binary and no elevation, and because it is what findmnt reads anyway.
    """

    # My Code: 
    src = "/proc/mounts"
    raw = read_text(root, src)

    if not raw:
        return unknown(src, "mounts table absent — /proc not mounted")

    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "/":
            device = parts[0]
            return {"value": device, "source": src, "status": "ok"}


    return unknown(src, "no root mount entry found in mount table")


# Probe 3
def probe_nvme_present(root: Path = Path("/")) -> dict[str, Any]:
    """Is there an NVMe device visible as a block device at all?

    Deliberately separate from probe_root_source. A machine can have an NVMe
    fitted and still boot from the SD card, and telling those two states apart
    is what lets the troubleshooting tree in the lab guide send a student to
    the right branch.
    """

    # My Code:
    # Define the primary sysfs directory where NVMe block devices register
    block_dir_rel = "sys/block/nvme0n1"
    nvme_path = Path(root) / block_dir_rel

    # Check if block device path exists in target root filesystem
    if not nvme_path.exists():
        return {
            "value": False,
            "model": None,
            "source": f"/{block_dir_rel}",
            "status": "ok",
        }

    # If present, read model name reported by device firmware
    model_src = f"/{block_dir_rel}/device/model"
    model_name = read_text(root, model_src)


    return {
        "value": True,
        "model": model_name if model_name else "Unknown NVMe",
        "source": f"/{block_dir_rel}",
        "status": "ok",
    }


# Probe 4
def probe_pcie_link(root: Path = Path("/"), lspci_output: str | None = None) -> dict[str, Any]:
    """What did the PCIe link negotiate, and what was it capable of?

    Two numbers, not one. The gap between them is the lab's worked example of
    spec sheet against measured reality: a Gen4 drive in a Gen3 slot advertises
    16 GT/s and settles at 8 GT/s, and a student who reports only the second
    number has recorded a fact without recording what it means.

    `lspci_output` exists so the tests can drive this without root or hardware.
    In normal use it is None and the probe shells out.
    """
    src = "lspci -vvv"

    # My Code:
    # Step 1: Obtain lspci output (use provided mock string if available, else run command)
    if lspci_output is not None:
        raw = lspci_output
    else:
        # Request verbose output to capture LnkCap and LnkSta fields
        raw = run(["lspci", "-vvv"])

    # If execution failed or returned no stdout, return unknown dictionary state
    if not raw:
        return unknown(src, "lspci command failed, missing, or returned empty output")

    # Step 2: Extract LnkSta (negotiated status) and LnkCap (device capability) lines
    lnksta_match = re.search(r"LnkSta:\s*(.*)", raw)
    lnkcap_match = re.search(r"LnkCap:\s*(.*)", raw)

    # If either PCIe link descriptor line is missing, return unknown status
    if not lnksta_match or not lnkcap_match:
        return unknown(src, "could not locate LnkSta or LnkCap in lspci output")

    # Step 3: Parse raw lines into structured dicts with gts, width, and gen
    negotiated = _parse_link_line(lnksta_match.group(1))
    capability = _parse_link_line(lnkcap_match.group(1))

    # Step 4: Generate explanation string comparing capability vs negotiated link rate
    # (Frequent error spot)
    # Pass the numeric GT/s speed values, e.g., 8.0 vs 16.0
    # Pass the integer generation numbers, e.g., 3 vs 4
    interpretation = generate_interpretation_string(
        negotiated["gen"], 
        capability["gen"]
    )



    # Return structured dict containing negotiated speed, capability, and interpretation
    return {
        "value": f"Gen{negotiated.get('gen')} x{negotiated.get('width')}",
        "negotiated": negotiated,
        "capability": capability,
        "interpretation": interpretation,
        "source": src,
        "status": "ok",
    }


# Probe 5
def probe_thermal_zones(root: Path = Path("/")) -> dict[str, Any]:
    """Every thermal zone the kernel exposes, in degrees C.

    Sysfs reports millidegrees. The division by 1000 is the entire trap: a
    report claiming the board idles at 43,000 degrees has been submitted more
    than once, and it is a good, cheap lesson in reading units before reading
    numbers.
    """

    # My Code:
    src = "/sys/class/thermal"
    thermal_dir = Path(root) / src.lstrip("/")

    # If thermal class directory does not exist or /sys is not mounted
    if not thermal_dir.exists():
        return unknown(src, "thermal class directory absent — /sys not mounted")

    zones: dict[str, float] = {}

    # Iterate over all thermal_zone* subdirectories (e.g., thermal_zone0, thermal_zone1)
    for zone_path in sorted(thermal_dir.glob("thermal_zone*")):
        zone_name = zone_path.name  # e.g., "thermal_zone0"

        # Read thermal zone type (e.g., "CPU-therm", "GPU-therm")
        type_rel = f"{src}/{zone_name}/type"
        zone_type = read_text(root, type_rel)

        # Read raw millidegree temperature string from sysfs
        temp_rel = f"{src}/{zone_name}/temp"
        temp_raw = read_text(root, temp_rel)

        # Skip this zone if either file could not be read
        if zone_type is None or temp_raw is None:
            continue

        try:
            # Sysfs reports temperature in millidegrees C ,  /1000 to get degrees C
            temp_c = int(temp_raw) / 1000.0
            zones[zone_type] = temp_c
        except ValueError:
            # Handle malformed temperature strings gracefully
            continue

    # If no thermal zones successfully parsed, report unknown state
    if not zones:
        return unknown(src, "no thermal zones found or readable under sysfs")

    return {
        "value": zones,
        "zones": zones,
        "source": src,
        "status": "ok",
    }


# Probe 6
def probe_power_mode(root: Path = Path("/"), nvpmodel_output: str | None = None) -> dict[str, Any]:
    """Which nvpmodel power mode is active?

    Recorded on every artifact this course produces. Lecture 01 slide 24 is
    the argument for why: two students reporting different throughput for the
    same model are usually reporting different power modes, and without this
    field there is no way to find that out after the fact.
    """
    src = "nvpmodel -q"

    # My Code:
    # Step 1: Get nvpmodel status output (use injected string if testing, else execute binary)
    if nvpmodel_output is not None:
        raw = nvpmodel_output
    else:
        # Run `nvpmodel -q` to query active power profile
        raw = run(["nvpmodel", "-q"])

    # If command failed, missing from PATH or returned nothing
    if not raw:
        return unknown(src, "nvpmodel command failed, missing, or returned empty output")

    # Step 2: Extract power mode ID and mode name from output
    # Sample nvpmodel -q output:
    # NVPMWARN: ...
    # NVPOWERMODE: 15W
    # or:
    # NV Power Mode: MAXN
    # 0
    mode_id_match = re.search(r"NVPOWERMODE:\s*(\S+)", raw) or re.search(r"Power Mode:\s*(\S+)", raw)
    id_num_match = re.search(r"^\s*(\d+)\s*$", raw, re.MULTILINE)

    # If can't parse current power mode name/identifier
    if not mode_id_match:
        return unknown(src, f"unable to parse power mode from nvpmodel output: {raw!r}")

    mode_name = mode_id_match.group(1).strip()
    mode_id = int(id_num_match.group(1)) if id_num_match else None

    # Step 3: Construct diagnostic report dictionary
    return {
        "value": mode_name,
        "mode_id": mode_id,
        "source": src,
        "status": "ok",
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
#     out = probe_power_mode()
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    report = {
        "module_model": probe_module_model(),
        "memory_total_kb": probe_memory_total_kb(),
        "root_source": probe_root_source(),
        "nvme_present": probe_nvme_present(),
        "pcie_link": probe_pcie_link(),
        "thermal_zones": probe_thermal_zones(),
        "power_mode": probe_power_mode(),
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

