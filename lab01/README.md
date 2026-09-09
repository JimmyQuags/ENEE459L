# Lab 01 — What is this machine, and how do you know?

**Platform health: probing NVMe boot, temperature, link speed, and memory—with evidence.**

## The Question

A Jetson can have an NVMe drive fitted, visible to the kernel, and completely unused, because the operating system boots from the SD card. It works. It passes casual inspection. It also runs every benchmark this semester against storage an order of magnitude slower than the drive sitting unused in its slot, and every number it produces is quietly, unreproducibly wrong. Can you tell the two apart?

## What You Will Build

Seven standalone probes reading sysfs, /proc, and CLI tools:

- **`probe_module_model`**: The CPU module name from `/proc/device-tree/model`.
- **`probe_memory_total_kb`**: Total RAM in kB (always less than 8 GB on Orin Nano).
- **`probe_root_source`**: Which device the root filesystem boots from (NVMe, SD, or other).
- **`probe_nvme_present`**: Whether an NVMe drive exists in `/sys/block/nvme0n1` (fitted ≠ booted-from).
- **`probe_pcie_link`**: Two PCIe numbers—what the link *can* do (capability) and what it *did* do (negotiated).
- **`probe_thermal_zones`**: All thermal zones in millidegrees Celsius, divided by 1000 to get degrees.
- **`probe_power_mode`**: Current NVIDIA power mode (MAXN SUPER, 15W, etc.) from `nvpmodel`.

All reads use a `root` parameter so tests can inject fake filesystems. When you cannot read something, you return `unknown(source, why)`, not a plausible default.

## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **probe_module_model**: slide 6
2. **probe_memory_total_kb**: slide 8
3. **probe_root_source**: slide 10.
4. **probe_nvme_present**: slide 11.
5. **probe_pcie_link**: slide 13.
6. **probe_thermal_zones**: slide 15
7. **probe_power_mode**: slide 17

## How to Run

```bash
cd lab01/

# Generate the report on the board
sudo python probes.py
```

After completing the code, please validate the resulting JSON output file against `sample_system_report.json` to ensure it conforms to the expected format before submission.

## How to Debug your code

There are two ways to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 
2. Another way is to just the output by using command `print(out)` where out is output of any function. 

## How to save your work

For saving your work, you create a new branch named `solution` by running the following command.
```bash
git switch -c solution
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution
```

## The Two Questions This Lab Answers

**Is the board booting from NVMe or SD?** The `probe_root_source` verdict `boots_from_nvme` checks this. If it says your board boots from `/dev/mmcblk0p1`, the semester's benchmarks are running against the slow card, not the fast drive.

**Did the PCIe link negotiate to Gen3 or Gen4?** The drive's box says Gen4. The slot is Gen3 ×4. `probe_pcie_link` returns both the link's capability and what it actually negotiated. The gap is the lesson: a component's spec sheet is an upper bound, not a statement about your system.

## Analysis

1. **the_module_exists**: The CPU module name is readable.
2. **the_memory_matches_the_label**: Total memory ≥ 6 GB (Orin Nano should report ~7.6 GB).
3. **boots_from_nvme**: Root filesystem is on `/dev/nvme0n1`, not `/dev/mmcblk0p1`.
4. **nvme_is_fitted**: An NVMe drive exists and reports a model.
5. **the_link_is_negotiated**: Both PCIe capability and negotiated speed are known.
6. **the_link_is_gen3**: The link negotiated at Gen3, not higher (due to board wiring).
7. **the_thermal_zones_are_readable**: At least one thermal zone reports a temperature.
8. **the_power_mode_is_set**: Power mode is readable and is MAXN SUPER (or an expected lower mode).

