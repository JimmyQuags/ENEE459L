from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from pathlib import Path
import json

SCHEMA_VERSION = "1.0"
SCALE_BYTES = 4
ZERO_BYTES = 4
CLASSES = 4
_LCG_A = 1103515245
_LCG_C = 12345
_LCG_M = 2 ** 31
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

PRECISIONS = ("fp32", "fp16", "int8")
GRANULARITIES = ("per_tensor", "per_channel")
CLIP_METHODS = ("minmax", "percentile", "mse")
CALIBRATIONS = ("all", "two_class", "first_8", "one_class")
DEFAULT_BITS = 8
DEFAULT_PERCENTILE = 99.9
MSE_CANDIDATES = 32


class TensorError(ValueError):
    pass


def elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def dtype_bytes(dtype: str) -> float:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise TensorError(
            f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}"
        ) from None


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    data: tuple[float, ...]
    dtype: str = "fp32"

    def __post_init__(self) -> None:
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise TensorError(f"{self.name}: shape {self.shape} is not a shape")
        if self.dtype not in DTYPE_BYTES:
            raise TensorError(f"{self.name}: unknown dtype {self.dtype!r}")
        if len(self.data) != elements(self.shape):
            raise TensorError(
                f"{self.name}: shape {self.shape} wants {elements(self.shape)} "
                f"values, got {len(self.data)}"
            )

    @property
    def parameters(self) -> int:
        return elements(self.shape)

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1


@dataclass(frozen=True)
class Sample:
    layer: str
    cls: int
    data: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.data:
            raise TensorError(f"{self.layer}: an empty activation sample")
        if not 0 <= self.cls < CLASSES:
            raise TensorError(
                f"{self.layer}: class {self.cls} is not in 0..{CLASSES - 1}"
            )


def _lcg(seed: int) -> Iterable[float]:
    x = seed % _LCG_M
    while True:
        x = (_LCG_A * x + _LCG_C) % _LCG_M
        yield (x / _LCG_M) * 2.0 - 1.0


def _channel_scale(channel: int, total: int) -> float:
    return 0.20 + 0.90 * (channel + 1) / total


def _class_scale(class_index: int) -> float:
    return (1.0, 2.0, 3.5, 5.0)[class_index % CLASSES]


def materialise(name: str, shape: Sequence[int], seed: int,
                dtype: str = "fp32") -> Tensor:
    shape = tuple(int(dimension) for dimension in shape)
    count = elements(shape)
    stride = elements(shape[1:]) if len(shape) > 1 else 1
    generator = _lcg(seed)
    values = []
    for index in range(count):
        value = next(generator)
        values.append(
            round(value * _channel_scale(index // stride, shape[0]), 8)
        )
    return Tensor(name=name, shape=shape, data=tuple(values), dtype=dtype)


def materialise_samples(layer: str, width: int, count: int,
                        seed: int) -> tuple[Sample, ...]:
    generator = _lcg(seed)
    samples: list[Sample] = []
    for index in range(int(count)):
        class_index = index % CLASSES
        scale = _class_scale(class_index)
        values = tuple(
            round(abs(next(generator)) * scale, 8)
            for _ in range(int(width))
        )
        samples.append(Sample(layer=layer, cls=class_index, data=values))
    return tuple(samples)


def load_model(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TensorError(f"{path}: not JSON — {exc}") from None
    except OSError as exc:
        raise TensorError(f"{path}: cannot read — {exc}") from None

    if not isinstance(document, dict):
        raise TensorError(
            f"{path}: top level is {type(document).__name__}, want object"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise TensorError(
            f"{path}: unsupported schema_version "
            f"{document.get('schema_version')!r}"
        )
    for key in ("name", "seed", "tensors"):
        if key not in document:
            raise TensorError(f"{path}: no {key!r}")
    if not isinstance(document["tensors"], list) or not document["tensors"]:
        raise TensorError(f"{path}: 'tensors' must be a non-empty list")

    dtype = document.get("dtype", "fp32")
    dtype_bytes(dtype)
    base_seed = int(document["seed"])
    tensors = []
    for index, specification in enumerate(document["tensors"]):
        if not isinstance(specification, dict):
            raise TensorError(f"{path}: tensor {index} is not an object")
        if "name" not in specification or "shape" not in specification:
            raise TensorError(f"{path}: tensor {index} needs 'name' and 'shape'")
        shape = specification["shape"]
        if not isinstance(shape, list) or not shape:
            raise TensorError(
                f"{path}: tensor {specification.get('name')} has no shape"
            )
        tensors.append(
            materialise(
                specification["name"],
                shape,
                base_seed + 7919 * index,
                dtype,
            )
        )

    activations: list[Sample] = []
    for index, specification in enumerate(document.get("activations", [])):
        if not isinstance(specification, dict):
            raise TensorError(f"{path}: activations[{index}] is not an object")
        for key in ("layer", "width", "samples"):
            if key not in specification:
                raise TensorError(
                    f"{path}: activations[{index}] has no {key!r}"
                )
        if int(specification["width"]) <= 0 or int(specification["samples"]) <= 0:
            raise TensorError(
                f"{path}: activations[{index}] wants positive width and count"
            )
        activations.extend(
            materialise_samples(
                str(specification["layer"]),
                int(specification["width"]),
                int(specification["samples"]),
                base_seed + 104729 * (index + 1),
            )
        )

    return {
        "name": document["name"],
        "source": document.get("source", str(path)),
        "dtype": dtype,
        "seed": base_seed,
        "tensors": tuple(tensors),
        "activations": tuple(activations),
        "notes": list(document.get("notes", [])),
    }


def total_parameters(tensors: Sequence[Tensor]) -> int:
    return sum(tensor.parameters for tensor in tensors)


# helper class and functions to carry the quantized tensor and its metadata around together
@dataclass(frozen=True)
class Quantized:
    name: str
    shape: tuple[int, ...]
    codes: tuple[int, ...]
    scales: tuple[float, ...]
    zeros: tuple[int, ...]
    bits: int
    granularity: str
    symmetric: bool
    clipped: int = 0

    @property
    def parameters(self) -> int:
        return elements(self.shape)

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def scale_count(self) -> int:
        """How many (scale, zero) pairs this tensor carries on disk."""
        return len(self.scales)

    @property
    def clipped_fraction(self) -> float:
        return round(self.clipped / len(self.codes), 8) if self.codes else 0.0

def to_fp16(t: Tensor) -> Tensor:
    out = []
    for v in t.data:
        if not -65504.0 <= v <= 65504.0:
            raise TensorError(
                f"{t.name}: {v} is outside binary16's range; FP16 would store "
                "infinity and every number after this one would be a lie"
            )
        out.append(round(struct.unpack("e", struct.pack("e", float(v)))[0], 8))
    return Tensor(name=t.name, shape=t.shape, data=tuple(out), dtype="fp16")


def as_tensor(name: str, values: Sequence[float]) -> Tensor:
    vals = tuple(float(v) for v in values)
    if not vals:
        raise TensorError(f"{name}: no values")
    return Tensor(name=name, shape=(len(vals),), data=vals, dtype="fp32")


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _qmin_qmax(bits: int, symmetric: bool) -> tuple[int, int]:
    bit_count = int(bits)
    if bit_count < 2:
        raise TensorError(f"{bit_count} bits cannot carry a range")
    if symmetric:
        return -(2 ** (bit_count - 1)) + 1, 2 ** (bit_count - 1) - 1
    return -(2 ** (bit_count - 1)), 2 ** (bit_count - 1) - 1


def _scale_zero(rmin: float, rmax: float, bits: int,
                symmetric: bool) -> tuple[float, int]:
    qmin, qmax = _qmin_qmax(bits, symmetric)
    lo, hi = float(min(rmin, rmax)), float(max(rmin, rmax))
    if symmetric:
        magnitude = max(abs(lo), abs(hi))
        lo, hi = -magnitude, magnitude
    else:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        return 1.0, 0
    scale = (hi - lo) / (qmax - qmin)
    if symmetric:
        return scale, 0
    zero = _round_half_up(qmin - lo / scale)
    return scale, max(qmin, min(qmax, zero))


def _channel_slice(tensor: Tensor, channel: int) -> tuple[int, int]:
    if not 0 <= channel < tensor.channels:
        raise TensorError(
            f"{tensor.name}: no channel {channel} in {tensor.channels}"
        )
    stride = tensor.channel_stride
    return channel * stride, (channel + 1) * stride


def _range_of(values: Sequence[float], method: str = "minmax",
              percentile: float = DEFAULT_PERCENTILE,
              bits: int = DEFAULT_BITS) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        raise TensorError("no values to take a range over")
    if method not in CLIP_METHODS:
        raise TensorError(
            f"unknown clip method {method!r}; known: {list(CLIP_METHODS)}"
        )
    if method == "minmax":
        return min(values), max(values)
    if method == "percentile":
        if not 0.0 < float(percentile) <= 100.0:
            raise TensorError(
                f"percentile {percentile} is not in (0, 100]"
            )
        ordered = sorted(values)
        count = len(ordered)
        low_index = min(
            count - 1,
            int(math.floor((100.0 - float(percentile)) / 200.0 * count)),
        )
        high_index = max(0, count - 1 - low_index)
        return (
            ordered[min(low_index, high_index)],
            ordered[max(low_index, high_index)],
        )

    low, high = min(values), max(values)
    qmin, qmax = _qmin_qmax(bits, False)
    levels = float(qmax - qmin)
    best: tuple[float, float, float] | None = None
    for candidate in range(1, MSE_CANDIDATES + 1):
        fraction = float(candidate) / MSE_CANDIDATES
        candidate_low = low * fraction
        candidate_high = high * fraction
        if candidate_high <= candidate_low:
            continue
        step = (candidate_high - candidate_low) / levels
        error = sum(
            (value - min(candidate_high, max(candidate_low, value))) ** 2
            for value in values
        )
        error += len(values) * step * step / 12.0
        if best is None or error < best[0]:
            best = (error, candidate_low, candidate_high)
    if best is None:
        raise TensorError("the MSE search found no usable clip")
    return best[1], best[2]


def select_calibration(activations: Sequence[Sample],
                       policy: str = "all") -> tuple[float, ...]:
    acts = tuple(activations)
    if policy == "all":
        chosen = acts
    elif policy == "two_class":
        chosen = tuple(s for s in acts if s.cls in (0, 1))
    elif policy == "first_8":
        chosen = acts[:8]
    elif policy == "one_class":
        chosen = tuple(s for s in acts if s.cls == 0)
    else:
        raise TensorError(
            f"unknown calibration policy {policy!r}; known: {list(CALIBRATIONS)}"
        )
    if not chosen:
        raise TensorError(f"calibration policy {policy!r} selected no samples")
    return tuple(v for s in chosen for v in s.data)


########################################
# start filling in the code from here.
#########################################

# 1
def quantize_tensor(t: Tensor, bits: int = DEFAULT_BITS,
                    granularity: str = "per_tensor", symmetric: bool = True,
                    clip: str = "minmax",
                    percentile: float = DEFAULT_PERCENTILE,
                    given_range: tuple[float, float] | None = None) -> Quantized:
    
    if granularity not in GRANULARITIES:
        raise TensorError(
            f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}"
        )
    
    if clip not in CLIP_METHODS:
        raise TensorError(f"unknown clip method {clip!r}; known: {list(CLIP_METHODS)}")
    qmin, qmax = _qmin_qmax(bits, symmetric)
    data = t.data

    if granularity == "per_channel":
        groups = [_channel_slice(t, c) for c in range(t.channels)]
    else:
        groups = [(0, len(data))]
    codes = [0] * len(data)
    scales: list[float] = []
    zeros: list[int] = []
    clipped = 0

    for lo, hi in groups:
        if given_range is not None:
            rlo, rhi = float(given_range[0]), float(given_range[1])
        else:
            rlo, rhi = _range_of(t.data[lo:hi], clip, percentile, bits)

        s, z = _scale_zero(rlo, rhi, bits, symmetric)
        scales.append(s)
        zeros.append(z)

        for i in range(lo, hi):
            q = math.floor(data[i] / s + 0.5) + z

            if q < qmin:
                clipped += 1
                codes[i] = qmin
            elif q > qmax:
                clipped += 1
                codes[i] = qmax
            else:
                codes[i] = q

    return Quantized(
        name=t.name, shape=tuple(t.shape), codes=tuple(codes),
        scales=tuple(scales), zeros=tuple(zeros), bits=int(bits),
        granularity=granularity, symmetric=bool(symmetric), clipped=clipped,
    )

# 2
def dequantize(qt: Quantized) -> Tensor:

    codes = qt.codes
    scales = qt.scales
    zeros = qt.zeros
    stride = qt.channel_stride if qt.granularity == "per_channel" else len(codes)

    if stride <= 0:
        raise TensorError(f"{qt.name}: a group of {stride} values is not a group")
    
    scale_count = len(scales)
    out = []
    code_count = len(codes)

    for group_start in range(0, code_count, stride):
        g = group_start // stride

        if g >= scale_count:
            raise TensorError(
                f"{qt.name}: value {group_start} falls in group {g} and there are "
                f"{scale_count} scales"
            )
        
        scale = scales[g]
        zero = zeros[g]
        group_end = min(group_start + stride, code_count)

        for i in range(group_start, group_end):
            out.append(round(scale * (int(codes[i]) - zero), 8))

    return Tensor(name=qt.name, shape=qt.shape, data=tuple(out), dtype="fp32")


# 3
def quantization_error(before: Sequence[Tensor],
                       after: Sequence[Tensor]) -> dict[str, float]:
    
    if len(before) != len(after):
        raise TensorError(f"{len(before)} tensors against {len(after)}")
    
    num = 0.0
    den = 0.0
    worst = 0.0

    for x, y in zip(before, after):
        xdata = x.data
        ydata = y.data

        if len(xdata) != len(ydata):
            raise TensorError(
                f"{x.name}: {len(xdata)} values against {len(ydata)}"
            )
        
        for u, v in zip(xdata, ydata):
            d = u - v
            num += d * d
            den += u * u
            abs_d = abs(d)

            if abs_d > worst:
                worst = abs_d
    frob = math.sqrt(num)

    return {
        "frobenius": round(frob, 8),
        "relative": round(frob / math.sqrt(den), 8) if den > 0.0 else 0.0,
        "max_abs": round(worst, 8),
    }


# 4
def bytes_stored(tensors: Sequence[Tensor], precision: str = "fp32",
                 granularity: str = "per_tensor") -> int:
    
    if precision not in PRECISIONS:
        raise TensorError(
            f"unknown precision {precision!r}; known: {list(PRECISIONS)}"
        )
    
    if granularity not in GRANULARITIES:
        raise TensorError(
            f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}"
        )
    
    per = dtype_bytes(precision)
    if precision != "int8":
        return int(sum(t.parameters * per for t in tensors))

    parameters = 0
    if granularity == "per_channel":
        groups = 0
        for t in tensors:
            parameters += t.parameters
            groups += t.channels
    else:
        for t in tensors:
            parameters += t.parameters
        groups = len(tensors)

    return int(parameters * per + groups * (SCALE_BYTES + ZERO_BYTES))

# 5
def effective_bits(tensors: Sequence[Tensor], precision: str = "fp32",
                   granularity: str = "per_tensor") -> float:
    
    n = total_parameters(tensors)

    if n <= 0:
        raise TensorError("a model with no parameters has no bits per parameter")
    
    return round(bytes_stored(tensors, precision, granularity) * 8.0 / n, 4)

# 6
def precision_row(model_name: str, config: Mapping[str, Any],
                  tensors: Sequence[Tensor],
                  calibration_values: Sequence[float],
                  all_values: Sequence[float]) -> dict[str, Any]:
    
    label = str(config.get("label") or config.get("precision") or "")
    precision = str(config.get("precision", "fp32"))

    if precision not in PRECISIONS:
        raise TensorError(
            f"unknown precision {precision!r}; known: {list(PRECISIONS)}"
        )
    
    granularity = str(config.get("granularity", "per_tensor"))
    symmetric = bool(config.get("symmetric", True))
    clip = str(config.get("clip", "minmax"))
    percentile = float(config.get("percentile", DEFAULT_PERCENTILE))
    calibration = str(config.get("calibration", "all"))

    if not label:
        raise TensorError("a configuration with no label cannot be plotted")

    if precision == "int8":
        after = []
        scale_count = 0
        
        for t in tensors:
            q = quantize_tensor(t, DEFAULT_BITS, granularity, symmetric, clip,
                                percentile)
            scale_count += q.scale_count
            after.append(dequantize(q))
    elif precision == "fp16":
        after = [to_fp16(t) for t in tensors]
        scale_count = 0
    else:
        after = list(tensors)
        scale_count = 0

    acts = as_tensor("activations", all_values)
    if precision == "int8":
        rlo, rhi = _range_of(calibration_values, clip, percentile, DEFAULT_BITS)
        qa = quantize_tensor(acts, DEFAULT_BITS, "per_tensor", False, clip,
                             percentile, given_range=(rlo, rhi))
        act_after = dequantize(qa)
        clipped = qa.clipped_fraction
    else:
        rlo, rhi = min(acts.data), max(acts.data)
        act_after = to_fp16(acts) if precision == "fp16" else acts
        clipped = 0.0

    parameters = total_parameters(tensors)
    weight_error = quantization_error(tensors, after)["relative"]
    activation_error = quantization_error([acts], [act_after])["relative"]
    stored = bytes_stored(tensors, precision, granularity)

    if parameters <= 0:
        raise TensorError("a model with no parameters has no bits per parameter")
    bits = round(stored * 8.0 / parameters, 4)

    return {
        "model": str(model_name),
        "label": label,
        "precision": precision,
        "granularity": granularity if precision == "int8" else "n/a",
        "symmetric": symmetric if precision == "int8" else None,
        "calibration": calibration if precision == "int8" else "not calibrated",
        "clip": clip if precision == "int8" else "n/a",
        "parameters": parameters,
        "weight_error": weight_error,
        "activation_error": activation_error,
        "clipped_fraction": clipped,
        "activation_range": [round(float(rlo), 8), round(float(rhi), 8)],
        "scale_count": scale_count,
        "bytes_stored": stored,
        "effective_bits": bits,
    }


# 7
def sweep_precisions(
    model: Mapping[str, Any],
    configs: Sequence[Mapping[str, Any]],
) -> tuple[dict, ...]:
    original_tensors = list(model["tensors"])

    activations = tuple(model.get("activations") or ())
    if not activations:
        raise TensorError(
            "no activation data: an INT8 build cannot be calibrated"
        )

    all_activation_values = tuple(
        value
        for sample in activations
        for value in sample.data
    )

    rows: list[dict] = []
    seen_labels: set[str] = set()

    for config in configs:
        label = str(config.get("label") or config.get("precision") or "")
        if label in seen_labels:
            raise TensorError(f"duplicate configuration label {label!r}")
        seen_labels.add(label)

        policy = str(config.get("calibration", "all"))
        calibration_values = select_calibration(activations, policy)
        row = precision_row(
            model_name=str(model["name"]),
            config=config,
            tensors=original_tensors,
            calibration_values=calibration_values,
            all_values=all_activation_values,
        )
        rows.append(row)

    if not rows:
        raise TensorError("no precision configurations produced benchmark rows")

    return tuple(rows)


# 7
def pareto_front(points: Sequence[Mapping[str, Any]],
                 objectives: Mapping[str, float]) -> tuple[int, ...]:
    pts = list(points)
    if not pts:
        raise TensorError("no points to take a front over")
    if not objectives:
        raise TensorError("a front with no objectives is every point")
    for name in objectives:
        for k, p in enumerate(pts):
            if p.get(name) is None:
                raise TensorError(f"point {k} has no {name!r} to compare on")
    front: list[int] = []
    for i, a in enumerate(pts):
        beaten = False
        for j, b in enumerate(pts):
            if i == j:
                continue
            better = False
            worse = False
            for name, sense in objectives.items():
                d = (float(b[name]) - float(a[name])) * float(sense)
                if d > 0.0:
                    better = True
                elif d < 0.0:
                    worse = True
            if better and not worse:
                beaten = True
                break
        if not beaten:
            front.append(i)
    return tuple(front)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    demo_seed = 459
    demo_shapes = (
        ("conv1.weight", (20, 3, 3, 3)),
        ("conv2.weight", (20, 20, 3, 3)),
        ("conv3.weight", (40, 20, 3, 3)),
        ("fc.weight", (10, 40)),
    )
    model_spec = {
        "name": "tinyconv",
        "seed": demo_seed,
        "tensors": tuple(
            materialise(name, shape, demo_seed + 7919 * index)
            for index, (name, shape) in enumerate(demo_shapes)
        ),
    }
    model_tensors = model_spec["tensors"]
    model_name = model_spec["name"]
    total_params = total_parameters(model_tensors)

    test_tensor = model_tensors[0]  

    raw_samples = materialise_samples(layer="conv3", width=40, count=32, seed=model_spec["seed"] + 100)
    model_spec["activations"] = raw_samples
    all_act_vals = tuple(v for s in raw_samples for v in s.data)
    calib_all = select_calibration(raw_samples, policy="all")
    calib_one_cls = select_calibration(raw_samples, policy="one_class")

    # 1. Test quantize_tensor
    print("\n[1/7] Testing quantize_tensor...")
    q_pt = quantize_tensor(test_tensor, bits=8, granularity="per_tensor", symmetric=True)
    q_pc = quantize_tensor(test_tensor, bits=8, granularity="per_channel", symmetric=True)

    assert len(q_pt.codes) == test_tensor.parameters
    assert q_pt.scale_count == 1
    assert q_pc.scale_count == test_tensor.channels
    print(f"  ✓ Per-tensor quantized:  {len(q_pt.codes)} codes, {q_pt.scale_count} scale")
    print(f"  ✓ Per-channel quantized: {len(q_pc.codes)} codes, {q_pc.scale_count} scales")

    # 2. Test dequantize
    print("\n[2/7] Testing dequantize...")
    deq_pt = dequantize(q_pt)
    deq_pc = dequantize(q_pc)

    assert deq_pt.shape == test_tensor.shape
    assert len(deq_pt.data) == len(test_tensor.data)
    assert deq_pc.shape == test_tensor.shape
    print(f"  ✓ Reconstructed tensor shape: {deq_pt.shape}, dtype: {deq_pt.dtype}")

    # 3. Test quantization_error
    print("\n[3/7] Testing quantization_error...")
    err_pt = quantization_error([test_tensor], [deq_pt])
    err_pc = quantization_error([test_tensor], [deq_pc])

    assert err_pt["relative"] >= 0.0 and err_pc["relative"] >= 0.0
    print(f"  ✓ Per-tensor relative error:  {err_pt['relative']:.6f} (max_abs: {err_pt['max_abs']:.6f})")
    print(f"  ✓ Per-channel relative error: {err_pc['relative']:.6f} (max_abs: {err_pc['max_abs']:.6f})")

    # 4. Test bytes_stored
    print("\n[4/7] Testing bytes_stored...")
    bytes_fp32 = bytes_stored(model_tensors, precision="fp32")
    bytes_fp16 = bytes_stored(model_tensors, precision="fp16")
    bytes_int8_pt = bytes_stored(model_tensors, precision="int8", granularity="per_tensor")
    bytes_int8_pc = bytes_stored(model_tensors, precision="int8", granularity="per_channel")

    assert bytes_fp32 > bytes_fp16 > bytes_int8_pt
    assert bytes_int8_pc >= bytes_int8_pt 
    print(f"  ✓ FP32 bytes:            {bytes_fp32} B")
    print(f"  ✓ FP16 bytes:            {bytes_fp16} B")
    print(f"  ✓ INT8 per-tensor bytes: {bytes_int8_pt} B")
    print(f"  ✓ INT8 per-channel bytes:{bytes_int8_pc} B")

    # 5. Test effective_bits
    print("\n[5/7] Testing effective_bits...")
    eff_fp32 = effective_bits(model_tensors, precision="fp32")
    eff_fp16 = effective_bits(model_tensors, precision="fp16")
    eff_int8_pt = effective_bits(model_tensors, precision="int8", granularity="per_tensor")
    eff_int8_pc = effective_bits(model_tensors, precision="int8", granularity="per_channel")

    assert eff_fp32 == 32.0
    assert eff_fp16 == 16.0
    assert eff_int8_pc > 8.0
    print(f"  ✓ FP32 effective bits:            {eff_fp32:.4f}")
    print(f"  ✓ FP16 effective bits:            {eff_fp16:.4f}")
    print(f"  ✓ INT8 per-tensor effective bits: {eff_int8_pt:.4f}")
    print(f"  ✓ INT8 per-channel effective bits:{eff_int8_pc:.4f}")

    # 6. Test precision_row
    print("\n[6/7] Testing precision_row...")
    configs = [
        {"label": "FP32 Baseline", "precision": "fp32"},
        {"label": "FP16", "precision": "fp16"},
        {
            "label": "INT8 Per-Tensor (All)",
            "precision": "int8",
            "granularity": "per_tensor",
            "symmetric": False,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (All)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (Class 0 Only)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "one_class",
        },
    ]

    rows = sweep_precisions(model_spec, configs)
    for row in rows:
        print(f"  ✓ {row['label']:<32} | W_Err: {row['weight_error']:<8.6f} | Act_Err: {row['activation_error']:<8.6f} | Bits: {row['effective_bits']:<6.3f}")

    # 7. Test pareto_front & Save Output
    print("\n[7/7] Testing pareto_front...")
    objectives = {
        "effective_bits": -1.0,
        "weight_error": -1.0,
    }
    front_indices = pareto_front(rows, objectives)

    assert len(front_indices) > 0
    print(f"  ✓ Pareto optimal candidate indices: {front_indices}")
    for idx in front_indices:
        p = rows[idx]
        print(f"    * Index {idx}: {p['label']} (Bits: {p['effective_bits']}, Weight Err: {p['weight_error']})")


    # Output
    report_path = script_dir / "quantization_test_results.json"
    results = {
        "model": model_name,
        "total_parameters": total_params,
        "objectives": objectives,
        "pareto_front_indices": list(front_indices),
        "configurations": rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Full output written to {report_path.resolve()}")
    