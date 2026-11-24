import math


PRIMARY_METRICS = (
    "seg_mIoU",
    "seg_pixAcc",
    "depth_abs_err",
    "depth_rel_err",
    "normal_mean",
    "normal_median",
)

# Direction used when comparing every metric with the STL baseline.
# Positive signed improvement means that the method is better.
METRIC_DIRECTIONS = {
    "seg_mIoU": 1.0,
    "seg_pixAcc": 1.0,
    "depth_abs_err": -1.0,
    "depth_rel_err": -1.0,
    "normal_mean": -1.0,
    "normal_median": -1.0,
}

METHOD_ORDER = (
    "STL",
    "MTL",
    "EncoderAvg",
    "EncoderAvg+OutputKD",
    "EncoderAvg+IntermediateKD+OutputKD",
)


def relative_improvement_pct(current, baseline, metric):
    """Relative improvement where larger is always better."""
    if baseline == 0:
        return math.nan
    direction = METRIC_DIRECTIONS[metric]
    return direction * (current - baseline) / abs(baseline) * 100.0


def delta_m_pct(metrics, stl_metrics):
    """Mean relative improvement over STL across all six reported metrics."""
    values = [
        relative_improvement_pct(metrics[metric], stl_metrics[metric], metric)
        for metric in PRIMARY_METRICS
    ]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan
