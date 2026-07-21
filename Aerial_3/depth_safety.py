from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


ACTION_TO_CAMERA = {
    "forward": 0,
    "left": 1,
    "right": 2,
    "descend": 3,
}


@dataclass(frozen=True)
class DirectionSafety:
    clearance: float
    required_clearance: float
    near_fraction: float
    valid_fraction: float
    safe: bool
    risk: float

    def to_dict(self):
        result = asdict(self)
        result["clearance"] = round(float(self.clearance), 3)
        result["required_clearance"] = round(float(self.required_clearance), 3)
        result["near_fraction"] = round(float(self.near_fraction), 4)
        result["valid_fraction"] = round(float(self.valid_fraction), 4)
        result["risk"] = round(float(self.risk), 4)
        return result


@dataclass(frozen=True)
class TargetDepthEvidence:
    depth: float
    median_depth: float
    window_fraction: float
    valid_fraction: float


class DepthSafetyAnalyzer:
    """Turn four directional depth maps into an action-level safety shield."""

    def __init__(
        self,
        horizontal_step=2.0,
        vertical_step=1.0,
        safety_margin=0.5,
        percentile=5.0,
    ):
        if horizontal_step <= 0 or vertical_step <= 0:
            raise ValueError("movement steps must be positive")
        if safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        self.horizontal_step = float(horizontal_step)
        self.vertical_step = float(vertical_step)
        self.safety_margin = float(safety_margin)
        self.percentile = float(percentile)

    @staticmethod
    def to_meters(depth_image):
        depth = np.asarray(depth_image)
        if depth.ndim != 2:
            raise ValueError("each depth image must be a two-dimensional array")
        if np.issubdtype(depth.dtype, np.integer):
            return depth.astype(np.float32) / 255.0 * 100.0
        return depth.astype(np.float32)

    @staticmethod
    def corridor(depth_meters):
        height, width = depth_meters.shape
        # Cover the projected vehicle body and lateral drift, not just the
        # optical-axis strip.  The narrower former 36% corridor could report
        # tens of metres of clearance while a propeller/body edge clipped a
        # wall immediately after the action.
        y0, y1 = int(height * 0.05), max(int(height * 0.95), 1)
        x0, x1 = int(width * 0.08), max(int(width * 0.92), 1)
        return depth_meters[y0:y1, x0:x1]

    def analyze_image(self, depth_image, required_clearance=None):
        if required_clearance is None:
            required_clearance = self.horizontal_step + self.safety_margin
        required_clearance = float(required_clearance)
        depth = self.to_meters(depth_image)
        corridor = self.corridor(depth)
        finite = np.isfinite(corridor)
        valid = finite & (corridor >= 0.0) & (corridor <= 100.0)
        valid_fraction = float(np.mean(valid)) if valid.size else 0.0
        values = corridor[valid]

        if values.size == 0 or valid_fraction < 0.90:
            return DirectionSafety(
                clearance=0.0,
                required_clearance=required_clearance,
                near_fraction=1.0,
                valid_fraction=valid_fraction,
                safe=False,
                risk=1.0,
            )

        # Evaluate several vertical bands.  Taking the minimum band percentile
        # catches wall edges and poles that a single whole-corridor percentile
        # can otherwise dilute.
        band_clearances = []
        for band in np.array_split(corridor, 5, axis=1):
            band_values = band[np.isfinite(band) & (band >= 0.0) & (band <= 100.0)]
            if band_values.size:
                band_clearances.append(
                    float(np.percentile(band_values, self.percentile))
                )
        clearance = min(band_clearances) if band_clearances else 0.0
        near_fraction = float(np.mean(values <= required_clearance))
        # The requested rule is deliberately strict: equality is not enough.
        # A translation is safe only when the directional clearance is larger
        # than its fixed step plus the 0.5 m safety margin.
        # A small obstacle can occupy less than one percentile band but still
        # intersect a propeller or body edge.  The independent near-pixel cap
        # blocks that case without making a single noisy pixel fatal.
        safe = bool(
            clearance > required_clearance
            and near_fraction < 0.01
        )
        risk = float(
            np.clip(
                (required_clearance + 2.0 - clearance) / 2.0,
                0.0,
                1.0,
            )
        )
        return DirectionSafety(
            clearance=clearance,
            required_clearance=required_clearance,
            near_fraction=near_fraction,
            valid_fraction=valid_fraction,
            safe=safe,
            risk=risk,
        )

    def analyze(self, depth_images: Iterable[np.ndarray]):
        images = list(depth_images)
        if len(images) != 4:
            raise ValueError("expected four depth images: front, left, right, down")
        result = {}
        for action, camera_index in ACTION_TO_CAMERA.items():
            step = (
                self.vertical_step if action == "descend" else self.horizontal_step
            )
            result[action] = self.analyze_image(
                images[camera_index], step + self.safety_margin
            )
        return result

    def target_region_depth(
        self, depth_image, normalized_box: Tuple[float, float, float, float]
    ):
        return self.target_region_evidence(
            depth_image,
            normalized_box,
            min_depth=0.0,
            max_depth=100.0,
        ).depth

    def target_region_evidence(
        self,
        depth_image,
        normalized_box: Tuple[float, float, float, float],
        min_depth: float,
        max_depth: float,
    ):
        depth = self.to_meters(depth_image)
        height, width = depth.shape
        x0, y0, x1, y1 = normalized_box
        ix0 = max(0, min(width - 1, int(x0 * width)))
        iy0 = max(0, min(height - 1, int(y0 * height)))
        ix1 = max(ix0 + 1, min(width, int(x1 * width)))
        iy1 = max(iy0 + 1, min(height, int(y1 * height)))
        region = depth[iy0:iy1, ix0:ix1]
        valid = np.isfinite(region) & (region >= 0.0) & (region <= 100.0)
        valid_fraction = float(np.mean(valid)) if valid.size else 0.0
        values = region[valid]
        if not values.size:
            return TargetDepthEvidence(
                depth=float("inf"),
                median_depth=float("inf"),
                window_fraction=0.0,
                valid_fraction=valid_fraction,
            )
        # A lower percentile is more useful than the median for a small object
        # inside a crop, while still rejecting isolated one-pixel noise.
        target_depth = float(np.percentile(values, 15.0))
        window_fraction = float(
            np.mean((values >= float(min_depth)) & (values <= float(max_depth)))
        )
        return TargetDepthEvidence(
            depth=target_depth,
            median_depth=float(np.median(values)),
            window_fraction=window_fraction,
            valid_fraction=valid_fraction,
        )

    @staticmethod
    def serializable(result: Dict[str, DirectionSafety]):
        return {action: safety.to_dict() for action, safety in result.items()}
