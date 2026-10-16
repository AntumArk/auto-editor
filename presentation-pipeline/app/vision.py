"""Conservative slide-change detection; OpenCV is needed only when called."""

import math
from collections.abc import Mapping


def _get(obj, name, default):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def detect_slide_changes(video_path, cfg) -> dict:
    """Sample grayscale frames, excluding transition sampling windows from static time.

    Optional visual settings: pixel_change_threshold (0.08) and
    min_changed_pixel_fraction (0.01), both normalized to [0, 1].
    """
    visual = _get(cfg, "visual", {})
    sample_fps = float(_get(visual, "sample_fps", 3))
    width = float(_get(visual, "downscale_width", 640))
    threshold = float(_get(visual, "change_threshold", 0.04))
    pixel_threshold = float(_get(visual, "pixel_change_threshold", 0.08))
    min_fraction = float(_get(visual, "min_changed_pixel_fraction", 0.01))
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("visual.sample_fps must be positive and finite")
    if not math.isfinite(width) or width < 1 or not width.is_integer():
        raise ValueError("visual.downscale_width must be a positive integer")
    if any(not math.isfinite(v) or not 0 <= v <= 1
           for v in (threshold, pixel_threshold, min_fraction)):
        raise ValueError("visual thresholds must be finite and between 0 and 1")

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        if not math.isfinite(frame_count) or frame_count <= 0:
            raise ValueError("Video has no valid frame count")
        duration = frame_count / fps
        changes, intervals, samples = [], [], []
        previous = None
        previous_time = static_start = 0.0
        frame_index = sample_index = 0
        while capture.grab():
            time = frame_index / fps
            frame_index += 1
            if time + 1e-9 < sample_index / sample_fps:
                continue
            sample_index = math.floor(time * sample_fps + 1e-9) + 1
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode video frame at {time:.3f}s")
            height, source_width = frame.shape[:2]
            if source_width > width:
                frame = cv2.resize(
                    frame, (int(width), max(1, round(height * width / source_width))),
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            diff = 0.0
            if previous is not None:
                delta = cv2.absdiff(gray, previous)
                diff = float(delta.mean()) / 255.0
                _, mask = cv2.threshold(delta, pixel_threshold * 255, 255, cv2.THRESH_BINARY)
                fraction = cv2.countNonZero(mask) / mask.size
                if diff > threshold and fraction >= min_fraction:
                    changes.append(time)
                    # The transition happened somewhere between the two samples.
                    if previous_time > static_start:
                        intervals.append((static_start, previous_time))
                    static_start = time
            samples.append((time, diff))
            previous, previous_time = gray, time
        if previous is None or frame_index + 1 < frame_count:
            raise ValueError("Video ended before its frames could be decoded")
        duration = min(duration, frame_index / fps)
        if duration > static_start:
            intervals.append((static_start, duration))
        return {"change_times": changes, "static_intervals": intervals, "samples": samples}
    finally:
        capture.release()
