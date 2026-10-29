"""Apply non-maximum suppression to overlapping plate candidates."""


def bounding_box(candidate):
    """Make an axis-aligned [x1, y1, x2, y2] box around four corners."""
    xs = [point[0] for point in candidate["corners_px"]]
    ys = [point[1] for point in candidate["corners_px"]]
    return min(xs), min(ys), max(xs), max(ys)


def box_iou(box_a, box_b):
    """Calculate intersection over union for two axis-aligned boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def non_max_suppression(candidates, iou_threshold=0.5):
    """Keep high-confidence candidates and suppress highly overlapping ones."""
    ordered = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
    kept = []

    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        best_box = bounding_box(best)

        # Class-agnostic suppression treats overlapping predictions as the
        # same physical plate, even if their predicted pattern labels differ.
        ordered = [
            candidate
            for candidate in ordered
            if box_iou(best_box, bounding_box(candidate)) <= iou_threshold
        ]

    return kept
