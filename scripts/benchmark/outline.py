"""Outline simplification for the example tiles, shared by make_tiles.py and publish_result.py."""


def simplify(xy: list[float], tol: float = 0.6) -> list[int]:
    """Ramer-Douglas-Peucker on a closed outline, in tile pixels, rounded to whole pixels.

    At tile resolution a refined outline carries far more points than can be seen; this keeps
    every vertex that moves the line by more than ``tol`` pixels.
    """
    pts = list(zip(xy[0::2], xy[1::2]))
    if len(pts) <= 4:
        return [round(v) for p in pts for v in p]

    def rdp(seg):
        (ax, ay), (bx, by) = seg[0], seg[-1]
        dx, dy = bx - ax, by - ay
        n = (dx * dx + dy * dy) ** 0.5 or 1e-9
        far, at = -1.0, 0
        for k in range(1, len(seg) - 1):
            px, py = seg[k]
            d = abs(dy * px - dx * py + bx * ay - by * ax) / n
            if d > far:
                far, at = d, k
        if far <= tol:
            return [seg[0], seg[-1]]
        return rdp(seg[:at + 1])[:-1] + rdp(seg[at:])

    half = len(pts) // 2  # split the ring in two so the end points are distinct
    keep = rdp(pts[:half + 1])[:-1] + rdp(pts[half:] + [pts[0]])[:-1]
    return [round(v) for p in keep for v in p]
