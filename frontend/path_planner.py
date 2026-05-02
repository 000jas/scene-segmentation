"""
path_planner.py — A* navigator on a segmentation-derived cost grid.

Public API:
  plan_path(seg_map) -> dict with keys:
    "path"       : list of (row, col) in full-image coords, or []
    "grid"       : downsampled binary obstacle grid (for debug)
    "scale"      : downsample factor applied
    "start"      : (row, col) start in full-image coords
    "goal"       : (row, col) goal  in full-image coords
    "found"      : bool
"""

import heapq
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Downsample factor — A* runs on this smaller grid (much faster)
GRID_SCALE = 6

# Cost per grid cell type (from segmentation.py TRAVERSAL_COST)
# 0=clear, 1=costly, 2=hard obstacle
MOVE_COST  = {0: 1.0, 1: 4.0, 2: float("inf")}

# 8-directional movement (dr, dc, base_cost_multiplier)
_DIRS = [
    (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
    (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414), ( 1,  1, 1.414),
]


def _build_grid(seg_map: np.ndarray, scale: int) -> np.ndarray:
    """Downsample and convert cost map to obstacle grid."""
    h, w  = seg_map.shape
    sh, sw = h // scale, w // scale

    # Import cost mapping from segmentation module
    from segmentation import TRAVERSAL_COST
    cost_map = np.full(seg_map.shape, 2, dtype=np.uint8)
    for cls, cost in TRAVERSAL_COST.items():
        cost_map[seg_map == cls] = cost

    # Downsample taking the max (worst-case obstacle wins)
    small = cv2.resize(cost_map, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return small


def _heuristic(a, b):
    # Octile distance (matches 8-directional movement)
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (1.414 - 1.0) * min(dr, dc)


def _astar(grid: np.ndarray, start: tuple, goal: tuple):
    """
    A* on a cost grid. Cells with cost==inf are impassable.
    Returns list of (r, c) from start to goal, or [].
    """
    rows, cols = grid.shape

    def cell_cost(r, c):
        v = int(grid[r, c])
        return MOVE_COST.get(v, float("inf"))

    # If goal is an obstacle, relax it to nearest clear cell
    if cell_cost(*goal) == float("inf"):
        best_dist, best_cell = float("inf"), goal
        for r in range(rows):
            for c in range(cols):
                if cell_cost(r, c) < float("inf"):
                    d = abs(r - goal[0]) + abs(c - goal[1])
                    if d < best_dist:
                        best_dist, best_cell = d, (r, c)
        goal = best_cell

    open_set  = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score   = {start: 0.0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]

        r, c = cur
        for dr, dc, base_mul in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cc = cell_cost(nr, nc)
            if cc == float("inf"):
                continue
            tentative_g = g_score[cur] + base_mul * cc
            nb = (nr, nc)
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = cur
                g_score[nb]   = tentative_g
                f               = tentative_g + _heuristic(nb, goal)
                heapq.heappush(open_set, (f, nb))

    return []   # no path found


def plan_path(seg_map: np.ndarray) -> dict:
    """
    Full pipeline: seg_map -> obstacle grid -> A* -> full-res path.

    Start : bottom-centre of image
    Goal  : top-centre of image (top 5% row)
    """
    scale = GRID_SCALE
    grid  = _build_grid(seg_map, scale)
    gh, gw = grid.shape

    # Grid-space start / goal
    g_start = (gh - 1,      gw // 2)
    g_goal  = (gh // 20,    gw // 2)   # top ~5 %

    raw_path = _astar(grid, g_start, g_goal)

    # Upscale back to full-image coordinates (centre of each cell)
    full_path = [
        (r * scale + scale // 2, c * scale + scale // 2)
        for r, c in raw_path
    ]

    # Full-image start/goal
    full_start = (g_start[0] * scale + scale // 2,
                  g_start[1] * scale + scale // 2)
    full_goal  = (g_goal[0]  * scale + scale // 2,
                  g_goal[1]  * scale + scale // 2)

    return {
        "path"  : full_path,
        "grid"  : grid,
        "scale" : scale,
        "start" : full_start,
        "goal"  : full_goal,
        "found" : len(full_path) > 0,
    }


# ── Compatibility API used by app.py and test_integration.py ─────────

def compute_path(image, obstacle_grid, start_pos, goal_pos):
    """
    Compatibility wrapper used by the Streamlit app.

    Args:
        image        : BGR image (H, W, 3) — used only to draw path on
        obstacle_grid: binary array (H, W), 0=navigable 1=obstacle
        start_pos    : (x, y) pixel coord in image space
        goal_pos     : (x, y) pixel coord in image space

    Returns:
        path_image : BGR image with path drawn on it
        path       : list of (x, y) points, or None if no path found
    """
    import cv2 as _cv2

    h, w = obstacle_grid.shape

    # Build our cost grid from binary obstacle grid
    # 0 = clear (navigable), 2 = hard obstacle
    cost_grid = (obstacle_grid * 2).astype(np.uint8)

    # Convert (x, y) start/goal to (row, col)
    s_col, s_row = int(start_pos[0]), int(start_pos[1])
    g_col, g_row = int(goal_pos[0]),  int(goal_pos[1])
    s_row = max(0, min(s_row, h - 1))
    s_col = max(0, min(s_col, w - 1))
    g_row = max(0, min(g_row, h - 1))
    g_col = max(0, min(g_col, w - 1))

    raw_path = _astar(cost_grid, (s_row, s_col), (g_row, g_col))

    path_image = image.copy()

    if not raw_path:
        _cv2.putText(path_image, "NO PATH FOUND", (w // 4, h // 2),
                     _cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return path_image, None

    # raw_path is (row, col) → convert back to (x, y) for drawing
    xy_path = [(c, r) for r, c in raw_path]

    pts = np.array(xy_path, dtype=np.int32)
    _cv2.polylines(path_image, [pts.reshape(-1, 1, 2)], False,
                   (0, 255, 0), thickness=3, lineType=_cv2.LINE_AA)

    # Start / goal markers
    _cv2.circle(path_image, (s_col, s_row), 10, (200, 80,  0 ), -1)
    _cv2.circle(path_image, (g_col, g_row), 10, (0,   200, 200), -1)

    return path_image, xy_path


class AStarPathPlanner:
    """Class wrapper around compute_path for use in app.py."""

    def plan(self, image, obstacle_grid, start_pos, goal_pos):
        return compute_path(image, obstacle_grid, start_pos, goal_pos)