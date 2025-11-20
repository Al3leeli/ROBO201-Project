import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, "projectscan.yaml")


BASE_DIR2 = os.path.dirname(os.path.abspath(__file__))
PGM_PATH = os.path.join(BASE_DIR2, "projectscan.pgm")


meta = {
    "resolution": 0.05,
    "origin": [-3.15, -1.25, 0.0],
    "negate": 0,
    "occupied_thresh": 0.65,
    "free_thresh": 0.25,
}

def _parse_yaml_if_present(path: str) -> None:
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = [s.strip() for s in line.split(":", 1)]
            if k == "resolution":
                meta["resolution"] = float(v)
            elif k == "origin":
                v = v.replace("[", "").replace("]", "")
                parts = [p.strip() for p in v.split(",")]
                if len(parts) >= 3:
                    meta["origin"] = [float(parts[0]), float(parts[1]), float(parts[2])]
            elif k == "negate":
                meta["negate"] = int(v)
            elif k in ("occupied_thresh", "occupied_threshold"):
                meta["occupied_thresh"] = float(v)
            elif k in ("free_thresh", "free_threshold"):
                meta["free_thresh"] = float(v)

def read_p5_pgm_numpy(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError("Only P5 (binary) PGM format supported.")
        def _next_noncomment():
            line = f.readline()
            while line.startswith(b"#") or len(line.strip()) == 0:
                line = f.readline()
            return line
        dims = _next_noncomment().split()
        while len(dims) < 2:
            dims += _next_noncomment().split()
        w, h = int(dims[0]), int(dims[1])
        maxval = int(_next_noncomment().strip())
        if maxval > 255:
            raise ValueError("maxval > 255 not supported.")
        buf = f.read(w * h)
        if len(buf) != w * h:
            raise ValueError("Unexpected raster size.")
        return np.frombuffer(buf, dtype=np.uint8).reshape((h, w))

def make_occupancy_grid(gray: np.ndarray, negate: int, occupied_thresh: float, free_thresh: float) -> np.ndarray:
    gray = gray.astype(np.uint8)
    occ_prob = (255.0 - gray.astype(np.float32)) / 255.0 if negate == 0 else gray.astype(np.float32) / 255.0
    grid = np.full(gray.shape, -1, dtype=np.int8)
    grid[occ_prob > occupied_thresh] = 1
    grid[occ_prob < free_thresh] = 0
    return grid

def flood_from_border_nonocc(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    stack = []
    def push(r, c):
        if 0 <= r < h and 0 <= c < w and (not exterior[r, c]) and grid[r, c] != 1:
            exterior[r, c] = True
            stack.append((r, c))
    for r in range(h):
        push(r, 0); push(r, w - 1)
    for c in range(w):
        push(0, c); push(h - 1, c)
    while stack:
        r, c = stack.pop()
        if r > 0: push(r - 1, c)
        if r + 1 < h: push(r + 1, c)
        if c > 0: push(r, c - 1)
        if c + 1 < w: push(r, c + 1)
    return exterior

def mark_outside_grey(grid: np.ndarray) -> np.ndarray:
    exterior = flood_from_border_nonocc(grid)
    out = grid.copy()
    out[(grid != 1) & (exterior)] = -1
    return out

def block_small_enclosed_free(grid: np.ndarray, max_area: int) -> np.ndarray:
    h, w = grid.shape
    exterior = flood_from_border_nonocc(grid)
    candidate = (grid == 0) & (~exterior)
    seen = np.zeros((h, w), dtype=bool)
    out = grid.copy()

    def bfs(start_r, start_c):
        q = [(start_r, start_c)]
        seen[start_r, start_c] = True
        cells = [(start_r, start_c)]
        head = 0
        while head < len(q):
            r, c = q[head]; head += 1
            if r > 0 and candidate[r - 1, c] and not seen[r - 1, c]:
                seen[r - 1, c] = True; q.append((r - 1, c)); cells.append((r - 1, c))
            if r + 1 < h and candidate[r + 1, c] and not seen[r + 1, c]:
                seen[r + 1, c] = True; q.append((r + 1, c)); cells.append((r + 1, c))
            if c > 0 and candidate[r, c - 1] and not seen[r, c - 1]:
                seen[r, c - 1] = True; q.append((r, c - 1)); cells.append((r, c - 1))
            if c + 1 < w and candidate[r, c + 1] and not seen[r, c + 1]:
                seen[r, c + 1] = True; q.append((r, c + 1)); cells.append((r, c + 1))
        return cells

    for r in range(h):
        for c in range(w):
            if candidate[r, c] and not seen[r, c]:
                cells = bfs(r, c)
                if len(cells) <= max_area:
                    for rr, cc in cells:
                        out[rr, cc] = 1
    return out

def main(max_hole_area: int = 60, pgm_path: str = PGM_PATH, yaml_path: str = YAML_PATH) -> None:
    _parse_yaml_if_present(yaml_path)
    gray = read_p5_pgm_numpy(pgm_path)
    grid = make_occupancy_grid(gray, meta["negate"], meta["occupied_thresh"], meta["free_thresh"])
    grid = mark_outside_grey(grid)
    grid = block_small_enclosed_free(grid, max_hole_area)

    np.save("occupancy_grid_numpy.npy", grid)
    np.savetxt("occupancy_grid_numpy.csv", grid, fmt="%d", delimiter=",")
    unique, counts = np.unique(grid, return_counts=True)
    stats = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
    print("Grid shape:", grid.shape)
    print("Meta:", meta)
    print("Value counts (0=free, 1=occ, -1=unknown):", stats)
    print("Saved: occupancy_grid_numpy.npy, occupancy_grid_numpy.csv")

    display_grid = np.zeros((*grid.shape, 3), dtype=np.uint8)
    display_grid[grid == 1]  = [0, 0, 0]
    display_grid[grid == 0]  = [255, 255, 255]
    display_grid[grid == -1] = [128, 128, 128]

    plt.figure(figsize=(8, 8))
    plt.imshow(display_grid, origin="upper")
    plt.title("Occupancy Grid")
    plt.axis("off")
    plt.tight_layout()

    black_patch = mpatches.Patch(color='black', label='Obstacle / Wall (1)')
    white_patch = mpatches.Patch(color='white', label='Free Space (0)')
    grey_patch  = mpatches.Patch(color='grey',  label='Unknown (-1)')
    plt.legend(handles=[white_patch, black_patch, grey_patch], loc='lower right', framealpha=0.9)

    plt.show()

if __name__ == "__main__":
    main()
