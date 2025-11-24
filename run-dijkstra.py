import os
import numpy as np
import math, heapq, time
import matplotlib.pyplot as plt


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GRID_FILE  = os.path.join(BASE_DIR, "occupancy_grid_numpy.npy")  
SCANGRID_PY = os.path.join(BASE_DIR, "scangrid.py")             

START_IDX = (50, 32)                     
GOAL_IDX  = (15, 57)                    
USE_8_CONNECTED = True

ROBOT_RADIUS_PX = 2

OUT_IMG = os.path.join(BASE_DIR, "dijkstra_path.png")

# Helpers 
def in_bounds(grid, y, x):
    h, w = grid.shape
    return 0 <= y < h and 0 <= x < w

def inflate_obstacles(grid, r):
    """Return a *copy* of grid with obstacles/unknown dilated by r pixels."""
    if r <= 0:
        return grid.copy()
    h, w = grid.shape
    out = grid.copy()
    occ_y, occ_x = np.where(grid != 0)  
    disk = [(dy, dx) for dy in range(-r, r+1) for dx in range(-r, r+1) if dy*dy + dx*dx <= r*r]
    for y, x in zip(occ_y, occ_x):
        for dy, dx in disk:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                out[yy, xx] = 1
    return out

def find_nearest_free(grid, y, x):
    """Snap (y,x) to the nearest free (0) cell on 'grid'. BFS over 4-neighbors."""
    if in_bounds(grid, y, x) and grid[y, x] == 0:
        return (y, x)
    from collections import deque
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    q = deque()
    if in_bounds(grid, y, x):
        q.append((y, x))
        vis[y, x] = True
    while q:
        cy, cx = q.popleft()
        if in_bounds(grid, cy, cx) and grid[cy, cx] == 0:
            return (cy, cx)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy + dy, cx + dx
            if in_bounds(grid, ny, nx) and not vis[ny, nx]:
                vis[ny, nx] = True
                q.append((ny, nx))
    raise ValueError(f"No free cell found near {(y, x)}")

def neighbors_8(y, x):
    r2 = math.sqrt(2.0)
    for dy, dx, c in [(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),
                      (-1,-1,r2),(-1,1,r2),(1,-1,r2),(1,1,r2)]:
        yield y+dy, x+dx, c

def dijkstra(grid, start, goal, eight_connected=True):
    """Run Dijkstra on 'grid' where 0=free, 1/-1=blocked."""
    h, w = grid.shape
    sy, sx = start
    gy, gx = goal

    g = np.full((h, w), np.inf)
    parent = np.full((h, w, 2), -1, dtype=int)
    g[sy, sx] = 0.0
    pq = [(0.0, (sy, sx))]
    nodes = 0
    t0 = time.perf_counter()

    while pq:
        cost, (y, x) = heapq.heappop(pq)
        if (y, x) == (gy, gx):
            break
        if cost > g[y, x]:
            continue
        nodes += 1
        it = neighbors_8(y, x) if eight_connected else [(y-1,x,1),(y+1,x,1),(y,x-1,1),(y,x+1,1)]
        for ny, nx, step in it:
            if not in_bounds(grid, ny, nx) or grid[ny, nx] != 0:
                continue
            new = cost + step
            if new < g[ny, nx]:
                g[ny, nx] = new
                parent[ny, nx] = [y, x]
                heapq.heappush(pq, (new, (ny, nx)))

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Reconstruct path
    path = []
    if np.isfinite(g[gy, gx]):
        y, x = gy, gx
        while not (y == sy and x == sx):
            path.append((y, x))
            py, px = parent[y, x]
            if py < 0:
                break
            y, x = py, px
        path.append((sy, sx))
        path.reverse()

    return {
        "path": path,
        "runtime_ms": runtime_ms,
        "nodes_expanded": nodes,
        "path_cost": float(g[gy, gx]) if np.isfinite(g[gy, gx]) else math.inf
    }

def main():
    if not os.path.exists(GRID_FILE):
        if os.path.exists(SCANGRID_PY):
            print(f"[info] '{GRID_FILE}' not found. Running scangrid.py to generate it...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("scangrid", SCANGRID_PY)
            scangrid = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scangrid)
            if hasattr(scangrid, "main"):
                scangrid.main()  
            else:
                raise RuntimeError("scangrid.py does not define main(). Please run 'python scangrid.py' manually.")
        else:
            raise FileNotFoundError(
                f"Missing '{GRID_FILE}'. Run your grid generator first: 'python scangrid.py' in {BASE_DIR}"
            )

    grid_orig = np.load(GRID_FILE)
    print("Loaded grid:", grid_orig.shape)

    grid_plan = inflate_obstacles(grid_orig, ROBOT_RADIUS_PX)

    start = find_nearest_free(grid_plan, *START_IDX)
    goal  = find_nearest_free(grid_plan, *GOAL_IDX)
    if start != START_IDX:
        print(f"[info] start {START_IDX} not free → snapped to {start}")
    if goal != GOAL_IDX:
        print(f"[info] goal  {GOAL_IDX} not free → snapped to {goal}")

    result = dijkstra(grid_plan, start, goal, eight_connected=USE_8_CONNECTED)
    print(f"Runtime (ms): {result['runtime_ms']:.2f}")
    print(f"Nodes expanded: {result['nodes_expanded']}")
    print(f"Path cost: {result['path_cost']:.3f}")
    print(f"Path length (cells): {len(result['path']) if result['path'] else 0}")

    disp = np.zeros((*grid_orig.shape, 3), dtype=np.uint8)
    disp[grid_orig == 1]  = [0, 0, 0]       
    disp[grid_orig == 0]  = [255, 255, 255] 
    disp[grid_orig == -1] = [128, 128, 128] 

    plt.figure(figsize=(8, 8))
    plt.imshow(disp, origin="upper")
    if result["path"]:
        ys, xs = zip(*result["path"])
        plt.plot(xs, ys, linewidth=2)                
        plt.scatter([xs[0]],[ys[0]], s=40)            
        plt.scatter([xs[-1]],[ys[-1]], s=40)          
    plt.title("Dijkstra Path")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=220)
    plt.show()
    print(f"Saved {OUT_IMG}")

if __name__ == "__main__":
    main()
