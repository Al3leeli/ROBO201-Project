import os
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_FILE = os.path.join(BASE_DIR, "occupancy_grid_numpy.npy")

START_NODE = (50, 32)
GOAL_NODE  = (15, 57)

def load_grids(path):
    g_orig = np.load(path)                 
    g_plan = np.where(g_orig == 0, 0, 1)   
    return g_orig, g_plan

def is_valid(n, grid):
    r, c = n
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == 0

def neighbours(n):
    r, c = n
    return [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]

def dijkstra(start_node, goal_node, grid):
    start_time = time.perf_counter()
    pq = PriorityQueue()
    pq.put((0, start_node))
    closed = set()
    parent = {start_node: None}
    g = {start_node: 0}
    nodes_expanded = 0

    while not pq.empty():
        cost, u = pq.get()
        if u == goal_node: break
        if u in closed: continue
        closed.add(u); nodes_expanded += 1
        for v in neighbours(u):
            if is_valid(v, grid) and v not in closed:
                nc = cost + 1
                if (v not in g) or (nc < g[v]):
                    g[v] = nc; parent[v] = u
                    pq.put((nc, v))

    end_time = time.perf_counter()
    runtime_ms = (end_time - start_time) * 1000

    path = []
    n = goal_node
    while n != start_node:
        path.append(n)
        n = parent.get(n)
        if n is None: return [], nodes_expanded, np.inf, runtime_ms
    path.append(start_node); path.reverse()
    return path, nodes_expanded, len(path) - 1, runtime_ms

def plot_like_sample(orig_grid, start_node, goal_node, path):
    h, w = orig_grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[orig_grid == 1]  = [0, 0, 0]       
    rgb[orig_grid == 0]  = [255, 255, 255] 
    rgb[orig_grid == -1] = [128, 128, 128] 

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb, origin="upper", interpolation="none")
    if path:
        ys, xs = zip(*path)
        plt.plot(xs, ys, linewidth=2)
        plt.scatter([xs[0]],[ys[0]], s=80, color="#1f77b4")   
        plt.scatter([xs[-1]],[ys[-1]], s=80, color="#ff7f0e") 
    plt.title("Dijkstra Path")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(GRID_FILE):
        raise FileNotFoundError("Run your scan grid script first to create occupancy_grid_numpy.npy")
    grid_orig, grid_plan = load_grids(GRID_FILE)
    path, nodes_expanded, path_cost, runtime_ms = dijkstra(START_NODE, GOAL_NODE, grid_plan)
    print(f"Nodes expanded: {nodes_expanded} | Path cost: {path_cost} | Runtime: {runtime_ms:.2f} ms")
    plot_like_sample(grid_orig, START_NODE, GOAL_NODE, path)
