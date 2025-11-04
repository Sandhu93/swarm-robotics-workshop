# Ant Colony Optimization for Pathfinding
# Ants search for optimal path from starting point to destination using pheromone trails

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation as mpl_animation
import argparse

# ---------- Parameters ----------
GRID = (100, 100)      # grid height, width
N_ANTS = 30            # number of ants 
STEPS = 500            # animation steps  
STEP_SIZE = 1.0        # ant step per frame (smaller to avoid skipping through walls)
BIAS = 0.90            # exploitation probability (higher to converge onto best trail)
DEPOSIT = 120.0        # total pheromone deposited per successful path (Q in ACO)
EVAP = 0.01            # evaporation per step 
DIFFUSE = 0.02         # diffusion weight (keep low to preserve trail contrast)
START_POS = np.array([20, 20])      # (row, col) of starting point
DESTINATION_POS = np.array([80, 80]) # (row, col) of destination
POINT_RADIUS = 3       # radius for start/destination detection
EXPLORATION_BONUS = 0.1 # bonus for exploring new areas
CORRIDOR_BONUS = 0.6    # bias for stepping within carved corridor cells

H, W = GRID

def build_walls():
    # Build a more challenging but solvable maze-like pattern
    w = np.zeros(GRID, dtype=bool)

    # Main vertical barrier with two gaps
    w[15:85, 49:51] = True
    w[45:55, 49:51] = False  # central gap
    w[75:80, 49:51] = False  # lower gap

    # Horizontal corridors with staggered gaps
    for r in [25, 40, 60, 75]:
        w[r:r+2, 10:90] = True
    # Stagger gaps to force zig-zag path
    gaps = [(25, 15), (40, 70), (60, 30), (75, 80)]
    for rr, gc in gaps:
        w[rr:rr+2, gc-2:gc+3] = False

    # Small blocks
    w[30:35, 65:70] = True
    w[68:72, 30:36] = True
    w[50:55, 58:63] = True
    w[20:23, 25:35] = True

    # Ensure start and destination disks are free
    sr0, sc0 = int(START_POS[0]), int(START_POS[1])
    dr0, dc0 = int(DESTINATION_POS[0]), int(DESTINATION_POS[1])
    for rr in range(max(0, sr0-POINT_RADIUS-1), min(GRID[0], sr0+POINT_RADIUS+2)):
        for cc in range(max(0, sc0-POINT_RADIUS-1), min(GRID[1], sc0+POINT_RADIUS+2)):
            w[rr, cc] = False
    for rr in range(max(0, dr0-POINT_RADIUS-1), min(GRID[0], dr0+POINT_RADIUS+2)):
        for cc in range(max(0, dc0-POINT_RADIUS-1), min(GRID[1], dc0+POINT_RADIUS+2)):
            w[rr, cc] = False

    # Carve a guaranteed 3-cell-wide corridor from START_POS to DESTINATION_POS via waypoints
    waypoints = [
        (int(START_POS[0]), int(START_POS[1])),
        (20, 45), (50, 45), (50, 55), (80, 55),
        (int(DESTINATION_POS[0]), int(DESTINATION_POS[1]))
    ]
    corridor_radius = 2
    corridor = np.zeros(GRID, dtype=bool)
    for (r0, c0), (r1, c1) in zip(waypoints[:-1], waypoints[1:]):
        rr0, rr1 = min(r0, r1), max(r0, r1)
        cc0, cc1 = min(c0, c1), max(c0, c1)
        if r0 == r1:
            for cc in range(cc0, cc1+1):
                rmin = max(0, r0-corridor_radius); rmax = min(GRID[0], r0+corridor_radius+1)
                cmin = max(0, cc-corridor_radius); cmax = min(GRID[1], cc+corridor_radius+1)
                w[rmin:rmax, cmin:cmax] = False
                corridor[rmin:rmax, cmin:cmax] = True
        elif c0 == c1:
            for rr in range(rr0, rr1+1):
                rmin = max(0, rr-corridor_radius); rmax = min(GRID[0], rr+corridor_radius+1)
                cmin = max(0, c0-corridor_radius); cmax = min(GRID[1], c0+corridor_radius+1)
                w[rmin:rmax, cmin:cmax] = False
                corridor[rmin:rmax, cmin:cmax] = True
        else:
            # L-shaped: clear rect covering the bend
            rmin = min(r0, r1) - corridor_radius; rmax = max(r0, r1) + corridor_radius + 1
            cmin = min(c0, c1) - corridor_radius; cmax = max(c0, c1) + corridor_radius + 1
            w[max(0,rmin):min(GRID[0],rmax), max(0,cmin):min(GRID[1],cmax)] = False
            corridor[max(0,rmin):min(GRID[0],rmax), max(0,cmin):min(GRID[1],cmax)] = True

    return w, corridor

walls, corridor_mask = build_walls()
pher = np.zeros(GRID, dtype=np.float32)  # pheromone field for pathfinding
path_memory = np.zeros(GRID, dtype=np.float32)  # tracks successful paths
# Seed faint pheromone along the guaranteed corridor at startup (animation mode)
pher[corridor_mask] += 0.05
path_memory[corridor_mask] += 0.2
rng = np.random.default_rng(42)

# Ant positions as float coordinates (row, col) - start at starting point
ants = np.vstack([
    rng.normal(START_POS[0], 2.0, size=N_ANTS),   # start near starting point
    rng.normal(START_POS[1], 2.0, size=N_ANTS)
]).T

# Each ant: 0 = searching for destination, 1 = returning to start (depositing trail)
state = np.zeros(N_ANTS, dtype=np.int8)
# Track if ant has found destination at least once
found_destination = np.zeros(N_ANTS, dtype=bool)
# Path length for each ant (to reward shorter paths)
path_length = np.zeros(N_ANTS, dtype=int)
# Track ant positions to detect if stuck
prev_positions = np.copy(ants)
stuck_counter = np.zeros(N_ANTS, dtype=int)

# Record lattice-paths for each ant while searching; used to lay pheromone Q/L on return
paths = [[] for _ in range(N_ANTS)]  # list of (r, c) integer cells per ant
# Index for following the recorded path backwards when returning
return_index = np.full(N_ANTS, -1, dtype=int)

# For simple heading inertia
heading = rng.uniform(0, 2*np.pi, size=N_ANTS)

def clamp_positions(p):
    p[:,0] = np.clip(p[:,0], 0, H-1 - 1e-6)
    p[:,1] = np.clip(p[:,1], 0, W-1 - 1e-6)
    
    # Check for walls and prevent ants from entering them
    rr = p[:,0].astype(int)
    cc = p[:,1].astype(int)
    in_wall = walls[rr, cc]
    
    if np.any(in_wall):
        # Move ants away from walls - they cannot pass through
        wall_ants = np.where(in_wall)[0]
        for ant_idx in wall_ants:
            # Find nearest free space by checking surrounding cells
            r, c = int(p[ant_idx, 0]), int(p[ant_idx, 1])
            
            # Check 8 directions around the ant to find free space
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if (0 <= new_r < H and 0 <= new_c < W and 
                    not walls[new_r, new_c]):
                    p[ant_idx, 0] = new_r + 0.5
                    p[ant_idx, 1] = new_c + 0.5
                    break
            else:
                # If no free space found, push ant back with random direction
                angle = rng.uniform(0, 2*np.pi)
                p[ant_idx, 0] += 3 * np.cos(angle)
                p[ant_idx, 1] += 3 * np.sin(angle)
        
        # Re-clamp after moving away from walls
        p[:,0] = np.clip(p[:,0], 0, H-1 - 1e-6)
        p[:,1] = np.clip(p[:,1], 0, W-1 - 1e-6)
    
    return p

def choose_direction(idx):
    r, c = ants[idx]
    s = state[idx]
    
    # Simple approach: try 8 directions around current heading
    n_directions = 8
    base_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    
    # Add some randomness but keep it simple
    jitter = rng.normal(0, 0.3)
    angles = base_angles + jitter
    
    # Candidate steps
    dr = STEP_SIZE * np.cos(angles)
    dc = STEP_SIZE * np.sin(angles)

    # Sample pheromone and heuristic at candidate target cells
    new_r = r + dr
    new_c = c + dc
    
    # Clamp to grid bounds
    rr = np.clip(new_r.astype(int), 0, H-1)
    cc = np.clip(new_c.astype(int), 0, W-1)
    
    # Check for walls and calculate values
    is_wall = walls[rr, cc]
    theta_candidates = angles

    if s == 0:
        # Searching for destination: probabilistic choice with pheromone (alpha) and heuristic (beta)
        alpha = 1.0  # pheromone influence
        beta = 3.0   # heuristic influence (greedy to goal)

        values = np.zeros(n_directions)
        for i in range(n_directions):
            if not is_wall[i]:
                # Baseline pheromone of 1.0 ensures heuristic dominates when trail is empty
                tau = 1.0 + float(pher[rr[i], cc[i]])
                # Heuristic: inverse of distance to destination from candidate
                d_row = DESTINATION_POS[0] - new_r[i]
                d_col = DESTINATION_POS[1] - new_c[i]
                heur = 1.0 / (np.hypot(d_row, d_col) + 1e-3)
                values[i] = (tau ** alpha) * (heur ** beta)
                # Light exploration term to avoid dead-ends
                values[i] += EXPLORATION_BONUS / (1.0 + path_memory[rr[i], cc[i]])
                # Encourage movement that stays inside the guaranteed corridor
                if corridor_mask[rr[i], cc[i]]:
                    values[i] += CORRIDOR_BONUS
            else:
                values[i] = 0.0

        # Add tiny noise to break ties so argmax isn't deterministic
        values += 1e-6 * rng.random(n_directions)

        if np.sum(values) <= 0:
            # Fallback to any non-wall direction
            candidates = np.where(~is_wall)[0]
            k = rng.choice(candidates) if len(candidates) > 0 else 0
        else:
            if rng.random() < BIAS:
                k = int(np.argmax(values))
            else:
                probs = values / np.sum(values)
                k = int(rng.choice(np.arange(n_directions), p=probs))
    else:
        # Returning: prefer directions that follow pheromone and head to start
        values = np.zeros(n_directions)
        vec_to_start = START_POS - np.array([r, c])
        th_start = np.arctan2(vec_to_start[1], vec_to_start[0])
        for i in range(n_directions):
            if not is_wall[i]:
                tau = pher[rr[i], cc[i]]
                angle_diff = abs(angles[i] - th_start)
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                direction_bonus = np.exp(-angle_diff * 1.5)
                values[i] = 0.5 * tau + 1.5 * direction_bonus
            else:
                values[i] = 0.0
        k = int(np.argmax(values))

    return theta_candidates[k], dr[k], dc[k]

def in_disk(p, center, radius):
    d = np.linalg.norm(p - center, axis=1)
    return d <= radius

def diffuse(field, w=None):
    if w is None:
        w = DIFFUSE
    if w <= 0:
        return field
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    left = np.roll(field, 1, axis=1)
    right = np.roll(field, -1, axis=1)
    return (1 - w)*field + (w/4.0)*(up + down + left + right)

ANIMATE = True
fig = None
ax = None
im = None
scat = None

def init_plot():
    global fig, ax, im, scat
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pher, vmin=0, vmax=12, origin='lower', animated=True, cmap='viridis', alpha=0.8)
    scat = ax.scatter(ants[:,1], ants[:,0], s=25, c=['red' if s==0 else 'blue' for s in state], animated=True)
    ax.set_title("Ant Colony Pathfinding with Impassable Barriers\nRed=Searching, Blue=Returning")
    ax.set_xlim(0, W-1)
    ax.set_ylim(0, H-1)
    ax.set_xticks([]); ax.set_yticks([])

    # Draw walls as impassable barriers
    wall_y, wall_x = np.where(walls)
    ax.scatter(wall_x, wall_y, c='black', s=15, marker='s', alpha=0.8, label='Impassable Walls')
    # Highlight the carved corridor faintly for debugging (optional visual cue)
    cy, cx = np.where(corridor_mask)
    ax.scatter(cx, cy, c='white', s=2, alpha=0.05)

    # Draw start and destination points
    phi = np.linspace(0, 2*np.pi, 100)
    ax.plot(START_POS[1] + POINT_RADIUS*np.cos(phi), START_POS[0] + POINT_RADIUS*np.sin(phi), 
            'g-', linewidth=3, label='Start')
    ax.plot(DESTINATION_POS[1] + POINT_RADIUS*np.cos(phi), DESTINATION_POS[0] + POINT_RADIUS*np.sin(phi), 
            'r-', linewidth=3, label='Destination')
    ax.legend()

def step(_frame):
    global ants, pher, state, heading, found_destination, path_length, path_memory, prev_positions, stuck_counter, paths, return_index
    
    for i in range(N_ANTS):
        if state[i] == 1 and return_index[i] >= 0 and len(paths[i]) > 0:
            # Follow recorded path back deterministically to reinforce the exact route
            target_cell = paths[i][return_index[i]]
            target_r = target_cell[0] + 0.5
            target_c = target_cell[1] + 0.5
            vec_r = target_r - ants[i,0]
            vec_c = target_c - ants[i,1]
            dist = np.hypot(vec_r, vec_c)
            if dist > 1e-6:
                step_r = (vec_r / dist) * min(STEP_SIZE, dist)
                step_c = (vec_c / dist) * min(STEP_SIZE, dist)
                ants[i,0] += step_r
                ants[i,1] += step_c
            # Arrived at the target cell center
            if np.hypot(target_r - ants[i,0], target_c - ants[i,1]) < 0.2:
                return_index[i] -= 1
            # Deposit pheromone proportional to 1/L on this path
            L = max(len(paths[i]), 1)
            cr = int(np.clip(ants[i,0], 0, H-1))
            cc = int(np.clip(ants[i,1], 0, W-1))
            if not walls[cr, cc]:
                pher[cr, cc] += DEPOSIT / L
                path_memory[cr, cc] += 0.1
            path_length[i] += 1
        else:
            # Normal movement (searching or returning without a path)
            theta, dr, dc = choose_direction(i)
            heading[i] = 0.7*heading[i] + 0.3*theta  # inertia
            ants[i,0] += dr
            ants[i,1] += dc
            path_length[i] += 1

        # While searching, record path as unique lattice cells
        if state[i] == 0:
            cell_r = int(np.clip(ants[i,0], 0, H-1))
            cell_c = int(np.clip(ants[i,1], 0, W-1))
            if not walls[cell_r, cell_c]:
                if len(paths[i]) == 0 or paths[i][-1] != (cell_r, cell_c):
                    paths[i].append((cell_r, cell_c))
    
    clamp_positions(ants)
    
    # Simple stuck detection and reset
    movement = np.linalg.norm(ants - prev_positions, axis=1)
    stuck_ants = movement < 0.3  # ants that barely moved
    stuck_counter[stuck_ants] += 1
    stuck_counter[~stuck_ants] = 0  # reset counter for moving ants
    
    # Reset ants that have been stuck for too long
    reset_ants = stuck_counter > 15  # stuck for 15+ frames
    if np.any(reset_ants):
        # Teleport stuck ants to random positions near start
        for ant_idx in np.where(reset_ants)[0]:
            # Place them in a wider area around start
            new_r = rng.uniform(START_POS[0] - 8, START_POS[0] + 8)
            new_c = rng.uniform(START_POS[1] - 8, START_POS[1] + 8)
            ants[ant_idx] = [np.clip(new_r, 0, H-1), np.clip(new_c, 0, W-1)]
            state[ant_idx] = 0  # set to searching
            path_length[ant_idx] = 0
            stuck_counter[ant_idx] = 0
            heading[ant_idx] = rng.uniform(0, 2*np.pi)  # random new direction
    
    prev_positions[:] = ants[:]

    # Check if ants reach destination or start
    to_destination = in_disk(ants, DESTINATION_POS, POINT_RADIUS)
    to_start = in_disk(ants, START_POS, POINT_RADIUS)
    
    # Ants that reach destination: switch to returning, set return index
    newly_found = to_destination & (state == 0)
    if np.any(newly_found):
        idxs = np.where(newly_found)[0]
        for i in idxs:
            # Ensure destination cell is included as last waypoint
            dest_r = int(DESTINATION_POS[0])
            dest_c = int(DESTINATION_POS[1])
            if len(paths[i]) == 0 or paths[i][-1] != (dest_r, dest_c):
                paths[i].append((dest_r, dest_c))
            state[i] = 1
            found_destination[i] = True
            # Start returning from the end of recorded path
            return_index[i] = len(paths[i]) - 1
    
    # Ants that return to start: reset to searching and clear recorded path
    returned_ants = to_start & (state == 1)
    if np.any(returned_ants):
        state[returned_ants] = 0
        for i in np.where(returned_ants)[0]:
            paths[i].clear()
            return_index[i] = -1
            path_length[i] = 0

    # Evaporate and diffuse pheromones
    pher[:] = np.maximum((1.0 - EVAP) * pher, 0)
    pher[:] = diffuse(pher, DIFFUSE)
    
    # Slowly decay path memory
    path_memory[:] = np.maximum((1.0 - EVAP/10) * path_memory, 0)

    # Update visualization if enabled
    successful_ants = np.sum(found_destination)
    max_pheromone = float(np.max(pher))
    if ANIMATE:
        im.set_data(pher)
        colors = ['red' if s==0 else 'blue' for s in state]
        scat.set_color(colors)
        scat.set_offsets(np.column_stack((ants[:,1], ants[:,0])))
        cur_dest = int(np.sum(to_destination))
        cur_start = int(np.sum(to_start))
        ax.set_title(
            f"Ant Pathfinding - Frame {_frame}\n"
            f"EverSuccess: {successful_ants}/{N_ANTS}, AtDest: {cur_dest}, AtStart: {cur_start}, Max Pher: {max_pheromone:.2f}")
    
    # Print progress every 50 frames
    if _frame % 50 == 0:
        print(f"Frame {_frame}: {successful_ants} ants have found paths, Max pheromone: {max_pheromone:.2f}")
    
    return (im, scat) if ANIMATE else (None, None)

def reset_simulation(seed=42):
    global pher, path_memory, rng, ants, state, found_destination, path_length, prev_positions, stuck_counter, heading, paths, return_index, walls, corridor_mask
    pher = np.zeros(GRID, dtype=np.float32)
    path_memory = np.zeros(GRID, dtype=np.float32)
    rng = np.random.default_rng(seed)
    ants = np.vstack([
        rng.normal(START_POS[0], 2.0, size=N_ANTS),
        rng.normal(START_POS[1], 2.0, size=N_ANTS)
    ]).T
    state = np.zeros(N_ANTS, dtype=np.int8)
    found_destination = np.zeros(N_ANTS, dtype=bool)
    path_length = np.zeros(N_ANTS, dtype=int)
    prev_positions = np.copy(ants)
    stuck_counter = np.zeros(N_ANTS, dtype=int)
    heading = rng.uniform(0, 2*np.pi, size=N_ANTS)
    paths = [[] for _ in range(N_ANTS)]
    return_index = np.full(N_ANTS, -1, dtype=int)
    walls, corridor_mask = build_walls()
    # Seed a faint pheromone along the guaranteed corridor to accelerate discovery
    pher[corridor_mask] += 0.05
    path_memory[corridor_mask] += 0.2

def run_benchmark(frames=600, success_threshold=20, pher_threshold=6.0, seed=42):
    reset_simulation(seed)
    global ANIMATE
    ANIMATE = False
    lock_success_frame = None
    lock_pher_frame = None
    for f in range(frames):
        step(f)
        succ = int(np.sum(found_destination))
        maxp = float(np.max(pher))
        if lock_success_frame is None and succ >= success_threshold:
            lock_success_frame = f
        if lock_pher_frame is None and maxp >= pher_threshold:
            lock_pher_frame = f
        if lock_success_frame is not None and lock_pher_frame is not None:
            # Early stop when both conditions reached
            break
    return {
        'frames_run': f,
        'success_frame': lock_success_frame if lock_success_frame is not None else -1,
        'pheromone_frame': lock_pher_frame if lock_pher_frame is not None else -1,
        'final_success': int(np.sum(found_destination)),
        'final_max_pher': float(np.max(pher)),
    }

def apply_params(params):
    global STEP_SIZE, BIAS, DEPOSIT, EVAP, DIFFUSE
    for k, v in params.items():
        if k == 'STEP_SIZE': STEP_SIZE = v
        elif k == 'BIAS': BIAS = v
        elif k == 'DEPOSIT': DEPOSIT = v
        elif k == 'EVAP': EVAP = v
        elif k == 'DIFFUSE': DIFFUSE = v
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['anim','bench'], default='anim')
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--bias', type=float)
    parser.add_argument('--deposit', type=float)
    parser.add_argument('--evap', type=float)
    parser.add_argument('--diffuse', type=float)
    parser.add_argument('--step-size', type=float)
    parser.add_argument('--radius', type=int, help='Arrival radius for start/destination disks')
    parser.add_argument('--export', type=str, help='Path to save animation (mp4 or gif)')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second for export')
    args = parser.parse_args()

    global POINT_RADIUS
    overrides = {}
    if args.bias is not None: overrides['BIAS'] = args.bias
    if args.deposit is not None: overrides['DEPOSIT'] = args.deposit
    if args.evap is not None: overrides['EVAP'] = args.evap
    if args.diffuse is not None: overrides['DIFFUSE'] = args.diffuse
    if args.step_size is not None: overrides['STEP_SIZE'] = args.step_size
    if args.radius is not None: POINT_RADIUS = int(args.radius)
    if overrides:
        apply_params(overrides)

    if args.mode == 'bench':
        # Two presets to compare
        presets = {
            'A': {'BIAS': 0.88, 'DEPOSIT': 100.0, 'EVAP': 0.012, 'DIFFUSE': 0.02, 'STEP_SIZE': 1.0},
            'B': {'BIAS': 0.94, 'DEPOSIT': 160.0, 'EVAP': 0.012, 'DIFFUSE': 0.01, 'STEP_SIZE': 1.0},
        }
        results = {}
        for name, p in presets.items():
            apply_params(p)
            res = run_benchmark(frames=args.steps, success_threshold=20, pher_threshold=6.0, seed=42)
            results[name] = (p, res)
        print("Benchmark results (lower is better):")
        for name, (p, r) in results.items():
            print(f"Preset {name}: bias={p['BIAS']}, deposit={p['DEPOSIT']}, evap={p['EVAP']}, diffuse={p['DIFFUSE']}")
            print(f"  frames_to_20_success: {r['success_frame']}, frames_to_maxpher>=6: {r['pheromone_frame']}, final_success={r['final_success']}, final_max_pher={r['final_max_pher']:.2f}")
        # Pick winner by min of success_frame then pheromone_frame
        winner = min(results.items(), key=lambda kv: (
            kv[1][1]['success_frame'] if kv[1][1]['success_frame'] != -1 else 1e9,
            kv[1][1]['pheromone_frame'] if kv[1][1]['pheromone_frame'] != -1 else 1e9
        ))
        print(f"Winner: Preset {winner[0]}")
        return

    # Animation mode
    global ANIMATE
    ANIMATE = True
    init_plot()
    anim = FuncAnimation(fig, step, frames=args.steps, interval=40, blit=True, repeat=False)

    if args.export:
        out = args.export
        ext = out.split('.')[-1].lower() if '.' in out else ''
        writers = mpl_animation.writers.list()
        writer = None
        if ext == 'mp4' and 'ffmpeg' in writers:
            writer = mpl_animation.FFMpegWriter(fps=args.fps, bitrate=3000, codec='libx264')
        elif ext == 'gif' and 'pillow' in writers:
            writer = mpl_animation.PillowWriter(fps=args.fps)
        elif 'pillow' in writers:
            # fallback to GIF if requested writer unavailable
            out = (out.rsplit('.', 1)[0] if '.' in out else out) + '.gif'
            writer = mpl_animation.PillowWriter(fps=args.fps)
            print(f"ffmpeg unavailable or extension unsupported. Falling back to GIF: {out}")
        else:
            print('No suitable writer found (need ffmpeg for mp4 or pillow for gif). Showing animation window instead.')
            plt.show()
            return

        anim.save(out, writer=writer, dpi=150)
        print(f"Saved animation to {out}")
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    main()
