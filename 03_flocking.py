import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation as mpl_animation
import argparse


def limit_norm(v, max_val):
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    scale = np.minimum(1.0, max_val / n)
    return v * scale


class Boids:
    def __init__(self, n=120, width=100, height=100, seed=3,
                 radius=12.0, sep_radius=6.0,
                 w_sep=1.2, w_align=0.8, w_coh=0.5, w_avoid=2.0,
                 max_speed=2.0, max_force=0.06,
                 wrap=True,
                 obstacles=None):
        self.n = n
        self.W = width
        self.H = height
        self.radius = radius
        self.sep_radius = sep_radius
        self.w_sep = w_sep
        self.w_align = w_align
        self.w_coh = w_coh
        self.w_avoid = w_avoid
        self.max_speed = max_speed
        self.max_force = max_force
        self.wrap = wrap
        self.rng = np.random.default_rng(seed)

        self.pos = np.column_stack([
            self.rng.uniform(10, self.H-10, size=n),
            self.rng.uniform(10, self.W-10, size=n),
        ]).astype(np.float32)
        theta = self.rng.uniform(0, 2*np.pi, size=n)
        self.vel = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)
        self.vel *= self.max_speed * 0.5

        # obstacles: list of (row, col, radius)
        if obstacles is None:
            self.obstacles = [(50.0, 50.0, 8.0)]  # central obstacle to show split/merge
        else:
            self.obstacles = obstacles

    def neighbors(self):
        # compute pairwise distances
        d = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.linalg.norm(d, axis=2) + 1e-9
        mask = dist < self.radius
        np.fill_diagonal(mask, False)
        close = dist < self.sep_radius
        np.fill_diagonal(close, False)
        return d, dist, mask, close

    def obstacle_avoidance(self):
        if not self.obstacles:
            return np.zeros_like(self.pos)
        force = np.zeros_like(self.pos)
        for (r, c, rad) in self.obstacles:
            vec = self.pos - np.array([r, c])
            dist = np.linalg.norm(vec, axis=1) + 1e-9
            influence = np.clip((rad + 6.0 - dist) / (rad + 6.0), 0.0, 1.0)
            away = vec / dist[:, None]
            force += (away * influence[:, None])
        return force

    def step(self):
        d, dist, mask, close = self.neighbors()

        # Separation: steer away from neighbors inside sep_radius (stronger when closer)
        sep_dir = np.where(close[:, :, None], d, 0.0)
        # away from neighbors: subtract vector (current minus neighbor) to push apart
        sep = np.sum(sep_dir / (dist[:, :, None] + 1e-6), axis=1)
        sep = -sep

        # Alignment: match average heading of neighbors
        align = np.zeros_like(self.pos)
        counts = mask.sum(axis=1)[:, None] + 1e-9
        if np.any(mask):
            avg_vel = (mask @ self.vel) / counts
            align = avg_vel - self.vel

        # Cohesion: steer toward neighbors' center
        coh = np.zeros_like(self.pos)
        if np.any(mask):
            center = (mask @ self.pos) / counts
            coh = center - self.pos

        # Obstacle avoidance
        avoid = self.obstacle_avoidance()

        # Combine
        acc = (self.w_sep * sep +
               self.w_align * align +
               self.w_coh * coh +
               self.w_avoid * avoid)

        # Limit force and update velocity
        acc = limit_norm(acc, self.max_force)
        self.vel = limit_norm(self.vel + acc, self.max_speed)

        # Integrate position
        self.pos += self.vel

        if self.wrap:
            self.pos[:, 0] = np.mod(self.pos[:, 0], self.H)
            self.pos[:, 1] = np.mod(self.pos[:, 1], self.W)
        else:
            for ax, m in [(0, self.H), (1, self.W)]:
                hit_low = self.pos[:, ax] < 0
                hit_high = self.pos[:, ax] > m
                self.pos[hit_low, ax] = 0
                self.pos[hit_high, ax] = m
                self.vel[hit_low | hit_high, ax] *= -1


def build_obstacles(preset, W, H):
    if preset == 'none':
        return []
    if preset == 'center':
        return [(H/2, W/2, 8.0)]
    if preset == 'split':
        return [(H/2, W/2, 10.0), (H/2, W*0.35, 6.0), (H/2, W*0.65, 6.0)]
    # default
    return [(H/2, W/2, 8.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agents', type=int, default=120)
    ap.add_argument('--size', type=int, nargs=2, default=[100, 100], help='H W')
    ap.add_argument('--radius', type=float, default=12.0)
    ap.add_argument('--sep-radius', type=float, default=6.0)
    ap.add_argument('--sep', type=float, default=1.2)
    ap.add_argument('--align', type=float, default=0.8)
    ap.add_argument('--coh', type=float, default=0.5)
    ap.add_argument('--avoid', type=float, default=2.0)
    ap.add_argument('--max-speed', type=float, default=2.0)
    ap.add_argument('--max-force', type=float, default=0.06)
    ap.add_argument('--wrap', action='store_true', help='Enable toroidal wrap (default)')
    ap.add_argument('--no-wrap', dest='wrap', action='store_false', help='Disable wrap, bounce on edges')
    ap.set_defaults(wrap=True)
    ap.add_argument('--preset', choices=['tight','loose','split','none','center'], default='split')
    ap.add_argument('--steps', type=int, default=700)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--export', type=str, help='Save animation to mp4/gif')
    ap.add_argument('--fps', type=int, default=25)
    args = ap.parse_args()

    H, W = args.size

    # Presets to match slide narrative
    if args.preset == 'tight':
        args.sep, args.align, args.coh = 1.6, 1.2, 0.9
        args.radius, args.sep_radius = 14.0, 7.0
    elif args.preset == 'loose':
        args.sep, args.align, args.coh = 0.9, 0.5, 0.3
        args.radius, args.sep_radius = 10.0, 5.0
    elif args.preset == 'split':
        args.sep, args.align, args.coh = 1.3, 0.9, 0.6
        args.radius, args.sep_radius = 12.0, 6.0

    obstacles = build_obstacles(args.preset, W, H)

    boids = Boids(
        n=args.agents, width=W, height=H, seed=args.seed,
        radius=args.radius, sep_radius=args.sep_radius,
        w_sep=args.sep, w_align=args.align, w_coh=args.coh, w_avoid=args.avoid,
        max_speed=args.max_speed, max_force=args.max_force,
        wrap=args.wrap, obstacles=obstacles)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    scat = ax.scatter(boids.pos[:, 1], boids.pos[:, 0], s=18, c='tab:blue', alpha=0.85)
    ax.set_xlim(0, boids.W)
    ax.set_ylim(0, boids.H)
    ax.set_xticks([]); ax.set_yticks([])
    ttitle = ax.set_title('Boids — Separation, Alignment, Cohesion')

    # Draw obstacles
    theta = np.linspace(0, 2*np.pi, 150)
    for (r, c, rad) in boids.obstacles:
        ax.plot(c + rad*np.cos(theta), r + rad*np.sin(theta), 'k--', lw=1)

    def update(frame):
        boids.step()
        scat.set_offsets(np.column_stack((boids.pos[:, 1], boids.pos[:, 0])))
        ttitle.set_text(
            f"Boids — Sep {boids.w_sep:.1f}, Align {boids.w_align:.1f}, Coh {boids.w_coh:.1f}, R {boids.radius:.0f}\n"
            f"Preset: {args.preset}, Agents: {boids.n}")
        return scat, ttitle

    anim = FuncAnimation(fig, update, frames=args.steps, interval=40, blit=False, repeat=False)

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
            out = (out.rsplit('.', 1)[0] if '.' in out else out) + '.gif'
            writer = mpl_animation.PillowWriter(fps=args.fps)
            print(f"ffmpeg unavailable or extension unsupported. Falling back to GIF: {out}")
        else:
            print('No suitable writer found (ffmpeg for mp4 or pillow for gif). Showing window instead.')
            plt.show()
            return
        anim.save(out, writer=writer, dpi=150)
        print(f"Saved animation to {out}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()

