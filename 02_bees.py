import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation as mpl_animation
import argparse

rng = np.random.default_rng(7)

# Parameters
GRID = (100, 100)
N_BEES = 60
STEPS = 700
DT = 1.0
STEP_SIZE = 0.9
INTERVAL_MS = 40  # animation interval (ms); used to compute seconds

# Areas (row, col)
H, W = GRID
HIVE = np.array([50.0, 50.0])
BROOD = np.array([50.0, 35.0])
RESOURCE = np.array([50.0, 85.0])
SITE_A = np.array([20.0, 80.0])
SITE_B = np.array([80.0, 80.0])
R_HIVE = 6
R_BROOD = 5
R_RESOURCE = 5
R_SITE = 4

# Decision parameters
DECISION_START = 200
SCOUT_FRACTION = 0.2
QUORUM_FRACTION = 0.60
SITE_QUALITY_A = 0.70
SITE_QUALITY_B = 0.62

# Need dynamics
STORE_MAX = 120.0
STORE_DECAY = 0.25
DELIVERY_GAIN = 2.5
BROOD_INC = 0.006
BROOD_DECAY_PER_NURSE = 0.012

# States
# 0=rest, 1=forage, 2=nurse, 3=scout

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def clamp_pos(p):
    p[:, 0] = np.clip(p[:, 0], 0, H - 1e-6)
    p[:, 1] = np.clip(p[:, 1], 0, W - 1e-6)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--ants', type=int, default=N_BEES)  # keep name simple
    ap.add_argument('--quorum', type=float, default=QUORUM_FRACTION)
    ap.add_argument('--export', type=str, help='Path to save animation (mp4 or gif)')
    ap.add_argument('--fps', type=int, default=25, help='Frames per second for export')
    args = ap.parse_args()

    n = args.ants
    quorum_fraction = args.quorum

    pos = np.vstack([
        rng.normal(HIVE[0], 3.0, size=n),
        rng.normal(HIVE[1], 3.0, size=n)
    ]).T
    vel = rng.normal(0, 0.5, size=(n, 2))

    state = np.zeros(n, dtype=np.int8)
    thresh_forage = rng.uniform(0.2, 0.9, size=n)
    thresh_nurse = rng.uniform(0.2, 0.9, size=n)

    carrying = np.zeros(n, dtype=bool)
    store = 60.0
    brood_need = 0.4

    # Decision variables
    is_scout = np.zeros(n, dtype=bool)
    vote_A = 0.0
    vote_B = 0.0
    commit_site = None  # 'A' or 'B'
    dance_intensity_A = 0.0
    dance_intensity_B = 0.0

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    left, right = ax
    scat = left.scatter(pos[:, 1], pos[:, 0], s=25, c='gold', alpha=0.9)
    left.set_xlim(0, W)
    left.set_ylim(0, H)
    left.set_title('Bees: Task Allocation and Quorum')
    left.set_xticks([]); left.set_yticks([])

    # Markers
    theta = np.linspace(0, 2*np.pi, 120)
    left.plot(HIVE[1] + R_HIVE*np.cos(theta), HIVE[0] + R_HIVE*np.sin(theta), 'k-', lw=2, label='Hive')
    left.plot(BROOD[1] + R_BROOD*np.cos(theta), BROOD[0] + R_BROOD*np.sin(theta), 'b--', lw=1, label='Brood')
    left.plot(RESOURCE[1] + R_RESOURCE*np.cos(theta), RESOURCE[0] + R_RESOURCE*np.sin(theta), 'g--', lw=1, label='Flowers')
    left.plot(SITE_A[1] + R_SITE*np.cos(theta), SITE_A[0] + R_SITE*np.sin(theta), 'm:', lw=1, label='Site A')
    left.plot(SITE_B[1] + R_SITE*np.cos(theta), SITE_B[0] + R_SITE*np.sin(theta), 'c:', lw=1, label='Site B')
    left.legend(loc='upper left', fontsize=8)

    bars = right.bar(['Rest', 'Forage', 'Nurse', 'Scout'], [0, 0, 0, 0], color=['gray','green','blue','magenta'])
    right.set_ylim(0, max(10, n))
    right.set_title('Counts and Decision')
    txt = right.text(0.02, 0.95, '', transform=right.transAxes, va='top', fontsize=10)
    # Highlighted time-to-quorum indicator
    qtxt = right.text(0.5, 0.02, '', transform=right.transAxes, va='bottom', ha='center', fontsize=12, fontweight='bold')

    def in_disk(p, c, r):
        return np.linalg.norm(p - c, axis=1) <= r

    def step(frame):
        nonlocal pos, vel, store, brood_need, is_scout, vote_A, vote_B, commit_site, dance_intensity_A, dance_intensity_B
        # Track when quorum is reached (in frames)
        if not hasattr(step, 'commit_frame'):
            step.commit_frame = None

        # Needs based on current status
        need_forage = max(0.0, 1.0 - store / STORE_MAX)
        need_nurse = brood_need

        # Determine tasks via response thresholds for non-scouts
        if frame < DECISION_START and commit_site is None:
            # Compute stimulus over threshold
            stim_f = need_forage - thresh_forage
            stim_n = need_nurse - thresh_nurse
            choose_f = stim_f > 0
            choose_n = stim_n > 0
            # Pick dominant stimulus; ties prefer forage when store is low
            prefer_f = stim_f >= stim_n
            new_state = np.zeros_like(state)
            new_state[choose_f & prefer_f] = 1
            new_state[choose_n & ~prefer_f] = 2
            state[:] = np.where(is_scout, 3, new_state)
        else:
            # Decision phase: some resters become scouts; scouts remain scouts until commit
            state[:] = np.where(is_scout, 3, state)

        # Recruitment and dancing during decision phase
        if frame >= DECISION_START and commit_site is None:
            target_scouts = int(np.ceil(SCOUT_FRACTION * n))
            # Turn some resters into scouts
            resters = np.where((state == 0) & (~is_scout))[0]
            to_promote = min(len(resters), max(0, target_scouts - int(is_scout.sum())))
            if to_promote > 0:
                idx = rng.choice(resters, size=to_promote, replace=False)
                is_scout[idx] = True
                state[idx] = 3

            # Scouts evaluate and advertise
            scouts = np.where(is_scout)[0]
            if len(scouts) > 0:
                quality_A = SITE_QUALITY_A + rng.normal(0, 0.03, size=len(scouts))
                quality_B = SITE_QUALITY_B + rng.normal(0, 0.03, size=len(scouts))
                # Probability to choose A vs B, softmax by quality and existing votes (positive feedback)
                bias_A = vote_A / (vote_A + vote_B + 1e-6)
                logits_A = 2.5*quality_A + 1.5*bias_A
                logits_B = 2.5*quality_B + 1.5*(1 - bias_A)
                pA = 1.0 / (1.0 + np.exp(-(logits_A - logits_B)))
                chooseA = rng.random(len(scouts)) < pA
                vote_A += chooseA.sum()
                vote_B += (~chooseA).sum()
                dance_intensity_A = 0.9*dance_intensity_A + 0.1*chooseA.sum()
                dance_intensity_B = 0.9*dance_intensity_B + 0.1*(~chooseA).sum()

                quorum = int(np.ceil(quorum_fraction * max(1, len(scouts))))
                if vote_A >= quorum:
                    commit_site = 'A'
                    if step.commit_frame is None:
                        step.commit_frame = frame
                elif vote_B >= quorum:
                    commit_site = 'B'
                    if step.commit_frame is None:
                        step.commit_frame = frame

        # Movement
        colors = np.empty(n, dtype=object)
        for i in range(n):
            if state[i] == 1:  # forage
                colors[i] = 'green'
                target = RESOURCE if not carrying[i] else HIVE
                d = unit(target - pos[i])
                pos[i] += STEP_SIZE * d
                if in_disk(pos[i:i+1], target, R_RESOURCE if not carrying[i] else R_HIVE)[0]:
                    if not carrying[i]:
                        carrying[i] = True
                    else:
                        carrying[i] = False
                        store = min(STORE_MAX, store + DELIVERY_GAIN)
            elif state[i] == 2:  # nurse
                colors[i] = 'blue'
                d = unit(BROOD - pos[i])
                pos[i] += STEP_SIZE * d
            elif state[i] == 3:  # scout
                colors[i] = 'magenta'
                if commit_site is None:
                    # wander near hive and sites
                    attractors = np.array([HIVE, SITE_A, SITE_B])
                    wts = np.array([0.4, 0.3, 0.3])
                    target = attractors[rng.choice([0,1,2], p=wts)]
                else:
                    target = SITE_A if commit_site == 'A' else SITE_B
                d = unit(target - pos[i]) + 0.3*rng.normal(0,1,2)
                pos[i] += STEP_SIZE * unit(d)
            else:  # rest
                colors[i] = 'gray'
                d = unit(HIVE - pos[i]) + 0.2*rng.normal(0,1,2)
                pos[i] += 0.6 * unit(d)

        clamp_pos(pos)

        # Need dynamics updates
        store = max(0.0, store - STORE_DECAY)
        n_nurse = int((state == 2).sum())
        brood_need = np.clip(brood_need + BROOD_INC - BROOD_DECAY_PER_NURSE * n_nurse, 0.0, 1.0)

        # Counts for bars
        counts = [int((state == s).sum()) for s in (0,1,2,3)]
        for rect, val in zip(bars, counts):
            rect.set_height(val)
        bars[0].set_color('gray'); bars[1].set_color('green'); bars[2].set_color('blue'); bars[3].set_color('magenta')

        scat.set_offsets(np.column_stack((pos[:, 1], pos[:, 0])))
        scat.set_color(colors)

        # Right text
        # Quorum timing (seconds)
        sec_per_frame = INTERVAL_MS / 1000.0
        if frame >= DECISION_START and step.commit_frame is None:
            elapsed = max(0, frame - DECISION_START) * sec_per_frame
            qtxt.set_text(f"Quorum pending: {elapsed:.1f}s")
            qtxt.set_color('tab:red')
            qtxt.set_bbox(dict(facecolor='white', edgecolor='tab:red', alpha=0.7, boxstyle='round'))
        elif step.commit_frame is not None:
            ttq = max(0, step.commit_frame - DECISION_START) * sec_per_frame
            qtxt.set_text(f"Time to Quorum: {ttq:.1f}s")
            qtxt.set_color('white')
            qtxt.set_bbox(dict(facecolor='tab:green', edgecolor='black', alpha=0.8, boxstyle='round'))
        else:
            qtxt.set_text("")
            qtxt.set_bbox(None)

        txt.set_text(
            f"Forage need: {max(0,1-store/STORE_MAX):.2f}\n"
            f"Brood need: {brood_need:.2f}\n"
            f"Votes A/B: {int(vote_A)}/{int(vote_B)}\n"
            f"Dance A/B: {dance_intensity_A:.1f}/{dance_intensity_B:.1f}\n"
            f"Commit: {commit_site if commit_site else '-'}\n"
        )

        left.set_xlabel(f"Frame {frame} — Store {store:.1f}")
        return scat, bars, txt, qtxt

    anim = FuncAnimation(fig, step, frames=args.steps, interval=INTERVAL_MS, blit=False, repeat=False)
    plt.tight_layout()

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
