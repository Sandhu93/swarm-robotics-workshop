# Swarm Robotics Simulations

This repository contains simple Python simulations that demonstrate swarm intelligence and emergent behavior. Each script is self-contained and can be run from the command line.

The main simulations are:
- Ant Colony pathfinding (`01_ants.py`)
- Bees task allocation and quorum (`02_bees.py`)
- Boids flocking (`03_flocking.py`)

All instructions use plain, simple English and show both Windows PowerShell and macOS/Linux shell examples.

**Requirements**
- Python 3.9+ recommended
- Packages: `numpy`, `matplotlib`

Install packages:
- Windows PowerShell: `pip install numpy matplotlib`
- macOS/Linux: `pip3 install numpy matplotlib`

Optional for saving videos:
- MP4 export needs `ffmpeg` installed on your system
- GIF export works if Pillow is available (usually bundled with `matplotlib`)

**Create a virtual environment (optional but recommended)**
- Windows PowerShell
  - `python -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`
  - `pip install numpy matplotlib`
- macOS/Linux
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install numpy matplotlib`

**Quick Start**
- Ants (default animation):
  - Windows: `python 01_ants.py`
  - macOS/Linux: `python3 01_ants.py`
- Bees (default animation):
  - Windows: `python 02_bees.py`
  - macOS/Linux: `python3 02_bees.py`
- Boids (split preset):
  - Windows: `python 03_flocking.py --preset split --agents 120`
  - macOS/Linux: `python3 03_flocking.py --preset split --agents 120`

**Running Headless (no display, e.g., servers/CI)**
- Use the non-interactive backend and export to a file
- Windows PowerShell
  - `$env:MPLBACKEND = 'Agg'`
  - `python 03_flocking.py --steps 500 --export flock.gif`
- macOS/Linux
  - `export MPLBACKEND=Agg`
  - `python3 03_flocking.py --steps 500 --export flock.gif`

**Saving Animations**
- Save to GIF: add `--export output.gif`
- Save to MP4: add `--export output.mp4` and ensure `ffmpeg` is installed
- Control frame rate: `--fps 25`

Examples:
- Ants to GIF: `python 01_ants.py --steps 400 --export ants.gif`
- Bees to MP4: `python 02_bees.py --steps 600 --export bees.mp4 --fps 30`
- Boids to GIF: `python 03_flocking.py --preset tight --export boids.gif`

**Ant Colony Pathfinding (`01_ants.py`)**
- Purpose: ants discover and reinforce a path from start to destination using pheromone trails and evaporation
- Basic run: `python 01_ants.py`
- Modes
  - Animation: `--mode anim` (default)
  - Benchmark: `--mode bench` (prints preset comparison, no window)
- Useful arguments
  - `--steps N` total frames to simulate
  - `--bias F` exploit vs. explore (0–1)
  - `--deposit F` pheromone deposited by successful ants
  - `--evap F` evaporation rate per step
  - `--diffuse F` trail diffusion weight
  - `--step-size F` movement per frame
  - `--radius N` arrival radius for start/destination
  - `--export PATH` save animation to GIF/MP4
  - `--fps N` frames per second for export
- Examples
  - Explore more: `python 01_ants.py --bias 0.8`
  - Faster evaporation: `python 01_ants.py --evap 0.02`
  - Headless export: `python 01_ants.py --steps 500 --export ants.gif`

**Bees: Task Allocation and Quorum (`02_bees.py`)**
- Purpose: bees switch tasks (rest, forage, nurse) based on needs; some scouts discover sites; a quorum triggers a final choice
- Basic run: `python 02_bees.py`
- Useful arguments
  - `--steps N` total frames to simulate
  - `--ants N` number of bees
  - `--quorum F` fraction needed to commit to a site
  - `--export PATH` save animation to GIF/MP4
  - `--fps N` frames per second for export
- Examples
  - More bees: `python 02_bees.py --ants 120`
  - Stricter quorum: `python 02_bees.py --quorum 0.75`
  - Save: `python 02_bees.py --steps 800 --export bees_quorum.gif`

**Boids Flocking (`03_flocking.py`)**
- Purpose: classic flocking with separation, alignment, and cohesion; includes simple obstacle avoidance
- Basic run: `python 03_flocking.py`
- Presets
  - `tight` dense flocking
  - `loose` spread out movement
  - `split` split-and-merge around an obstacle (default)
  - `center` single central obstacle
- Useful arguments
  - `--agents N` number of boids
  - `--size H W` simulation height and width
  - `--radius F` neighborhood radius
  - `--sep-radius F` separation radius
  - `--sep F`, `--align F`, `--coh F` behavior weights
  - `--avoid F` obstacle avoidance weight
  - `--max-speed F`, `--max-force F` limits
  - `--wrap` or `--no-wrap` world boundaries
  - `--steps N`, `--export PATH`, `--fps N`
- Examples
  - Default preset: `python 03_flocking.py --preset split --agents 120`
  - No wrap, bounce at edges: `python 03_flocking.py --no-wrap`
  - Larger field: `python 03_flocking.py --size 150 150 --agents 200`

**Troubleshooting**
- No window appears
  - Ensure a desktop session is available or use headless export with `MPLBACKEND=Agg` and `--export`
- MP4 export fails
  - Install `ffmpeg`, or export to GIF instead
- Slow rendering
  - Reduce `--agents`, lower `--steps`, or shrink `--size`
- Import errors
  - Activate your virtual environment and reinstall dependencies

**Repository Notes**
- `ants.gif`, `bees_quorum.gif`, `flocking.gif` are example outputs
- `bench_probe.py`, `bench_probe_nowalls.py` show how to step the ants simulation without a window
  - Run: `python bench_probe.py` or `python bench_probe_nowalls.py`

**Background**
- Ant Colony Optimization (Dorigo, Stützle)
- Boids (Craig Reynolds, 1987)
- Swarm intelligence and emergent behavior

License: MIT
