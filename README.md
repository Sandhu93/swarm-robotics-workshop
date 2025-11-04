# Swarm Robotics Workshop

A collection of swarm intelligence algorithms and simulations implemented in Python, demonstrating emergent behaviors and collective intelligence.

## 🐜 Projects Included

### 1. Ant Colony Pathfinding (`01_ants.py`)
- **Ant Colony Optimization (ACO)** for pathfinding
- Ants find shortest paths from start to destination using **pheromone trails**
- **Obstacle avoidance** with impassable barriers
- **Swarm intelligence** - optimal paths emerge from simple local rules
- Features: Dynamic barrier navigation, stuck-ant recovery, real-time visualization

### 2. Boids Flocking Simulation (`03_flocking.py`)
- **Craig Reynolds' Boids algorithm** for realistic flocking behavior
- **Three core rules**: Separation, Alignment, Cohesion
- **Obstacle avoidance** with split-and-merge behavior
- **Emergent patterns**: Flocking, swarming, streaming formations
- Features: Multiple presets, configurable parameters, animation export

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running the Simulations

#### Ant Colony Pathfinding
```bash
python 01_ants.py
```
Watch ants discover and optimize paths around barriers using pheromone communication!

#### Boids Flocking
```bash
python 03_flocking.py --preset split --agents 120
```

**Available presets:**
- `tight`: Dense flocking behavior
- `loose`: Relaxed, spread-out movement
- `split`: Demonstrates split-and-merge around obstacles
- `center`: Single central obstacle

**Export animations:**
```bash
python 03_flocking.py --export flock_animation.gif --steps 500
```

## 🧠 Key Concepts Demonstrated

### Swarm Intelligence Principles
- **Emergence**: Complex behaviors from simple rules
- **Self-organization**: No central control needed
- **Stigmergy**: Indirect coordination through environment modification (pheromones)
- **Collective problem-solving**: Group intelligence exceeds individual capabilities

### Algorithms Implemented
- **Ant Colony Optimization (ACO)**: Bio-inspired pathfinding and optimization
- **Boids Algorithm**: Flocking and group movement simulation
- **Obstacle avoidance**: Dynamic navigation around barriers
- **Pheromone-based communication**: Chemical signaling simulation

## 🎯 Applications

These algorithms have real-world applications in:
- **Multi-robot coordination**
- **Traffic flow optimization**
- **Supply chain and logistics**
- **Computer graphics and game AI**
- **Distributed sensor networks**
- **Crowd simulation and evacuation planning**

## 📊 Parameters and Customization

### Ant Colony (`01_ants.py`)
- `N_ANTS`: Number of ants in the colony
- `BIAS`: Probability of following pheromone trails vs exploration
- `DEPOSIT`: Pheromone strength deposited by successful ants
- `EVAP`: Pheromone evaporation rate
- Barrier configuration through `walls` array

### Boids (`03_flocking.py`)
- `--agents`: Number of boids
- `--radius`: Neighborhood detection radius
- `--sep`, `--align`, `--coh`: Weights for the three core behaviors
- `--max-speed`, `--max-force`: Movement constraints
- `--wrap` / `--no-wrap`: Toroidal vs bounded environment

## 🔬 Scientific Background

This project demonstrates fundamental principles from:
- **Swarm Intelligence** (Bonabeau, Dorigo, Theraulaz)
- **Artificial Life** (Craig Reynolds' Boids, 1987)
- **Bio-inspired Computing** (Ant Colony Optimization)
- **Complex Adaptive Systems**
- **Emergent Behavior in Multi-Agent Systems**

## 🎨 Visualization Features

- **Real-time animation** of agent behaviors
- **Pheromone trail visualization** (ant simulation)
- **Dynamic obstacle interaction**
- **Parameter adjustment during runtime**
- **Export capabilities** for creating presentations and demonstrations

## 🤝 Contributing

Feel free to experiment with:
- New swarm algorithms (Particle Swarm Optimization, Fish Schooling, etc.)
- Different obstacle configurations
- Parameter optimization studies
- 3D extensions of the algorithms
- Performance benchmarking

## 📚 References

- Reynolds, C. W. (1987). "Flocks, herds and schools: A distributed behavioral model"
- Dorigo, M. & Stützle, T. (2004). "Ant Colony Optimization"
- Bonabeau, E., Dorigo, M., & Theraulaz, G. (1999). "Swarm Intelligence: From Natural to Artificial Systems"

---

**Author**: Swarm Robotics Workshop  
**Date**: November 2025  
**License**: MIT