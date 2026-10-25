# AI Search & Pathfinding

Implementations of graph search algorithms for a custom grid environment, developed as part of an Artificial Intelligence course at the Technion.

The project explores how different search strategies select states and construct paths to a goal.

## Implemented Algorithms

- **Breadth-First Search:** explores states in FIFO order while avoiding repeated states.
- **Greedy Best-First Search:** prioritizes states using a goal-distance heuristic.
- **A*:** combines accumulated path cost with a heuristic estimate.
- **A*-epsilon:** selects states from a FOCAL set controlled by an epsilon parameter.

## Implementation Details

- Parent tracking to reconstruct action sequences.
- Tracking of accumulated path costs.
- Filtering of terminal failure states.
- Deterministic tie-breaking.
- Reopening states when A*-epsilon discovers a lower-cost path.
- A Manhattan-distance heuristic capped by the environment's passway cost.

Each search method returns an action sequence, its cost, and a frontier-insertion count. The counter is named `expanded` in the implementation and follows the assignment's counting convention.

## Files

| File | Purpose |
|---|---|
| `Algorithms.py` | Search agents and shared helper functions |

## Dependencies

- Python 3
- NumPy
- heapdict — recommended; a fallback implementation is included
- The course-provided `HaifaEnv` module

Install the external packages:

    pip install numpy heapdict

## Usage

The agents operate on a configured `HaifaEnv` instance:

    from Algorithms import AStarAgent

    agent = AStarAgent()
    actions, cost, frontier_insertions = agent.search(env)

Here, `env` must be initialized using the course environment.

## Current Packaging Status

The uploaded submission includes the search implementations but does not include `HaifaEnv.py`. The environment is required to run the agents.

## Academic Context

Developed as a course assignment at the Technion. The search implementations integrate with a course-provided environment.
