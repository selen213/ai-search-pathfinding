import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple

try:
    from heapdict import heapdict
except ImportError:
    class heapdict(dict):
        """Small fallback used only if heapdict is unavailable locally."""
        def peekitem(self):
            if not self:
                raise KeyError("peekitem(): heapdict is empty")
            return min(super().items(), key=lambda item: item[1])

        def popitem(self):
            key, value = self.peekitem()
            del self[key]
            return key, value


# ------------------------------------------------------------
# Helper functions used by all the search agents.
# I separated these parts so the actual algorithms below are easier to read.
# ------------------------------------------------------------

PASSWAY_COST = 100.0


def _num_actions(env: HaifaEnv) -> int:
    """The environment has 4 actions, but I read it from action_space when possible."""
    return getattr(env.action_space, "n", 4)


def _is_forbidden_terminal(env: HaifaEnv, state, terminated: bool) -> bool:
    """
    Holes are terminal failure states.
    I skip them in the search frontier, because they cannot be part of a valid solution path.
    """
    return terminated and (state is None or not env.is_final_state(int(state)))


def _reconstruct_path(parent, goal_state: int) -> List[int]:
    """Go backwards through the parents dictionary and recover the action path."""
    actions = []
    state = goal_state

    while parent[state][0] is not None:
        prev_state, action = parent[state]
        actions.append(action)
        state = prev_state

    actions.reverse()
    return actions


def _haifa_heuristic(env: HaifaEnv, state: int) -> float:
    """
    The heuristic from the homework:
    minimum Manhattan distance to a goal, capped by the passway cost.
    """
    goals = env.get_goal_states()
    if len(goals) == 0:
        return 0.0

    row, col = env.to_row_col(int(state))
    min_dist = np.inf

    for goal in goals:
        goal_row, goal_col = env.to_row_col(int(goal))
        dist = abs(row - goal_row) + abs(col - goal_col)
        min_dist = min(min_dist, dist)

    return float(min(min_dist, PASSWAY_COST))



# ------------------------------------------------------------
# Question 2 - BFS-G
# Regular graph BFS: FIFO queue + do not insert the same state twice
# ------------------------------------------------------------

class BFSGAgent():
    def __init__(self) -> None:
        pass

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        start = int(env.get_initial_state())

        queue = [start]
        head = 0

        parent = {start: (None, None)}
        path_cost = {start: 0.0}

        # I count expanded according to the notebook tests:
        # legal states inserted into the frontier, including start and goal, excluding holes
        expanded = 1

        while head < len(queue):
            state = queue[head]
            head += 1

            if env.is_final_state(state):
                return _reconstruct_path(parent, state), path_cost[state], expanded

            for action in range(_num_actions(env)):
                next_state, cost, terminated = env.succ(state)[action]

                if next_state is None or _is_forbidden_terminal(env, next_state, terminated):
                    continue

                next_state = int(next_state)
                if next_state in parent:
                    continue

                parent[next_state] = (state, action)
                path_cost[next_state] = path_cost[state] + float(cost)
                queue.append(next_state)
                expanded += 1

        return [], np.inf, expanded



# ------------------------------------------------------------
# Question 6 - Greedy Best First Search
# The priority is only h_HAIFA. The state index is used as a deterministic tie breaker
# ------------------------------------------------------------

class GreedyAgent():
    def __init__(self) -> None:
        pass

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        start = int(env.get_initial_state())

        open_queue = heapdict()
        open_queue[start] = (_haifa_heuristic(env, start), start)
        in_open = {start}
        closed = set()

        parent = {start: (None, None)}
        path_cost = {start: 0.0}

        # Same counting convention as in BFS: legal states inserted into OPEN/frontier.
        expanded = 1

        while len(open_queue) > 0:
            state, _ = open_queue.popitem()
            in_open.remove(state)

            if env.is_final_state(state):
                return _reconstruct_path(parent, state), path_cost[state], expanded

            closed.add(state)

            for action in range(_num_actions(env)):
                next_state, cost, terminated = env.succ(state)[action]

                if next_state is None or _is_forbidden_terminal(env, next_state, terminated):
                    continue

                next_state = int(next_state)
                if next_state in closed:
                    continue

                new_cost = path_cost[state] + float(cost)

                if next_state not in path_cost:
                    parent[next_state] = (state, action)
                    path_cost[next_state] = new_cost
                    open_queue[next_state] = (_haifa_heuristic(env, next_state), next_state)
                    in_open.add(next_state)
                    expanded += 1
                elif next_state in in_open and new_cost < path_cost[next_state]:
                    # Greedy chooses nodes by h, but I still update the parent/cost
                    # so the returned path cost is the best one currently known for this state.
                    parent[next_state] = (state, action)
                    path_cost[next_state] = new_cost
                    open_queue[next_state] = (_haifa_heuristic(env, next_state), next_state)

        return [], np.inf, expanded



# ------------------------------------------------------------
# Question 7.1 - A*-epsilon
# First build FOCAL using f=g+h, then choose inside FOCAL according to h_focal
# ------------------------------------------------------------

class AStarEpsilonAgent():
    def __init__(self):
        self.env = None

    def h_focal(self, state: int) -> float:
        if self.env is None:
            return 0.0
        return _haifa_heuristic(self.env, state)

    def search(self, env: HaifaEnv, epsilon: float = None) -> Tuple[List[int], float, int]:
        self.env = env
        if epsilon is None:
            epsilon = 0.0

        start = int(env.get_initial_state())

        open_queue = heapdict()
        start_h = _haifa_heuristic(env, start)
        open_queue[start] = (start_h, start)
        in_open = {start}
        closed = set()

        parent = {start: (None, None)}
        g_score = {start: 0.0}

        # Same counting convention: legal states inserted into OPEN/frontier
        expanded = 1

        while len(open_queue) > 0:
            f_min = open_queue.peekitem()[1][0]
            focal_bound = (1.0 + float(epsilon)) * f_min

            focal = [state for state in in_open if g_score[state] + _haifa_heuristic(env, state) <= focal_bound]

            # From FOCAL I choose the smallest h_focal.
            # If there is still a tie, I take the lower state index to keep the run deterministic
            state = min(focal, key=lambda s: (self.h_focal(s), s))
            del open_queue[state]
            in_open.remove(state)

            if env.is_final_state(state):
                return _reconstruct_path(parent, state), g_score[state], expanded

            closed.add(state)

            for action in range(_num_actions(env)):
                next_state, cost, terminated = env.succ(state)[action]

                if next_state is None or _is_forbidden_terminal(env, next_state, terminated):
                    continue

                next_state = int(next_state)
                tentative_g = g_score[state] + float(cost)

                if next_state not in g_score:
                    parent[next_state] = (state, action)
                    g_score[next_state] = tentative_g
                    f_score = tentative_g + _haifa_heuristic(env, next_state)
                    open_queue[next_state] = (f_score, next_state)
                    in_open.add(next_state)
                    expanded += 1
                    continue

                if tentative_g >= g_score[next_state]:
                    continue

                parent[next_state] = (state, action)
                g_score[next_state] = tentative_g
                f_score = tentative_g + _haifa_heuristic(env, next_state)

                if next_state in closed:
                    # If a better path to a closed node is found, I reopen it.
                    # This keeps the implementation safe even when the heuristic is admissible
                    # but not necessarily consistent.
                    closed.remove(next_state)
                    open_queue[next_state] = (f_score, next_state)
                    in_open.add(next_state)
                elif next_state in in_open:
                    open_queue[next_state] = (f_score, next_state)
                else:
                    open_queue[next_state] = (f_score, next_state)
                    in_open.add(next_state)

        return [], np.inf, expanded



# ------------------------------------------------------------
# Question 7.3 - A*
# Instead of duplicating code, I run A*-epsilon with epsilon=0
# ------------------------------------------------------------

class AStarAgent():
    def __init__(self):
        self.astar_epsilon_agent = AStarEpsilonAgent()

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        return self.astar_epsilon_agent.search(env, epsilon=0.0)
