class VacuumWorld:
    def __init__(self):
        self.moves = {
            1: {'R': 2, 'D': 3},
            2: {'L': 1, 'D': 4},
            3: {'U': 1, 'R': 4},
            4: {'U': 2, 'L': 3}
        }

    def get_next_states(self, room):
        return [(next_room, direction) for direction, next_room in self.moves[room].items()]

    def solve(self, start_room, dirt_state, method='dfs'):
        path = []
        visited = set()
        state = [start_room, dirt_state.copy()]
        
        def is_goal(dirt):
            return sum(dirt) == 0

        def dfs(room, dirt):
            if is_goal(dirt):
                return True
            visited.add(room)
            if dirt[room-1] == 1:
                dirt[room-1] = 0
                path.append((room, 'S'))
            for next_room, direction in self.get_next_states(room):
                if next_room not in visited:
                    path.append((room, direction))
                    if dfs(next_room, dirt):
                        return True
            return False

        def bfs():
            from collections import deque
            queue = deque([(start_room, dirt_state.copy(), [])])
            while queue:
                room, dirt, actions = queue.popleft()
                if room not in visited:
                    visited.add(room)
                    if dirt[room-1] == 1:
                        dirt[room-1] = 0
                        actions.append((room, 'S'))
                    if is_goal(dirt):
                        return actions
                    for next_room, direction in self.get_next_states(room):
                        if next_room not in visited:
                            new_actions = actions + [(room, direction)]
                            queue.append((next_room, dirt.copy(), new_actions))
            return []
        if method == 'dfs':
            dfs(start_room, state[1])
            return path
        else:
            return bfs()

def main():
    import sys
    # so isko chalane ke liye 2 arguments chahiye input and output file
    with open(sys.argv[1], 'r') as f: 
        start_room = int(f.readline())
        dirt_state = [int(x) for x in f.readline().strip().split(',')]
        method = f.readline().strip()
    vacuum = VacuumWorld()
    path = vacuum.solve(start_room, dirt_state, method)
    with open(sys.argv[2], 'w') as f:
        for room, action in path:
            f.write(f"{room},{action}\n")

if __name__ == "__main__":
    main()
