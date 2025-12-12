'''An example of using the ant algorithm with weighted edges'''
import random
import pygame
import sys
import argparse

parser = argparse.ArgumentParser(description="Ant Colony Algorithm Visualization")

parser.add_argument('file', type=str, help='Шлях до файлу з графом')
parser.add_argument('ants', type=int, help='Кількість мурах')
parser.add_argument('iterations', type=int, help='Кількість ітерацій')

args= parser.parse_args()

def file_reader(filename: str) -> tuple[dict, dict]:
    '''
    Reading a given file with the weighted undirected graph
    :args: filename of file with weighted undirected graph
    :return: tuple of (graph dictionary, weights dictionary)
        graph: keys are vertices, values are lists of connected vertices
        weights: keys are edges (tuples), values are weights
    '''
    with open(filename, 'r' , encoding='utf-8') as f:
        vertices, edges = [int(item) for item in f.readline().split()]
        lines = f.readlines()
    
    graph = {}
    weights = {}
    
    for line in lines:
        parts = [int(item) for item in line.split()]
        a_vertice, b_vertice, weight = parts[0], parts[1], parts[2]
    
        if a_vertice not in graph:
            graph[a_vertice] = [b_vertice]
        else:
            graph[a_vertice].append(b_vertice)
        
        if b_vertice not in graph:
            graph[b_vertice] = [a_vertice]
        else:
            graph[b_vertice].append(a_vertice)

        weights[(a_vertice, b_vertice)] = weight
        weights[(b_vertice, a_vertice)] = weight
    
    return graph, weights


def ant_algorithm(num_ants: int, iterations: int, graph: dict, weight: dict):
    '''
    Algorithm made for searching the shortest way for ant to go
    which is also a hamiltonian cycle.
    '''
    dirak_theorem = True
    n_vertices = len(graph)
    half = n_vertices / 2
    if n_vertices >= 3:
        for _ , connections in graph.items():
            vertex_degree = len(connections)
            if vertex_degree < half:
                dirak_theorem = False

    pheromone = {edge: 0.1 for edge in weight}
    edge_visits = {edge: 0 for edge in pheromone}
    alpha = 1
    beta = 2
    evaporation = 0.1

    def path_weight(path: list) -> int:
        '''
        Function made for calculating the total weight of ant's path
        :arg: path - list of vertices visited by an ant
        :return: total weight of all edges visited by an ant
        '''
        total = 0
        for j in range(len(path)-1):
            total += weight[(path[j], path[j+1])]
        return total

    def choose_next_vertice(current: int, visited: list[int]) -> None | int:
        '''
        Function made for calculating and making ant's choose
        based on factors:
        1. weight of the edge (less weight = better)
        2. pheromone amount of the edge (more pheromone = better)
        :arg1: current = vertice where ant is
        :arg2: list of vertices visited by ant (first it is just a start vertice)
        :return: None if ant has no way to go
                number of next vertice based on calculated factors + random
        '''
        neighbors = [n for n in graph[current] if n not in visited]
        if not neighbors:
            return None
        probabilities = []
        for n in neighbors:
            tau = pheromone[(current, n)] ** alpha
            eta = (1 / weight[(current, n)]) ** beta
            probabilities.append(tau * eta)
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return random.choices(neighbors, weights=probabilities)[0]

    def ant_run(start: int) -> None | list[int]:
        '''
        Function made for creating a path of ant
        :arg1: random start vertice
        :return: None if there is NO way back to start vertice
                path if there IS way back to start vertice
        '''
        path = [start]
        visited = set(path)

        while len(path) < len(graph):
            next_city = choose_next_vertice(path[-1], visited)
            if next_city is None:
                return None
            path.append(next_city)
            visited.add(next_city)

        if start in graph[path[-1]]:
            path.append(start)
            return path
        else:
            return None

    best_path = None
    best_length = float('inf')

    for it in range(iterations):
        print(graph)
        print(f"Ітерація {it+1}")
        all_paths = []
        paths_for_visual = []
        found_cycles = 0

        for ant in range(num_ants):
            start = random.choice(list(graph.keys()))
            path = ant_run(start)
            if path is None:
                print(f"Мурашка {ant+1}: не знайшла Гамільтоновий цикл")
                paths_for_visual.append([start])
                continue
            found_cycles += 1
            length = path_weight(path)
            all_paths.append((path, length))
            paths_for_visual.append(path)
            print(f"Мурашка {ant+1}: {path} довжина={length}")

            if length < best_length:
                best_length = length
                best_path = path
        
        for path in paths_for_visual:
            if len(path) == 1:
                all_paths.append((path, float('inf')))

        for edge in pheromone:
            pheromone[edge] *= (1 - evaporation)
        
        for path, length in all_paths:
            if length != float('inf'):
                deposit = 1.0 / length
                for i in range(len(path)-1):
                    a, b = path[i], path[i+1]
                    pheromone[(a, b)] += deposit
                    pheromone[(b, a)] += deposit
                    edge_visits[(a, b)] += 1
                    edge_visits[(b, a)] += 1

        print("Феромони:", {k: round(v,2) for k,v in pheromone.items()})
        print("-"*50)
        yield {"iteration": it+1, "best_path": best_path, "pheromone": pheromone.copy(), \
"edge_visits": edge_visits.copy(), "current_paths": paths_for_visual, "found_cycles": found_cycles}

    if best_path:
        print("Найкращий знайдений Гамільтоновий цикл:", best_path)
        print("Довжина:", best_length)
    else:
        print("Гамільтоновий цикл не знайдено")

def connectivity(graph: dict)->bool:
    """
    Checks connectivity in the graph.

    :param graph: a dictionary, representing a graph
    returns: True if graph is connected, False if else
    """
    visited=set()
    def dfs(v):
        visited.add(v)
        for a in graph[v]:
            if a not in visited:
                dfs(a)
    for v in graph:
        if v not in visited:
            dfs(v)
    if len(graph)==len(visited):
        return True
    return False

if not connectivity(graph):
    print("Граф не зв'язний. Гамільтоновий цикл не існує.")
    sys.exit()
#VISUALКА
WIDTH, HEIGHT = 800, 800
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant Algorithm Visualization")
clock= pygame.time.Clock()
font_num=pygame.font.Font(None, 40)
font_text=pygame.font.Font(None, 40)
font_weight=pygame.font.Font(None, 25)

def node_position(graph_, root_node=1):
    n_vertices = len(graph_)

    if n_vertices <= 4:
        level_spacing = 180
        sibling_spacing = 200
    elif n_vertices <= 6:
        level_spacing = 150
        sibling_spacing = 160
    elif n_vertices <= 8:
        level_spacing = 120
        sibling_spacing = 140
    else:
        level_spacing = 100
        sibling_spacing = 120

    start_x = 100
    start_y = 100
    node_positions = {}
    visited = set()

    def dfs(node, depth, index):
        x = start_x + index * sibling_spacing
        y = start_y + depth * level_spacing
        node_positions[node] = (x, y)
        visited.add(node)

        neighbors = [n for n in graph_[node] if n not in visited]
        for ind, neighbor in enumerate(neighbors):
            dfs(neighbor, depth+1, ind)

    dfs(root_node, 0, 0)
    return node_positions

def draw_nodes(surface_node, positions_nodes:dict):
    for node, pos in positions_nodes.items():
        pygame.draw.circle(surface_node, 'black', pos, 30)
        pygame.draw.circle(surface_node, 'white', pos, 27)
        node_num = font_num.render(str(node), True, 'black')
        text_width = node_num.get_width()
        text_height = node_num.get_height()

        surface_node.blit(node_num, (pos[0] - text_width // 2, pos[1] - text_height // 2))

def draw_edges(surface_node, graph_draw, positions_nodes, pheromone, weights, **kwargs):
    if not pheromone:
        return

    drawn = set()

    for u, neighbors in graph_draw.items():
        for v in neighbors:
            key = tuple(sorted((u, v)))
            if key in drawn:
                continue
            drawn.add(key)
            pher_u_v = pheromone.get((u, v), 0.1)
            pher_v_u = pheromone.get((v, u), 0.1)
            pher_val = max(pher_u_v, pher_v_u)
            if pher_val <= 0.11:    
                r, g, b = 0, 0, 0
                thickness = 2
            else:
                t = min(1.0, (pher_val - 0.11) / 0.39)
            
                r = 0
                g = int(255 * t)
                b = 0
                
                thickness = int(2 + 6 * t)

            pygame.draw.line(
                surface_node, (r, g, b),
                positions_nodes[u], positions_nodes[v], thickness
            )
    
            edge_weight = weights.get((u, v), weights.get((v, u), 0))
            mid_x = (positions_nodes[u][0] + positions_nodes[v][0]) // 2
            mid_y = (positions_nodes[u][1] + positions_nodes[v][1]) // 2
        
            dx = positions_nodes[v][0] - positions_nodes[u][0]
            dy = positions_nodes[v][1] - positions_nodes[u][1]
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                offset_x = -dy / length * 15
                offset_y = dx / length * 15
            else:
                offset_x, offset_y = 0, 0
            
            weight_text = font_weight.render(str(edge_weight), True, 'blue')
            text_x = mid_x + offset_x - weight_text.get_width() // 2
            text_y = mid_y + offset_y - weight_text.get_height() // 2
            surface_node.blit(weight_text, (text_x, text_y))

graph, weights = file_reader(args.file)
num_ants = args.ants
iterations = args.iterations
positions = node_position(graph)

surface = pygame.Surface((WIDTH, HEIGHT))
surface.fill('white')

algo = ant_algorithm(num_ants, iterations, graph, weights)
best_path = None
pheromone = None

for i, state in enumerate(algo):
    best_path = state["best_path"]
    pheromone = state["pheromone"]
    iteration_num = state["iteration"]
    edge_visits = state["edge_visits"]
    current_paths = state["current_paths"]
    found_cycles = state["found_cycles"]
    if found_cycles == 0:
        animation_frames = int(120)
    else:
        animation_frames = 300

    for frame in range(animation_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        surface.fill('white')

        iter_text = font_text.render(f"Ітерація {i + 1}", True, 'black')
        surface.blit(iter_text, (10, 10))

        draw_edges(surface, graph, positions,
               pheromone=pheromone,
               weights=weights,
               edge_visits=edge_visits,
               iteration=iteration_num,
               total_iterations=iterations)

        for path in current_paths:
            if len(path) > 1:
                progress = frame / animation_frames

                total_edges = len(path) - 1
                edge_progress = progress * total_edges

                a_index = int(edge_progress)
                if a_index >= total_edges:
                    a_index = total_edges - 1
                b_index = a_index + 1

                t = edge_progress - a_index

                a = positions[path[a_index]]
                b = positions[path[b_index]]

                x1, y1 = a
                x2, y2 = b
                x = x1 + (x2 - x1) * t
                y = y1 + (y2 - y1) * t

                pygame.draw.circle(surface, (255, 0, 0), (int(x), int(y)), 8)
            else:
                pos = positions[path[0]]
                pygame.draw.circle(surface, (255, 0, 0), pos, 8)

        draw_nodes(surface, positions)
        screen.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(60)

if best_path is not None:
    ants = []

    for _ in range(5):
        ants.append({
            "path": best_path,
            "edge": 0,
            "t": random.random(),
            "trail": []
        })

    def move_along_edge(a, b, t):
        x1, y1 = a
        x2, y2 = b
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        return(x, y)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        surface.fill("white")

        if pheromone:
            drawn = set()
            
            for node, neighbors in graph.items():
                for neighbor in neighbors:
                    key = tuple(sorted((node, neighbor)))
                    if key in drawn:
                        continue
                    drawn.add(key)

                    pher_val = max(pheromone.get((node, neighbor), 0.1),
                                   pheromone.get((neighbor, node), 0.1))

                    if pher_val <= 0.11:
                        r, g, b = 0, 0, 0
                        thickness = 2
                    else:
                        t = min(1.0, (pher_val - 0.11) / 0.39)
                        r = 0
                        g = int(255 * t)
                        b = 0
                        thickness = int(2 + 6 * t)

                    pygame.draw.line(surface, (r, g, b),
                                   positions[node], positions[neighbor], thickness)
                    edge_weight = weights.get((node, neighbor), weights.get((neighbor, node), 0))
                    mid_x = (positions[node][0] + positions[neighbor][0]) // 2
                    mid_y = (positions[node][1] + positions[neighbor][1]) // 2
        
                    dx = positions[neighbor][0] - positions[node][0]
                    dy = positions[neighbor][1] - positions[node][1]
                    length = (dx**2 + dy**2)**0.5
                    if length > 0:
                        offset_x = -dy / length * 15
                        offset_y = dx / length * 15
                    else:
                        offset_x, offset_y = 0, 0
                    
                    weight_text = font_weight.render(str(edge_weight), True, 'blue')
                    text_x = mid_x + offset_x - weight_text.get_width() // 2
                    text_y = mid_y + offset_y - weight_text.get_height() // 2
                    surface.blit(weight_text, (text_x, text_y))

        for i in range(len(best_path)-1):
            a = best_path[i]
            b = best_path[i+1]
            pygame.draw.line(surface, (255, 215, 0),
                             positions[a], positions[b], 7)

        for ant in ants:
            a_index = ant["edge"]
            b_index = a_index + 1
            if b_index >= len(ant["path"]):
                b_index = 0

            a = positions[ant["path"][a_index]]
            b = positions[ant["path"][b_index]]
            ant_pos = move_along_edge(a, b, ant["t"])
            pygame.draw.circle(surface, (255, 0, 0), (int(ant_pos[0]), int(ant_pos[1])), 10)

            ant["t"] += 0.02
            if ant["t"] >= 1:
                ant["t"] -= 1
                ant["edge"] = b_index

        draw_nodes(surface, positions)
        screen.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(60)

pygame.quit()
