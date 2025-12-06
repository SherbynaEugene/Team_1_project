'''An example of using the ant algorithm'''
import random
import pygame
import sys

def file_reader(filename: str) -> dict:
    '''
    Reading a given file with the undirected graph
    :args: filename of file with undirected graph
    :return: dictionary where keys are vertices
        and values are lists of all other vertices conected to the key one
    '''
    with open(filename, 'r' , encoding='utf-8') as f:
        vertices, edges =[int(item) for item in f.readline().split()]#<- перетворює '5 6\n' в [5, 6]
        lines = f.readlines()
    graph = {}
    for line in lines:
        a_vertice, b_vertice = [int(item) for item in line.split()]  # <-- Те саме
        if a_vertice not in graph:
            graph[a_vertice] = [b_vertice]
        else:
            graph[a_vertice].append(b_vertice)
        if b_vertice not in graph:
            graph[b_vertice] = [a_vertice]
        else:
            graph[b_vertice].append(a_vertice)
    return graph




def ant_algorithm(num_ants: int, iterations: int, graph: dict):
    '''
    Algorithm made for searching the shortes way for ant to go
    which is also a hamiltonian cycle.
    '''
    # CHECKing if there is any hamiltonian cycle in graph!!
    # Based on "Theorem of Dirak":
    dirak_theorem = True
    n_vertices = len(graph)
    half = n_vertices / 2
    if n_vertices >= 3:
        for _ , connections in graph.items():
            vertex_degree = len(connections)
            if vertex_degree < half:
                dirak_theorem = False






    # Start weight for all edges in graph
    weight = {}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            weight[(node, neighbor)] = 1

    # Start pheromone amount for all edges in graph
    pheromone = {edge: 0.1 for edge in weight}

    alpha = 1       # CONSTant for edge pheromone impact on ant's choose
    beta = 2        # CONSTant for edge weight impact on ant's choose
    evaporation = 0.5  # CONSTant for pheramone evaporation

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
        1. weight of the edge
        2. pheromone amount of the edge
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
            if next_city is None:  # немає куди йти
                return None
            path.append(next_city)
            visited.add(next_city)

        if start in graph[path[-1]]:
            path.append(start)  # hamiltonian cycle IS found
            return path
        else:
            return None  # hamiltonian cycle IS NOT found

    best_path = None
    best_length = float('inf')

    for it in range(iterations):
        print(graph)
        print(f"Ітерація {it+1}")
        all_paths = []
        for ant in range(num_ants):
            start = random.choice(list(graph.keys()))
            path = ant_run(start)
            if path is None:
                print(f"Мурашка {ant+1}: не знайшла Гамільтоновий цикл")
                continue
            length = path_weight(path)
            all_paths.append((path, length))
            print(f"Мурашка {ant+1}: {path} довжина={length}")

            if length < best_length:
                best_length = length
                best_path = path

        # block for evaporation of pheromone
        for edge in pheromone:
            pheromone[edge] *= (1 - evaporation)

        # block for adding pheromone to ant path
        for path, length in all_paths:
            deposit = 1 / length
            for i in range(len(path)-1):
                a, b = path[i], path[i+1]
                pheromone[(a, b)] += deposit
                pheromone[(b, a)] += deposit  # undirected graph

        print("Феромони:", {k: round(v,2) for k,v in pheromone.items()})
        print("-"*50)
        yield {"iteration": it+1, "best_path": best_path, "pheromone": pheromone.copy()}

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


ant_algorithm(3, 20, file_reader("graph.txt"))

#VISUALKA
WIDTH, HEIGHT = 800, 800
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant Algorithm Visualization")
clock= pygame.time.Clock()
font_num=pygame.font.Font(None, 40)
font_text=pygame.font.Font(None, 40)

def node_position(graph_, root_node=1):
    level_spacing=100
    sibling_spacing=120
    start_x=100
    start_y=100
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
        node_num= font_num.render(str(node), True, 'black')
        surface_node.blit(node_num, (pos[0]-8, pos[1]-12))

def draw_edges(graph_draw, positions_nodes):
    for node, neighbors in graph_draw.items():
        for neighbor in neighbors:
            pygame.draw.line(surface, ('black'), \
                             positions_nodes[node], positions_nodes[neighbor], 3)


graph = file_reader("graph.txt")
positions = node_position(graph)

surface= pygame.Surface((WIDTH, HEIGHT))
surface.fill('white')

algo = ant_algorithm(3, 20, graph)

for i, state in enumerate(algo):
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.blit(surface,(0,0))
    iter_text=font_text.render(f"Ітерація {i}", True,'black', 'white')
    surface.blit(iter_text,(400,0))
    draw_edges(graph, positions)
    draw_nodes(surface, positions)
    pygame.display.update()
    clock.tick(1)
