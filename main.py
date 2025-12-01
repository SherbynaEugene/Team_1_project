'''An example of using the ant algorithm'''
import random



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
    n_vertices = len(graph)
    half = n_vertices / 2
    if n_vertices >= 3:
        for _ , connections in graph.items():
            vertex_degree = len(connections)
            if vertex_degree < half:
                return






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

    if best_path:
        print("Найкращий знайдений Гамільтоновий цикл:", best_path)
        print("Довжина:", best_length)
    else:
        print("Гамільтоновий цикл не знайдено")


ant_algorithm(3, 20, file_reader("graph.txt"))
