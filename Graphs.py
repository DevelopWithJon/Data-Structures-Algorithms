def depthFirstIterative(graph, source):
    
    stack = [source]
    
    while stack:
        curr = stack.pop()
        print(curr)
        
        for neighbor in graph[curr]:
            stack.append(neighbor)



def depthFirstRec(graph, source):
    
    print(source)
    for neighbor in graph[source]:
        depthFirstRec(graph, neighbor)
        
def breadthFirst(graph, source):
    
    queue = [source]
    while queue:
        curr = queue.pop(0)
        print(curr)
        for neighbor in graph[curr]:
            queue.append(neighbor)


graph = {
    "a": ['b', 'c'],
    "b": ["d"],
    "c": ["e"],
    "d": ['f'],
    "e": [],
    "f": []
    }

#print(depthFirstIterative(graph, "a"))
#depthFirstRec(graph, "a")
# breadthFirst(graph, "a")

##  Complexity
# n = # nodes
# e = # edges
# time: O(e)
# space: O(n)

#OR
# n = # nodes
# n^2 = # edges


def hasPathR(graph, src, dst):
    
    if src == dst: return True
    
    for neighbor in graph[src]:
        if hasPathR(graph, neighbor, dst) is True:
            return True
    return False

def hasPathBFS(graph, src, dst, visited):
    
    if src in visited: return False
    
    visited.add(src)
    queue = [src]
    
    while queue:
        current = queue.pop(0)
        if current == dst: return True
        for neighbor in graph[current]:
            queue.append(neighbor)
    return False
        
        

def undirectedPath(edges, nodeA, nodeB):
    
    graph = buildGraph(edges)
    return hasPathBFS(graph, nodeA, nodeB, set())

def hasPath(graph, src, dst, visited):
    if src==dst: return True
    
    if src in visited: return False
    
    visited.add(src)
    
    for neighbor in graph[src]:
        if hasPath(graph, neighbor, dst, visited) is True:
            return True
        
    return False
    
def buildGraph(edges):
    
    graph = {}
    
    for edge in edges:
        a,b = edge
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a] += [b]
        graph[b] += [a]
        
    return graph

edges = [
  ['i', 'j'],
  ['k', 'i'],
  ['m', 'k'],
  ['k', 'l'],
  ['o', 'n']
]
def connectedcompnentCount(graph):
    
    visited = set()
    count = 0
    for node in graph:
        if explore(graph, node, visited) is True:
            count+=1
    return count
        
def explore(graph, current, visited):
    if current in visited: return False
    
    stack = [current]
    
    while len(stack):
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            
        for neighbor in graph[current]:
            if neighbor not in visited:
                stack.append(neighbor)      
    
    return True
        
graph = {
  0: [4,7],
  1: [],
  2: [],
  3: [6],
  4: [0],
  6: [3],
  7: [0],
  8: []
}

def longestPath(graph):
    
    visited = set()
    largest = 0
    
    for node in graph:
        currentPath = 0
        
        if node not in visited:
            stack = [node]
            
        while len(stack):
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                currentPath+=1
                
            for neighbor in graph[current]:
                if neighbor not in visited:
                    stack.append(neighbor) 
                    
        largest = max(largest, currentPath)
    return largest
                

def shortestPath(edges, src, dst):
    
    graph = buildPath(edges)
    
    
    # BFS
    
    visited = set(src)
    queue = [(src, 0)]
    
    while len(queue):
        
        current, dist = queue.pop(0)
        if current == dst: return dist
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist+1))
    
    return -1
            
        
def buildPath(edges):
    
    graph = {}
    
    for edge in edges:
        a,b = edge
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a]+=b
        graph[b]+=a
    
    return graph

edges = [
  ['w', 'x'],
  ['x', 'y'],
  ['z', 'y'],
  ['z', 'v'],
  ['w', 'v']
]

def islandCount(graph):
    
    visited = set()
    
    count = 0
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            if exploreI(graph,i,j,visited) is True:
                count+=1
    return count

def exploreI(graph, row, col, visited):
    
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    if (0 <= row and row < len(graph)) and (0 <= col and col < len(graph[0])):
        
        if graph[row][col] == "W": return False
        
        pos = str(row)+","+str(col)
        if pos in visited: return False
        visited.add(pos)

        
        for direction in directions:
            exploreI(graph, row+direction[0], col+direction[1], visited)
        
        return True
        
        
        
    else: return False
    
grid = [
  ['L', 'W'],
  ['L', 'L'],
  ['L', 'W'],
]


def minIsland(graph):
    
    visited = set()
    minI = len(graph)*len(graph[0])
    

    for i in range(len(graph)):
        for j in range(len(graph[0])):
            iSize = exploreMI(graph, i, j, visited)
            if iSize>0:
                minI = min(minI, iSize)
    return minI
            
def exploreMI(graph, row, col, visited):
    
    
    directions = [(0,1), (0,-1), (1,0), (1,0)]
    
    if (0 <= row < len(graph)) and (0 <= col < len(graph[0])):
        
        if graph[row][col] == "W": 
            return 0
        
        pos = str(row) + "," + str(col)
        if pos in visited:
            return 0
        
        visited.add(pos)
        
        iSize=1
        
        for direction in directions:
            iSize+=exploreMI(graph, row+direction[0], col+direction[1], visited)

        return iSize
            
    else:
        return 0
                                   
    

grid = [
  ['W', 'W'],
  ['L', 'L'],
  ['W', 'W'],
  ['W', 'L']
]

#print(minIsland(grid))
    
"""Is valid Tree no cycles one components"""
# # of edges is equal to n-1 nodes use a parent dict pass if its a direct child(undirected) but flag is a parent exists as a further decendents chil
class Solution:
    def validTree(self, n: int, edges) -> bool:
        
        if len(edges) != n - 1: return False
    
        adj_list = [[] for _ in range(n)]
        for A, B in edges:
            adj_list[A].append(B)
            adj_list[B].append(A)

        parent = {0: -1}
        queue = [0]

        while queue:
            node = queue.popleft()
            for neighbour in adj_list[node]:
                if neighbour == parent[node]:
                    continue
                if neighbour in parent:
                    return False
                parent[neighbour] = node
                queue.append(neighbour)

        return len(parent) == n
    
    
INF = float('inf')
m = [[0,4,1,INF],
     [INF,0,6,INF],
     [4,1,0,2],
     [INF,INF,INF, 0]]
    
def floydWarshall(m):
    N = len(m)
    dp = [[] for _ in range(N)]
    nextM = [[None] for _ in range(N)] 
    
    
    for i in range(N):
        for j in range(N):
            if m[i][j] != INF:
                pass
    

"""Shortest Path FloydWarshall"""
EDGES = {
    (0,1): 1,
    (1,0): 1,
    (0,2): 1.5,
    (2,0): 1.5,
    (2,3): 1.5,
    (3,2): 1.5,
    (1,3): 0.5,
    (3,1): 0.5,
    (0,3): 2,
    (3,0): 2,
    (1,4): 2.5,
    (4,1): 2.5,
    (5,3): -4.5,
    (4,5): 2,
    (5,4): 2
         }
NODES = [0,1,2,3,4,5]

def fWarshall(nodes, edges):
    
    d = {(u,v): float('inf') if u !=v else 0 for u in nodes for v in nodes}
    for (u,v), w_uv in edges.items():
        d[(u,v)] = w_uv
        
    for k in nodes:
        for u in nodes:
            for v in nodes:
                d[(u,v)] = min(d[(u,v)], d[(u,k)] + d[(k,v)])
    if any(d[(u,u)] < 0 for u in nodes):
        print("Graph has negative-weight cycle")
        
    return [node for node in d]
            
"""Shortest Path Dijkstra's no negative-weights"""
EDGES = {
    (0,1): 1,
    (1,0): 1,
    (0,2): 1.5,
    (2,0): 1.5,
    (2,3): 1.5,
    (3,2): 1.5,
    (1,3): 0.5,
    (3,1): 0.5,
    (0,3): 2,
    (3,0): 2,
    (1,4): 2.5,
    (4,1): 2.5,
    (5,3): 1,
    (4,5): 2,
    (5,4): 2
         }
NODES = [0,1,2,3,4,5]

def dijkstra(nodes, edges, s=0):
    
    path_length = {v: float('inf') for v in nodes}
    path_length[s] = 0
    
    adjecent_nodes = {v: {} for v in nodes}
    for (u,v), w_uv in edges.items():
        adjecent_nodes[u][v] = w_uv
        adjecent_nodes[v][u] = w_uv
    
    temp_list = [v for v in nodes]
    while temp_list:
        upper_bound = {v: path_length[v] for v in temp_list}
        u = min(upper_bound, key=upper_bound.get)
        
        temp_list.remove(u)
        
        for v, w_uv in adjecent_nodes[u].items():
            path_length[v] = min(path_length[v], path_length[u] + w_uv)
    return path_length

"""Shortest Path Bellman-Ford's no negative-weights cycles"""
EDGES = {
    (0,1): 1,
    (1,0): 1,
    (0,2): 1.5,
    (2,0): 1.5,
    (2,3): 1.5,
    (3,2): 1.5,
    (1,3): 0.5,
    (3,1): 0.5,
    (0,3): 2,
    (3,0): 2,
    (1,4): 2.5,
    (4,1): 2.5,
    (5,3): -4.5,
    (4,5): 2,
    (5,4): 2
         }
NODES = [0,1,2,3,4,5]

"""Topological Sort"""
from collections import defaultdict


def makeAdList(edges):
    ver = set()
    adList = defaultdict(list)
    for dest, src in edges:
        ver.add(src)
        ver.add(dest)
        adList[src].append(dest)
    return len(ver), adList
    
def topSortUtil(v, visited, stack, adList):
    visited[v] = True
    
    for i in adList: 
        if visited[i] == False:
            topSortUtil(i, visited, stack, adList)
    stack.insert(0,v)
                    
    
def topSort(graph):
    N, adList = makeAdList(graph)
    visited = [False]*N
    stack=[]
    
    for i in range(N):
        if visited[i] == False:
            topSortUtil(i, visited, stack, adList)
    return stack
print(topSort([[5,2],[5,0],[4,0],[4,1],[2,3],[3,1]]))

data = [[0,10,15,20], [10,0,35,25], [15,35,0,30], [20,25,30,0]]
s=0

def travelingSalesman(data, s):
    v = len(data)
    current_cost=0
    vertex = []
    for i in range(v):
        if i != s:
            vertex.append(i)
    min_path = float('inf')
    while True:
        current_cost = 0
        k = s
        for i in range(len(vertex)):
            current_cost += data[k][vertex[i]]
            k=vertex[i]
        current_cost+=data[k][s]
        min_path = min(min_path, current_cost)
        
        if not next_perm(vertex):
            break
    return min_path

def next_perm(l):
    N = len(l)
    m = N-2
    
    while m>=0 and l[m] > l[m+1]:
        m-=1
        
    if i == -1:
        return False
    j=m+1
    while j<N and l[j] > l[m]:
        j+=1
    j-=1
    
    l[m], l[j] = l[j], l[m]
    left = m+1
    right

import heapq
def dijkstrax(graph, src, dest):
    INF = float('inf')
    node_data = {node: {"cost": INF, "pred":[]} for node in graph}
    N = len(graph)
    
    node_data[src]['cost'] = 0
    visited = []
    temp = src

    for _ in range(N-1):
        if temp not in visited:
            visited.append(temp)
            pq = []
            for nei in graph[temp]:
                if nei not in visited:
                    cost = node_data[temp]['cost'] + graph[temp][nei]
                    if cost < node_data[nei]['cost']:
                        node_data[nei]['cost'] = cost
                        node_data[nei]['pred'] = node_data[temp]['pred'] + list(temp)
                    heapq.heappush(pq, (node_data[nei]['cost'],nei))
        heapq.heapify(pq)
        temp = pq[0][1]
    print(f"Shortest Distrance to node {dest}: {node_data[dest]['cost']}")
    print(f"Shortest Path to node {dest}: {node_data[dest]['pred'] + list(dest)}")

graph = {
    "A": {"B":2, "C":4},
    "B": {"A":2, "C":3, "D": 8},
    "C": {"A":4, "B":3, "E":5, "D":2},
    "D": {"B":8, "C":2, "E":11, "F":22},
    "E": {"C":5, "D":11, "F":1},
    "F": {"D":22, "E":1},
}
source = "A"
dest = "F"
print(dijkstrax(graph, source, dest))