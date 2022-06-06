import random
import networkx.algorithms.isomorphism as iso
import networkx as nx
import itertools


Q1_OUTPUT_FILE_PATH = "text_files/q1_output.txt"
Q2_OUTPUT_FILE_PATH = "text_files/q2_output.txt"

def debug(msg=""):
    '''if True:
        print(f"-debug- {msg}")
    pass'''


def debug_mat(mat):
    for row in mat:
        debug(row)

def debug_mats(mats):
    for mat in mats:
        debug_mat(mat)
        debug()
    

def mats_are_iso(m1, m2):
    if (len(m1) == len(m2)):
        if (len(m1[0]) == len(m2[0])):
            G1 = mat2DiGraph(m1)
            G2 = mat2DiGraph(m2)
            return nx.is_isomorphic(G1, G2)
    return False
    
def mat2DiGraph(mat):
    G = nx.DiGraph()
    n = len(mat)
    for i in range(n):
        for j in range(n):
            if mat[i][j]:
                G.add_edge(i, j)
    return G


class UndirectedGraph:
    def __init__(self, mat):
        self.n = len(mat)
        self.g = {}
        for node in range(self.n):
            self.g[node] = set()
        for i in range(self.n):
            for j in range(self.n):
                if (mat[i][j]):
                    self.g[i].add(j)
                    self.g[j].add(i)

    def __str__(self):
        result = ""
        for node in self.g:
            result += f"{node} -> {self.g[node]}"
        return result

    def is_connected(self):
        visited = set()
        self.dfs(0, visited)
        return len(visited) == self.n

    def dfs(self, current, visited):
        visited.add(current)
        for neighbor in self.g[current]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

        


class ConnectedSubGraphGenerator:

    @staticmethod
    def get_n():
        return int(input("please enter n:\n"))

    @staticmethod
    def write_result_to_output_file(csgg, output_file_path):
        with open(output_file_path, "w+") as output_file:
            result = csgg.get_result()
            for m in range(len(result)):
                mat = result[m]
                output_file.write(f"#{m+1}:\n")
                for i in range(len(mat)):
                    for j in range(len(mat[0])):
                        if mat[i][j]:    
                            output_file.write(f"{i+1} {j+1}\n")
                #output_file.write(f"\n")

        

    def __init__(self, n):
        self.n = n
        self.all_mats = None
        self.all_mats_without_1s_on_diagonal = None
        self.all_mats_without_1s_on_diagonal_connected = None
        self.all_mats_without_1s_on_diagonal_connected_without_iso = None
        self.result = None


    def generate(self):
        #debug("starting...")
        self.generate_all_mats()
        self.generate_all_mats_without_1s_on_diagonal()
        self.generate_all_mats_without_1s_on_diagonal_connected()
        self.generate_all_mats_without_1s_on_diagonal_connected_without_iso()
        #debug_mats(self.all_mats_without_1s_on_diagonal_connected_without_iso)
        #debug(f"length = {len(self.all_mats_without_1s_on_diagonal_connected_without_iso)}")
        #self.write_output(str(self.all_mats_without_1s_on_diagonal_connected_without_iso))
        self.result = self.all_mats_without_1s_on_diagonal_connected_without_iso
            
        #debug("done!")

    def get_result(self):
        return self.result

    #############################################################################
    # Generating All Mats                                                       #
    #############################################################################
    def generate_all_mats(self):
        k = self.n ** 2
        self.all_mats = [[]]

        # duplicating existing vectors and concatinating 0's and 1's
        for i in range(k):
            self.generte_next_iteration()

        # converting vectors (of size k=n*n) to mats (of size n*n)
        for i in range(2**k):
            self.vec2mat(self.all_mats[i])
                
    def generte_next_iteration(self):
        all_mats = self.all_mats
        size = len(all_mats)
        for i in range(size):
            all_mats.append(all_mats[i].copy())
            all_mats[i].append(0)
            all_mats[i+size].append(1)

    def vec2mat(self, vec):
        tmp = [element for element in vec]
        vec.clear()
        for i in range(self.n):
            vec.append([])
            for j in range(self.n):
                vec[i].append(tmp[self.n * i + j])

    #############################################################################
    # Removing Mats With 1's on the Diagoanl                                    #
    #############################################################################
    def generate_all_mats_without_1s_on_diagonal(self):
        tmp = []
        for mat in self.all_mats:
            if not self.mat_has_1_on_diagonal(mat):
                tmp.append(mat)

        self.all_mats_without_1s_on_diagonal = tmp

    def mat_has_1_on_diagonal(self, mat):
        for i in range(self.n):
            if mat[i][i] == 1:
                return True
        return False

    #############################################################################
    # Generating All Connected Mats                                             #
    #############################################################################
    def generate_all_mats_without_1s_on_diagonal_connected(self):
        self.all_mats_without_1s_on_diagonal_connected = []
        for mat in self.all_mats_without_1s_on_diagonal:
            if self.mat_is_connected(mat):
                self.all_mats_without_1s_on_diagonal_connected.append(mat)

    def mat_is_connected(self, mat):
        g = UndirectedGraph(mat)
        return g.is_connected()

    #############################################################################
    # Removing Isomorphics                                                      #
    #############################################################################
    def generate_all_mats_without_1s_on_diagonal_connected_without_iso(self):
        reps = [] #representatives
        for mat in self.all_mats_without_1s_on_diagonal_connected:
            is_new_rep = True
            for rep in reps:
                if mats_are_iso(mat, rep):
                    is_new_rep = False
                    break
            if is_new_rep:
                reps.append(mat)
        self.all_mats_without_1s_on_diagonal_connected_without_iso = reps


    

    
        

class MotifInstancesInGraphCounter:
    @staticmethod
    def get_mat():
        mx = 0
        edges = []
        e = input("Please enter graph edges and then type 'DONE'\n")
        while e != "DONE":
            e = e.split(" ")
            if len(e) != 2:
                raise Exception("invalid edge syntax")
            e[0] = int(e[0]) - 1
            e[1] = int(e[1]) - 1
            edges.append(e)
            mx = max(mx, e[0], e[1])
            e = input("")
        mat = [[0 for j in range(mx+1)] for i in range(mx+1)]
        for e in edges:
            mat[e[0]][e[1]] = 1
        return mat

    @staticmethod
    def get_n():
        return int(input("Please enter n\n"))
    
    @staticmethod
    def write_result_to_output_file(miigc, output_file_path):
        with open(output_file_path, "w+") as output_file:
            motifs = miigc.all_motifs_size_n
            counters = miigc.all_motifs_size_n_counters
            for m in range(len(motifs)):
                mat = motifs[m]
                output_file.write(f"#{m+1}:\n")
                for i in range(len(mat)):
                    for j in range(len(mat[0])):
                        if mat[i][j]:    
                            output_file.write(f"{i+1} {j+1}\n")
                output_file.write(f"count={counters[m]}\n")

    def __init__(self, n, mat):
        self.n = n
        self.mat = mat
        self.all_sub_graphs_size_n = None
        self.all_motifs_size_n = None
        self.all_motifs_size_n_counters = None
    

    def start(self):
        debug("starting...")
        self.generate_all_sub_graphs_size_n()
        debug(f"self.all_sub_graphs_size_n = {self.all_sub_graphs_size_n}")
        self.generate_all_motifs_size_n()
        debug(f"self.all_motifs_size_n = {self.all_motifs_size_n}")
        self.count_motif_instances()
        debug(f"self.all_motifs_size_n_counters = {self.all_motifs_size_n_counters}")
            
        debug("done!")

    #############################################################################
    # Generating Sub Graphs of Size N                                           #
    #############################################################################
    def generate_all_sub_graphs_size_n(self):
        self.all_sub_graphs_size_n = []
        itr = range(len(self.mat))
        num_elements = self.n
        options = itertools.combinations(itr, num_elements)
        for option in options:
            sub_graph = self.generate_sub_graph(option)
            self.all_sub_graphs_size_n.append(sub_graph)

    
    def generate_sub_graph(self, option):
        sub_graph = [[0 for j in range(len(option))] for i in range(len(option))]
        for i in range(len(sub_graph)):
            for j in range(len(sub_graph)):
                sub_graph[i][j] = self.mat[option[i]][option[j]]
        return sub_graph
        
    #############################################################################
    # Generating All Motifs of Size N                                           #
    #############################################################################
    def generate_all_motifs_size_n(self):
        csgg = ConnectedSubGraphGenerator(self.n)
        csgg.generate()
        self.all_motifs_size_n = csgg.get_result()

    #############################################################################
    # Counting Motif Instances                                                  #
    #############################################################################
    def count_motif_instances(self):
        self.all_motifs_size_n_counters = [0 for _ in self.all_motifs_size_n]
        for sg in self.all_sub_graphs_size_n:
            found = False
            for i in range(len(self.all_motifs_size_n)):
                if mats_are_iso(sg, self.all_motifs_size_n[i]):
                    found = True
                    self.all_motifs_size_n_counters[i] += 1
                    break
            if not found:
                #raise Exception("couldn't find motif")
                pass
            
                    

  
        

def part1():
    n = ConnectedSubGraphGenerator.get_n()
    csgg = ConnectedSubGraphGenerator(n)
    csgg.generate()
    ConnectedSubGraphGenerator.write_result_to_output_file(csgg, Q1_OUTPUT_FILE_PATH)

def part2():
    n = MotifInstancesInGraphCounter.get_n()
    mat = MotifInstancesInGraphCounter.get_mat()
    miigc = MotifInstancesInGraphCounter(n, mat)
    miigc.start()
    MotifInstancesInGraphCounter.write_result_to_output_file(miigc, Q2_OUTPUT_FILE_PATH)

def test():
    n = random.randint(2, 3)
    mat_size = random.randint(2, 6) # TODO
    mat = [[(random.randint(0, 1) if i != j else 0) for j in range(mat_size)] for i in range(mat_size)]

    if n > mat_size:
        n = mat_size
    '''
    mat = [
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ]
    n = 2'''
    
    debug(f"n = {n}")
    debug(f"mat =")
    debug_mat(mat)
    miigc = MotifInstancesInGraphCounter(n, mat)
    miigc.start()
    MotifInstancesInGraphCounter.write_result_to_output_file(miigc, Q2_OUTPUT_FILE_PATH)

def main():
    test_mode = False
    if test_mode:
        for i in range(10):
            input("input...")
            try:
                test()  
            except:
                debug("err")
    else:
        x = input("Please type '1' to start the solution for the first part of the homework.\nPlease type '2' to start the solution for the second part of the homework.\n")
        if (x == "1"):
            part1()
        elif (x == "2"):
            part2()
        else:
            debug("you need to provide a valid input...")
            main()
    
    

main()
