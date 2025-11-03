from collections import Counter
import gurobipy as gp
from gurobipy import GRB
import itertools 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import copy 

# Class with auxillary functions for triangulations. 
class Triangulator:

    def __init__(self, n: int):
        self.n = n
    
    # Generates for given n all the triangulations of the n-gon using Hurtado-Noy Hierarchy.
    def triangulations(self, n: int):
        
        if os.path.isfile('triangulations-%s.data' % str(n)):
            with open(os.path.join('triangulations-%s.data' % str(n)), "rb") as data_file:
                out = pickle.load(data_file)
                data_file.close()
                return out

        if n < 3:
            return []
        
        if n == 3:
            return [{(1, 2), (1, 3), (2, 3)}]

        prev = self.triangulations(n - 1)

        triangulations = []

        for triang in prev: 
            neighbors = [j for j in range(n - 1) if (j, n - 1) in triang]

            new_triang = triang.copy()
            new_triang.add((1, n))
            new_triang.add((n-1, n))

            triangulations.append(new_triang)

            for k in range(1, len(neighbors)):
                new_triang = triang.copy()
                new_triang.add((1, n))
                new_triang.add((n-1, n))

                for (it, j) in zip(neighbors[0:k], neighbors[1:(k+1)]):
                    new_triang.remove((it, n-1))
                    new_triang.add((j, n))
            
                triangulations.append(new_triang)
        
        with open(os.path.join('triangulations-%s.data' % str(n)), "wb") as data_file:
            pickle.dump(triangulations, data_file)
            data_file.close()
            
        return triangulations
    
    # Generates for given n all the triangulations of the n-gon using Hurtado-Noy Hierarchy but with outer edges removed.
    def triangulations_trim(self, n: int):

        if os.path.isfile('triangulations-trim-%s.data' % str(n)):
            with open(os.path.join('triangulations-trim-%s.data' % str(n)), "rb") as data_file:
                out = pickle.load(data_file)
                data_file.close()
                return out
    
        outer = {(min(1 + (j % n), 1 + ((j + 1) % n)), max(1 + (j % n), 1 + ((j + 1) % n))) for j in range(n)}
        trimmed = []
        for triang in self.triangulations(n):
            trimmed.append(triang.difference(outer))
        with open(os.path.join('triangulations-trim-%s.data' % str(n)), "wb") as data_file:
            pickle.dump(trimmed, data_file)
            data_file.close()

        return trimmed

    # Generate adjacency matrix for graph with t1 ~ t2 iff they don't share a(-n inner) chord. 
    def disjointness_adj(self, n: int):

        if os.path.isfile('adj-%s.npy' % str(n)):
            with open(os.path.join('adj-%s.npy' % str(n)), "rb") as np_file:
                out = np.load(np_file)
                np_file.close()
                return out

        if n < 3:
            return []
        
        triangulations = self.triangulations_trim(n)

        A = np.zeros((len(triangulations), len(triangulations)), dtype=bool)

        for (it, triang1), (j, triang2) in itertools.product(list(enumerate(triangulations)), repeat=2):
            A[it][j] = (len(triang1.intersection(triang2)) == 0)

        with open(os.path.join('adj-%s.npy' % str(n)), "wb") as np_file:
            np.save(np_file, A)
            np_file.close()
            
        return A
    
    # https://www.sciencedirect.com/science/article/pii/S0012365X19302699 
    # Independence bound using Hoffman-bound in the non-regular case. 
    def independence_bound(self, n: int):
        A = self.disjointness_adj(n)

        min_deg = min(np.matmul(t.disjointness_adj(n), np.ones((len(t.triangulations(n)), 1))))[0]

        eigvals = np.linalg.eigvals(A)
        
        min_val = eigvals.min()
        max_val = eigvals.max()

        a = - len(A) * min_val * max_val 
        b = min_deg * min_deg - min_val * max_val
        return a / b

    # TODO: Program rotation of triangulation. 

    # Rotates triangulations, if labels are ordered clock-wise, num_rot times to the right.
    def rotate(self, n: int, triangulation: set, num_rot=1):
        rotated = set()
        for j, k in triangulation:
            coords = ((j + num_rot - 1) % n + 1, (k + num_rot - 1) % n + 1)
            rotated.add((min(coords), max(coords)))
        return rotated
    
    # Calculates the minimum number of rotations till triangulation intersects 
    # rotational copy in an inner chord.
    def min_rotate(self, n: int, triangulation: set, trimmed=True):
        if n <= 3:
            return 0

        shared = (not trimmed) * n
        
        num = 0
        
        init = triangulation
        rotated = self.rotate(n, triangulation)

        while len(rotated.intersection(init)) == shared:
            num += 1 
            rotated = self.rotate(n, rotated)

        return num
    
    # Average min_rotate of a triangulations of n-gon.
    def average_min_rotate(self, n: int):

        if n <= 3:
            return - 1 
        
        triangulations = t.triangulations_trim(n)
        return sum([t.min_rotate(n, triang) for triang in triangulations]) / len(triangulations)
    
    # Average min_rotate of a triangulations of n-gon.
    def average_min_rotate(self, n: int):

        if n <= 3:
            return - 1 
        
        triangulations = t.triangulations_trim(n)
        return sum([t.min_rotate(n, triang) for triang in triangulations]) / len(triangulations)
    
    def min_rotate_distribution(self, n: int):
        
        if n <= 3:
            return - 1 
        
        triangulations = t.triangulations_trim(n)
        w = Counter([t.min_rotate(n, triang) for triang in triangulations])
        print(w)
        plt.bar(w.keys(), w.values())
        plt.show()        

    # Computes the maximum intersecting family by translating it into independence number problem. 
    def independence_exact(self, n: int):
        A = self.disjointness_adj(n)
        triangulations = self.triangulations_trim(n)

        m = gp.Model("LP")
        y = m.addMVar(len(triangulations), vtype = GRB.BINARY, name = "triangulations")

        m.addConstr(y @ A @ y <= 0)

        m.setObjective(y.sum(), GRB.MAXIMIZE)

        m.optimize()

        all_vars =  m.getVars()
        values =    m.getAttr("X", all_vars)
        
        for triang, val in zip(triangulations, values): 
            if val != 0:
                print(triang)

    # Shows that chromatic number is at least n-2.
    def chromatic_exact(self, n: int, A = None):
        
        if A is None: 
            A = self.disjointness_adj(n)

        colors = n - 3

        m = gp.Model("ILP")
        
        m.Params.OutputFlag = 0 

        y = m.addMVar(len(A) * colors, vtype = GRB.BINARY, name = "triangulation x color")

        for j in range(len(A)):
            m.addConstr(sum(y[colors * j:colors * (j + 1)]) == 1)
            conflicts = np.nonzero(A[j])[0]
            for k in conflicts:
                if k > j:
                    m.addConstr(y[(colors*j):colors*(j+1)] + y[colors*k:colors*(k+1)] <= np.ones(colors))

            #for it in range(colors):
            #   m.addConstr(y[it:((len(A) - 1) * colors + it + 1):colors] @ A @ y[it:((len(A) - 1) * colors + it + 1):colors] == np.zeros(len(A)))

        m.optimize()

        return m.status
    
        #all_vars =  m.getVars()
        #values =    m.getAttr("X", all_vars)
        #
        #triangulations = self.triangulations_trim(n)

        #for triang, val in zip(itertools.product(triangulations, range(n-3)), values): 
        #    if val == 1:
        #        print(triang)
    
    # Method that checks whether chi(Kneser(Triangulations - T)) drops iff T is a star.
    def chromatic_critical_stars(self, n: int):
        A = self.disjointness_adj(n)
        A_copy = self.disjointness_adj(n)

        triangulations = self.triangulations_trim(n)

        # Keep minimal bad sets.
        bad = []
        
        for it, triang in enumerate(triangulations):
            mask = np.arange(A.shape[0]) != it

            A = A[np.ix_(mask, mask)]

            if self.chromatic_exact(n, A) == GRB.OPTIMAL:
                bad.append(triang)

            A = np.array(A_copy)

        assert len(bad) == n

        for triang in bad:
            print(triang)
            triang = [set(tup) for tup in triang]
            assert len(set.intersection(*triang)) == 1

        return True
        
    def chromatic_critical(self, n: int):
        A = self.disjointness_adj(n)
        A_copy = self.disjointness_adj(n)

        triangulations = self.triangulations_trim(n)

        # Keep minimal bad sets.
        bad = set()
        
        candidates = None
        future_candidates = set()
        
        for it in range(len(A)):
            mask = np.arange(A.shape[0]) != it

            A = A[np.ix_(mask, mask)]

            if self.chromatic_exact(n, A) == GRB.INFEASIBLE:
                future_candidates.add((it,))
            else:
                bad.add((it,))

            A = np.array(A_copy)

        while len(future_candidates) > 0:
            
            checked = set()
            
            candidates = future_candidates
            future_candidates = set()

            for cand1, cand2 in itertools.combinations(candidates, 2):
                
                cand1 = set(cand1)

                if len(cand1.symmetric_difference(cand2)) != 2:
                    continue 
                    
                new_cand = cand1.union(cand2)

                tup = tuple(sorted(new_cand))
                if tup in checked:
                    continue 
                checked.add(tup)

                if any([new_cand.issuperset(bad_set) for bad_set in bad]):
                    continue

                mask = [it for it in range(len(A)) if it not in new_cand]

                A = A[np.ix_(mask, mask)]

                if self.chromatic_exact(n, A) == GRB.INFEASIBLE:
                    future_candidates.add(tup)
                else:
                    bad.add(tup)

                A = np.array(A_copy)
        
        print(candidates)
        for cand in candidates:
            print([triangulations[it] for it in cand])

        return candidates
    
    # Constructs vertex-critical subgraph native in Gurobi by computing an Irreducible Inconsistent System (IIS).
    # This will lead to the computation of some vertex-critical subgraph, but not necessarily the one with 
    # minimum number of vertices. 
    # To be specific, the output are the vertices of the color-critical graph.
    # The printed output are the ones excluded.
    def chromatic_critical_GRB(self, n: int):
        
        A = self.disjointness_adj(n)

        colors = n - 3

        m = gp.Model("ILP")
        y = m.addMVar(len(A) * colors, vtype = GRB.BINARY, name = "triangulation x color")

        for j in range(len(A)):
            m.addConstr(sum(y[colors * j:colors * (j + 1)]) == 1, name=str(j))
            conflicts = np.nonzero(A[j])[0]
            for k in conflicts:
                if k > j:
                    constr = m.addConstr(y[(colors*j):colors*(j+1)] + y[colors*k:colors*(k+1)] <= np.ones(colors), name="ignore")
                    # Forces constraint to be included in IIS. 
                    # Thus, the only constraints that can be excluded are the inclusion of vertices.
                    constr.IISConstrForce = 1

        m.computeIIS()

        m.write("model-%s.ilp" % str(n))

        critical_vertices = []
        for constr in m.getConstrs():
            if constr.ConstrName.startswith("ignore"):
                continue
            if constr.IISConstr:
                print(constr.ConstrName)
                critical_vertices.append(int(constr.ConstrName))

        triang = self.triangulations_trim(n)

        uncritical_vertices = set(range(len(triang)))
        uncritical_vertices.difference_update(critical_vertices)
        
        print([triang[it] for it in uncritical_vertices])

        return critical_vertices

    def draw_triangulation(self, n: int, index: int, save=False, show=True):
        edges = self.triangulations(n)[index]

        G = nx.Graph()
        G.add_edges_from(edges)
        
        pos = {}
        for k in range(1, n + 1):
            pos[k] = (np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n))

        plt.figure(figsize=(8, 8))
        plt.axis("equal")
        nx.draw(G, pos=pos, with_labels = True)

        if save:
            plt.savefig('t-%d-%d.png' % (n, index), bbox_inches='tight')

        if show:
            plt.show()
        
        plt.close()

    # Returns same diagonal if we do not flip, otherwise the flipped one. 
    # Assumes triangulation here is not trimmed. 
    # TODO: might not properly work
    def flippable(self, triang: set, edge: tuple):
        (it, j) = edge 

        if it - j in {-2, -1, 1, 2}:
            return edge 
        
        common = {k for k in range(1, n + 1) if (min(it, k), max(it, k)) in triang and (min(j, k), max(j, k)) in triang}
        
        common.difference_update({it, j})

        if len(common) != 2:
            print(common)
            print(triang)
            print(edge)

        assert len(common) == 2

        common = list(common)

        k, l = common[0], common[1]

        if k - l in {-2, 2}:
            return edge 

        q = k + l - it - j

        if q <= n / 2 + 1:
            return edge
        else:
            return (min(k, l), max(k, l))

    # Returns triangulation T' resulting from the edge getting flipped IF T' -> T is a good flip.
    # Assumes triangulation here is not trimmed. 
    # TODO: does not properly work
    def reach(self, triang: frozenset, edge: tuple):
        (it, j) = edge 

        if it - j in {1-n, 2-n, -2, -1, 1, 2, n-2, n-1}:
            return edge 
        
        common = {k for k in range(1, n + 1) if (min(it, k), max(it, k)) in triang and (min(j, k), max(j, k)) in triang}
        
        common.difference_update({it, j})

        assert len(common) == 2

        common = list(common)

        k, l = common[0], common[1]

        if k - l in {1-n, 2-n, -2, -1, 1, 2, n-2, n-1}:
            return edge 

        q = it + j - k - l

        if q <= n / 2 + 1:
            return triang
        else:
            new_triang = set(triang).difference({edge})
            new_triang.add((min(k, l), max(k, l)))
            return frozenset(new_triang)
        
    def color_critical_candidate(self): 
        n = self.n
        triangs = self.triangulations(n)
        cycle = frozenset([(j, j+1) for j in range(1, n)] + [(1, n)])
        
        candidate_set = set()
        for triang in triangs:
            gets_flipped = False
            for edge in triang:
                if edge in cycle:
                    continue 
                if self.flippable(triang, edge) != edge:
                    gets_flipped = True
                    break
            if not gets_flipped:
                candidate_set.add(frozenset(triang))

        to_explore = copy.deepcopy(candidate_set)
        reachable = copy.deepcopy(candidate_set)

        while len(to_explore) > 0 and len(triangs) > len(to_explore):
            triang = to_explore.pop()
            for edge in triang:
                if edge in cycle:
                    continue 
                new_triang = self.reach(triang, edge)
                if new_triang != triang and new_triang not in reachable:
                    reachable.add(new_triang)
                    to_explore.add(new_triang)
        
        print(len(candidate_set))
        print(len(reachable))
        print(len(triangs))

        print(reachable.symmetric_difference([frozenset(triang) for triang in triangs]))

        return len(reachable) == len(triangs)

    def color_critical_candidate_2(self, n):        
        assert n % 3 == 0

        triangs = []
        cycle = frozenset([(j, j+1) for j in range(1, n)] + [(1, n)])

        for m, triang in enumerate(self.triangulations(n)):
            not_flipped = True

            for edge in triang:
                if edge in cycle: 
                    continue 

                (it, j) = edge 

                index = it // 3 


                if it - 3 * index == 1 and j - 3 * index == 4:
                    common = {k for k in range(1, n + 1) if (min(it, k), max(it, k)) in triang and (min(j, k), max(j, k)) in triang}
                    common.difference_update({it, j})
                    assert len(common) == 2
                    common = list(common)
                    k, l = common[0], common[1]

                    if k == 3 * index + 2 and (l > 3 * index + 4 and l < n):
                        not_flipped = False
                        break
                        
            if not_flipped:
                triangs.append(m)
        
        print(len(triangs))
        return triangs

    def color_critical_candidate_2_check(self, n):
        triangs = self.color_critical_candidate_2(n)
        
        A = self.disjointness_adj(n)
        A = A[np.ix_(triangs, triangs)]
        
        A_copy = self.disjointness_adj(n)
        A_copy = A_copy[np.ix_(triangs, triangs)]

        for it in range(len(triangs)):
            mask = np.arange(A.shape[0]) != it

            A = A[np.ix_(mask, mask)]

            if self.chromatic_exact(n, A) == GRB.OPTIMAL:
                return "Not color-critical. :("

            A = np.array(A_copy)

        return "Is color-critical! :)"
    
    # Seems to behave more like ceil((n-2)/3) rather than ceil((n-2)/2).
    # Or ceil((n-4)/2) except when n = 6?
    # Data: n = 5 -> 1, n = 6 -> 2, n = 7 -> 2, n = 8 -> 2, n = 9 -> 3, ...
    def chromatic_exact_3_uniform(self):        
        triangs = self.triangulations_trim(self.n)

        A = self.disjointness_adj(self.n)

        upper_bound = int(np.ceil((self.n - 2) / 2))

        m = gp.Model("ILP")
        x = m.addMVar(len(triangs) * upper_bound, vtype = GRB.BINARY, name = "triangulation x color")
        y = m.addMVar(upper_bound, vtype = GRB.BINARY, name = "color used")
        m.setObjective(y.sum(), GRB.MINIMIZE)

        for j in range(1, upper_bound):
            m.addConstr(y[j - 1] >= y[j])

        for j in range(upper_bound):
            m.addConstr(y[j] * len(triangs) >= x[j::upper_bound].sum())

        for j in range(len(triangs)):
            m.addConstr(sum(x[upper_bound * j:upper_bound * (j + 1)]) == 1)
        
        disj = [set() for _ in range(len(triangs))]
        for (it, j) in np.transpose(np.nonzero(A)):
            disj[it].add(j)
        
        for n1 in range(len(triangs)):
            conflicts = disj[n1]
            for n2, n3 in itertools.combinations(conflicts, 2):
                if n3 in disj[n2]:
                    m.addConstr(x[upper_bound * n1:upper_bound * (n1 + 1)] + x[upper_bound * n2:upper_bound * (n2 + 1)] + x[upper_bound * n3: upper_bound * (n3 + 1)] <= 2 * np.ones(upper_bound))
        
        m.optimize()

        partition = [[] for _ in range(int(m.ObjVal))]

        all_vars =  m.getVars()
        values =    m.getAttr("X", all_vars)
        for (triang, c), val in zip(itertools.product(range(len(triangs)), range(upper_bound)), values):
            if val == 1:
                partition[c].append(triang)

        return partition
    

# Driver code 
if __name__ == "__main__":
    n = 8

    t = Triangulator(n)

    # print(t.chromatic_exact_3_uniform())

    for triang in t.triangulations_trim(n):
        t.draw_triangulation(n, t.triangulations_trim(n).index(triang), True, False)
    #print(t.color_critical_candidate_2_check(n))
    
    # {(1, 11), (5, 7), (1, 4), (11, 12), (2, 9), (4, 7), (5, 12)}
    #for k in range(14):
    #    t.draw_triangulation(n, k, True, False)
    

    #t.chromatic_critical(n)
    #for k in [0, 3, 4, 6, 10, 15, 18, 19, 22, 30, 32, 35, 38, 41, 42, 50, 51, 54, 59, 60, 64, 70, 74, 77, 79, 85, 87, 89, 92, 95, 97, 99, 100, 105, 108, 113, 117, 124, 125, 126, 129, 131]:
    #    t.draw_triangulation(n, k, True, False)
    #print(t.chromatic_critical_GRB(n))
    
    #t.independence_exact(n)

    #t.min_rotate_distribution(n)
    #t.min_rotate_distribution(n)
    # print(t.disjointness_adj(7)) 2 5 8 76 252 840 2959 10588 38064 507585 138362 1872872
    # print(np.all(np.linalg.eigvals(t.disjointness_adj(n)) > 0))
    
    # t.independence_exact(n)

    # print(t.average_min_rotate(n))

    # sum_min = sum([t.min_rotate(n, triang) for triang in t.triangulations(n)])

    # print(sum_min)
    # print(sum_min / len(t.triangulations(n)))

    # for it in range(3, 20):
    #     print(t.independence_bound(it) / len(t.triangulations(it)))
