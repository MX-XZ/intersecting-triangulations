import itertools 
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# Class with auxillary functions for triangulations. 
class Triangulator:

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
    def chromatic_exact(self, n: int): 
        A = self.disjointness_adj(n)
        A_copy = self.disjointness_adj(n)

        colors = n - 3

        triangulations = self.triangulations_trim(n)

        bad = []

        for it in range(len(A)):
            A = np.delete(np.delete(A, it, axis=0), it, axis=1)
            m = gp.Model("ILP")
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

            if m.status == GRB.OPTIMAL:
                bad.append(it)

            A = np.array(A_copy)

        #all_vars =  m.getVars()
        #values =    m.getAttr("X", all_vars)
        #
        #triangulations = self.triangulations_trim(n)

        #for triang, val in zip(itertools.product(triangulations, range(n-3)), values): 
        #    if val == 1:
        #        print(triang)

        print(bad)

# Driver code 
if __name__ == "__main__":
    t = Triangulator()

    n = 7
    t.chromatic_exact(n)
    triang = t.triangulations_trim(n)

    for it in [0, 8, 17, 22, 29, 37, 41]:
        print(triang[it])
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
