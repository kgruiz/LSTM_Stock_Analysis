###############################################################################
# Python code of ARO.                                                         #
# The python code is based on the following paper:                            #
# L. Wang, Q. Cao, Z. Zhang, S. Mirjalili, and W. Zhao, Artificial rabbits    #
# optimization: a new bio-inspired meta-heuristic algorithm for solving       #
# engineering optimization problems, Engineering Applications of Artificial   #
# Intelligence (2022) 105082.                                                 #
###############################################################################
import numpy as np
from torch import randperm
from matplotlib.pyplot import *
from pylab import *


def fun_range(fun_index):
    d = 30
    if fun_index == 1:
        l = [-100]
        u = [100]
    elif fun_index == 2:
        l = [-10]
        u = [10]
    elif fun_index == 3:
        l = [-100]
        u = [100]
    elif fun_index == 4:
        l = [-100]
        u = [100]
    elif fun_index == 5:
        l = [-30]
        u = [30]
    return l, u, d


def ben_functions(x, function_index):
    # Sphere
    if function_index == 1:
        s = sum(np.square(x))
    #Schwefel 2.22
    elif function_index == 2:
        s = np.sum(np.abs(x)) + np.prod(np.abs(x))
    ##Schwefel 1.2
    elif function_index == 3:
        d=len(x)
        s= np.sum([np.sum(x[:j]) ** 2 for j in range(d)])
    #Schwefel 2.21
    elif function_index == 4:
         s=np.max(np.abs(x))
    #Rosenbrock
    elif function_index == 5:
        s = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return s


def space_bound(X, Up, Low):
    dim = len(X)
    S = (X > Up) + (X < Low)
    res = (np.random.rand(dim) * (np.array(Up) - np.array(Low)) + np.array(Low)) * S + X * (~S)
    return res


def ARO(fun_index, max_it, npop):
    lb, ub, dim = fun_range(fun_index)
    if len(lb) == 1:
        lb = lb * dim
        ub = ub * dim
    pop_pos = np.zeros((npop, dim))
    for i in range(dim):
        pop_pos[:, i] = np.random.rand(npop) * (ub[i] - lb[i]) + lb[i]
    pop_fit = np.zeros(npop)
    for i in range(npop):
        pop_fit[i] = ben_functions(pop_pos[i, :], fun_index)
    best_f = float('inf')
    best_x = []
    for i in range(npop):
        if pop_fit[i] <= best_f:
            best_f = pop_fit[i]
            best_x = pop_pos[i, :]
    his_best_fit = np.zeros(max_it)

    for it in range(max_it):

        direct1=np.zeros((npop, dim))
        direct2=np.zeros((npop, dim))
        theta = 2 * (1 - (it+1) / max_it)
        for i in range(npop):
            L = (np.e - np.exp((((it+1) - 1) / max_it) ** 2)) * (np.sin(2 * np.pi * np.random.rand())) # Eq.(3)
            rd = np.floor(np.random.rand() * (dim))
            rand_dim = randperm(dim)
            direct1[i, rand_dim[:int(rd)]] = 1
            c = direct1[i,:]  #Eq.(4)
            R = L * c # Eq.(2)
            A = 2 * np.log(1 / np.random.rand()) * theta #Eq.(15)
            if A>1:
               K=np.r_[0:i,i+1:npop]
               RandInd=(K[np.random.randint(0,npop-1)])
               newPopPos = pop_pos[RandInd, :] + R * (pop_pos[i, :] - pop_pos[RandInd, :])+round(0.5 * (0.05 +np.random.rand())) * np.random.randn() # Eq.(1)
            else:
                ttt=int(np.floor(np.random.rand() * dim))

                direct2[i, ttt] = 1
                gr = direct2[i,:] #Eq.(12)
                H = ((max_it - (it+1) + 1) / max_it) * np.random.randn() # % Eq.(8)
                b = pop_pos[i,:]+H * gr * pop_pos[i,:] # % Eq.(13)
                newPopPos = pop_pos[i,:]+ R* (np.random.rand() * b - pop_pos[i,:]) #Eq.(11)

            newPopPos = space_bound(newPopPos, ub, lb)
            newPopFit = ben_functions(newPopPos, fun_index)
            if newPopFit < pop_fit[i]:
               pop_fit[i] = newPopFit
               pop_pos[i, :] = newPopPos


            if pop_fit[i] < best_f:
               best_f = pop_fit[i]
               best_x = pop_pos[i, :]
        his_best_fit[it] = best_f
    return best_x, best_f, his_best_fit


def main():
    # fun_index = 1: Sphere
    # fun_index = 2: Schwefel 2.22
    # fun_index = 3: Schwefel 1.2
    # fun_index = 4: Schwefel 2.21
    # fun_index = 5: Rosenbrock

    fun_index=1
    max_it = 1000
    npop = 50
    best_x, best_f, his_best_fit = ARO(fun_index, max_it, npop)
    if best_f > 0:
        yscale('log')
        plot(arange(1, max_it + 1), his_best_fit, 'r')
    else:
        plot(arange(1, max_it + 1), his_best_fit, 'r')
    xlim([0, max_it + 1])
    xlabel('Iterations')
    ylabel('Fitness')
    title('F' + str(fun_index))
    show()
    print('The best solution is: ', best_x)
    print('The fitness is: ', best_f)



if __name__ == '__main__':
    main()

