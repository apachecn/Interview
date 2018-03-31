import numpy as np
import functions

def get_solutions():
    paths = ['result_cnn.npy', 'result_svm.npy', 'result_knn.npy', 'result_rf.npy']
    solutions = [None for _ in paths]
    for i, path in enumerate(paths):
        solutions[i] = np.load(path)
    return solutions

def get_result(solutions):
    def get(sols):
        if(sols[1] == sols[2] and sols[1] == sols[3]): return sols[1]
        return sols[0]

    length = len(solutions[0])
    result = [None for _ in range(length)]
    for i in range(length):
        result[i] = get([solutions[j][i] for j in range(4)])
    return result

solutions = get_solutions()
result = get_result(solutions)
functions.save_result(result)