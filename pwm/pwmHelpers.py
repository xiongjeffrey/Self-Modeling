import numpy as np
def precode(pos, width = 5, height = 5):
    map = np.zeros((height, width))
    map[pos[0]][pos[1]] = 1

    return map

def encode(pos, width = 5, height = 5):
    map = precode(pos, width, height)
    map = map.flatten()
    return map

def round_prediction(x): # we want all outputs to be integer coordinates
    for i in range(len(x)):
        x[i] = int(x[i] + 0.5)

    return x