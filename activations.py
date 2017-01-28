import numpy as np

def relu(x):
    out = []
    for inp in x:
        if inp>0:
            out.append(inp)
        else:
            out.append(0)

    return out
