import numpy as np

def k2r(K):
    t, x, y, z = K[0], K[1], K[2], K[3]
    return [ t, z, -x, y ]

# read_2d is derived from read_gt
def read_2d(filename,delimiter):
    result = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            words = line.split(delimiter)
            result.append([ float(words[0].strip()), float(words[1].strip()), float(words[2].strip()), 0.0 ])
    return np.asarray(result, dtype=np.float64)

# read_3d is derived from read_orb
def read_3d(filename):
    result = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            words = line.split(' ')
            result.append(k2r([ float(words[0].strip()), float(words[1].strip()), float(words[2].strip()), float(words[3].strip()) ]))
    return np.asarray(result, dtype=np.float64)