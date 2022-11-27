from math import sqrt
import numpy as np
from scipy.interpolate import interp1d

def interpolate(source, target):
    tts = target[0,0]
    tte = target[-1,0]
    start = np.argmax(source[:,0]>tts)
    end = np.argmax(source[:,0]>tte) - 1

    target_timestamps, xs, ys, zs = target[:,0], target[:,1], target[:,2], target[:,3]
    source_timestamps = source[start:end,0]

    result_t = np.zeros(source[start:end,:].shape)
    result_s = np.zeros(source[start:end,:].shape)

    fx = interp1d(target_timestamps, xs)
    fy = interp1d(target_timestamps, ys)
    fz = interp1d(target_timestamps, zs)

    new_xs = fx(source_timestamps);
    new_ys = fy(source_timestamps);
    new_zs = fz(source_timestamps);

    result_t[:,0] = source_timestamps
    result_t[:,1] = new_xs
    result_t[:,2] = new_ys
    result_t[:,3] = new_zs

    result_s[:,0] = source_timestamps
    result_s[:,1:] = source[start:end,1:]

    return result_s, result_t

def interpolate_for_time(data, timestamps):
    tts = data[0,0]
    tte = data[-1,0]
    start = np.argmax(timestamps>tts)
    end = timestamps.shape[0] if timestamps[-1] < tte else  np.argmax(timestamps>tte)
    end -= 1

    data_timestamps, xs, ys, zs = data[:,0], data[:,1], data[:,2], data[:,3]
    fixed_timestamps = timestamps[start:end+1]

    result = np.zeros((fixed_timestamps.shape[0], data.shape[1]))
    result[:,0] = fixed_timestamps

    for col in range(1,data.shape[1]):
        f_col = interp1d(data_timestamps, data[:,col])
        new_col = f_col(fixed_timestamps)
        result[:,col] = new_col.copy()

def kabsch(P, Q):
    RP, RQ = np.zeros(P.shape), np.zeros(Q.shape)
    pt, qt = P[:,0], Q[:,0]
    P, Q = P[:,1:4], Q[:,1:4]

    # Find the centroids
    pc = np.mean(P, axis=0)
    qc = np.mean(Q, axis=0)

    # Center the shapes
    P_ = P - pc
    Q_ = Q - qc

    # Find the covariance matrix
    A = np.matmul(P_.T, Q_)

    # Compute optimal rotation U using SVD
    V, s, W = np.linalg.svd(A)
    d = np.linalg.det( np.matmul(V, W) )
    U = np.matmul(np.matmul(V, np.identity(P_.shape[1])), W)

    TRP = np.matmul(P_, U)
    TRQ = Q_

    # Find the Root Mean Squared error between the adjusted shapes
    tmp = TRP - TRQ
    rms = sqrt(np.sum(tmp*tmp)/P.shape[0])

    RP[:,0] = pt
    RP[:,1:4] = TRP
    RQ[:,0] = qt
    RQ[:,1:4] = TRQ

    return U, rms, RP, RQ

def analyze(gt, vanilla):
    gt, vanilla = interpolate(gt, vanilla)
    _, rms_vanilla, r_vanilla, gt = kabsch(vanilla, gt)

    # if r_vanilla.shape[0] > gt.shape[0]:
    #     r_vanilla = r_vanilla[:gt.shape[0],:]

    return gt, r_vanilla, rms_vanilla

def scale(model, data):
    dots, norms = 0.0, 0.0
    for column in range(1,data.shape[1]):
        dots += np.dot(data[:,column].T, model[:,column])
        normi = np.linalg.norm(model[:,column])
        norms += normi*normi
    s = float(dots/norms)
    return data/s, s
