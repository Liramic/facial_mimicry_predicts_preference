import numpy as np

def WTLCC(mat1, mat2, tMaxMs, downsample_window_ms, s, e, direction, shouldReturnXcorrVec=False):
    tMax = tMaxMs // downsample_window_ms
    tInc = 5
    s1 = 0
    if "one" in direction.lower():
        s1 = s
    elif s - tMax > 0:
        s1 = s - tMax

    mat1 = mat1.flatten()
    mat2 = mat2.flatten()

    e1 = e + tMax
    if e1 > len(mat1):
        e1 = len(mat1)
    
    mat1 = mat1[s1:e1]
    mat2 = mat2[s:e]

    max_corr = -np.inf
    max_ind = -1

    correlations = []
    # Compute Pearson correlation for each shift within the valid range
    for i in range(0, len(mat1) - len(mat2) + 1, tInc):
        window = mat1[i:i+len(mat2)]
        corr = np.corrcoef(window, mat2)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
        else:
            correlations.append(0)
        
        if corr > max_corr:
            max_corr = corr
            max_ind = i

    max_ind = np.abs(s1 - s + max_ind) * downsample_window_ms

    if shouldReturnXcorrVec:
        return max_corr, max_ind, correlations

    return max_corr, max_ind