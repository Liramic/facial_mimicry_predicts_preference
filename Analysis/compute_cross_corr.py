from General.init import *
from General.CorrolationMethods import  WTLCC
import numpy as np

def corr_headers(append_with=""):
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_corr{append_with}", f"cluster_{i}_lag{append_with}"]
    return headers

def add_mean_component_headers_listening():
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_mean_comp"]
    return headers

def add_mean_component_headers_reading():
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_mean_comp_reader", f"cluster_{i}_mean_comp_listener"]
    return headers

def corelation_lags_measurments(correlations, lags):
    if len(correlations) == 0:
        return [np.nan, np.nan]
    
    mean_corr = np.mean(correlations)
    mean_lag = np.mean(lags)
    return [mean_corr, mean_lag]

def add_mean_comps(chunk_data, current_reader):
    mean_comps = []
    for i in range(16):
        if chunk_data[i] is not np.nan:
            if current_reader == A:
                mean_comps += [np.mean(chunk_data[i][A]), np.mean(chunk_data[i][B])]
            elif current_reader == B:
                mean_comps += [np.mean(chunk_data[i][B]), np.mean(chunk_data[i][A])]
            elif current_reader == "actress":
                mean_comps += [np.mean(chunk_data[i])]
        else:
            if current_reader == "actress":
                mean_comps += [np.nan]
            else:
                mean_comps += [np.nan, np.nan]
    return mean_comps

def get_headers_reading():
    return corr_headers() + add_mean_component_headers_reading()

def get_headers_listening():
    return corr_headers() + add_mean_component_headers_listening()

def main_corrolation_of_single_comps_reading(current_session, both_comps, both_chunks, window_ms, max_lag_ms, downsample_factor_ms):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    numOfPixelsForXCorrWindow = window_ms//downsample_factor_ms
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow

    n = 16

    for key in both_chunks[0]:
        chunk_data = [np.nan]*16
        action = key.split("_")[0]
        current_reader = None
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)
        else:
            continue

        lengths = [0,0]
        for i in range(n):
            chunks_i = [[],[]]
            for participant in [A,B]:
                chunk = both_chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                if both_comps[participant][i] is not np.nan:
                    chunks_i[participant] = both_comps[participant][i][chunk_start:chunk_end]
                    lengths[participant] = len(chunks_i[participant])
                else:   
                    chunks_i[participant] = np.nan
            chunk_data[i] = chunks_i


        max_len = min(lengths)

        corrs = [[] for i in range(n)]
        lags =  [[] for i in range(n)]

        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            direction = "one-direction"
            for i in range(n):
                mat1 = chunk_data[i][A]
                mat2 = chunk_data[i][B]
                if mat1 is np.nan or mat2 is np.nan:
                    continue

                if ( current_reader == A):
                    max_corr, max_ind = WTLCC(mat2, mat1, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                else:
                    max_corr, max_ind = WTLCC(mat1, mat2, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                corrs[i].append(max_corr)
                lags[i].append(max_ind)

        cor_results = []
        for i in range(n):
            cor_results += corelation_lags_measurments(corrs[i], lags[i])

        current_result = [current_session, key, action] + \
            cor_results +\
            add_mean_comps(chunk_data, current_reader)
        
        results.append(current_result)
    return results


def main_corrolation_of_single_comps_listening(current_session, both_comps, both_chunks, actress_chunks, window_ms, max_lag_ms, downsample_factor_ms):
    results = []
    numOfPixelsForXCorrWindow = window_ms//downsample_factor_ms
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow
    n = 16
    current_reader = "actress"

    for key in both_chunks[0]:
        chunk_A = [np.nan]*16
        chunk_B = [np.nan]*16

        action = key.split("_")[0]
        if action.lower() != "listening":
            continue
        story = key.split("_")[1].split(".")[0]
        actress_data = actress_chunks[story].data

        lengths = [0,0]
        for i in range(n):
            for participant, chunk_p in zip([A,B], [chunk_A, chunk_B]):
                chunk = both_chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                if both_comps[participant][i] is not np.nan:
                    chunk_p[i] = both_comps[participant][i][chunk_start:chunk_end]
                    lengths[participant] = int(chunk_end-chunk_start)
                else:   
                    chunk_p[i] = np.nan
        
        max_len = min(lengths)
        max_len = min(max_len, len(actress_data[0]))

        corrs_A = [[] for i in range(n)]
        corrs_B = [[] for i in range(n)]
        lags_A = [[] for i in range(n)]
        lags_B =  [[] for i in range(n)]

        direction = "one-direction"

        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            for i in range(n):
                if actress_data[i] is np.nan:
                    continue

                matA = chunk_A[i]
                matB = chunk_B[i]
                matActress = actress_data[i]

                if matA is not np.nan:
                    max_corr, max_ind = WTLCC(matA, matActress, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                    corrs_A[i].append(max_corr)
                    lags_A[i].append(max_ind)
                
                if matB is not np.nan:
                    max_corr, max_ind = WTLCC(matB, matActress, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                    corrs_B[i].append(max_corr)
                    lags_B[i].append(max_ind)
    
        cor_results_A = []
        cor_results_B = []
        for i in range(n):
            cor_results_A += corelation_lags_measurments(corrs_A[i], lags_A[i])
            cor_results_B += corelation_lags_measurments(corrs_B[i], lags_B[i])

        current_result_A = [current_session, key, action, "A"] + \
            cor_results_A +\
            add_mean_comps(chunk_A, current_reader)
        current_result_B = [current_session, key, action, "B"] + \
            cor_results_B +\
            add_mean_comps(chunk_B, current_reader)
        
        results.append(current_result_A)
        results.append(current_result_B)
    
    return results