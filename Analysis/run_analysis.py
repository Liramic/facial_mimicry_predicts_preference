from General.init import *
from General.HelperFunctions import cleanSpace, GetSessions
from Analysis.data_loading import load_dyad_data, load_actress_data
from Analysis.compute_cross_corr import get_headers_reading, get_headers_listening, main_corrolation_of_single_comps_reading, main_corrolation_of_single_comps_listening, get_headers_both_listeners, main_corrolation_of_single_comps_listeninig_participants, main_correlation_of_combinations_reading, get_headers_reading_combinations, get_headers_reading_GC, main_correlation_reading_GC
import pandas as pd
from PsychData.Experiment_CSV import get_user_choice_results
from tqdm import tqdm

def wrap_in_default_headers(headers, action="reading"):
    if "reading" in action or action == "listening_participants" :
        return ["session", "key","action"] + headers + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']
    else:
        return ["session", "key","action", 'participant'] + headers + ['choice', 'rtChoice', 'isOther']

def run_analysis(rms_size_ms, ds_factor, window_ms, max_lag_ms, csv_out_path, action="reading"):
    all_correlations = []
    actress_chunks = None
    
    if action == "reading":
        headers = wrap_in_default_headers(get_headers_reading())
    elif action == "listening":
        headers = wrap_in_default_headers(get_headers_listening(), action=action)
        actress_chunks = load_actress_data(rms_size_ms, ds_factor)

    for session in tqdm(GetSessions(data_path)):
        both_comps, both_chunks = load_dyad_data(session, ds_factor, rms_size_ms)
        if action == "reading":
            corr_results = main_corrolation_of_single_comps_reading(session, both_comps, both_chunks, window_ms, max_lag_ms, ds_factor)
        elif action == "listening":
            corr_results = main_corrolation_of_single_comps_listening(session, both_comps, both_chunks, actress_chunks, window_ms, max_lag_ms, ds_factor)


        if action == "reading":
            for i in range(len(corr_results)):
                key = corr_results[i][1]
                story = key.split("_")[1].split(".")[0]
                choices = get_user_choice_results(data_path, session, None, story)
                corr_results[i] += choices

        else: #listening
            for i in range(len(corr_results)):
                key = corr_results[i][1]
                story = key.split("_")[1].split(".")[0]
                p = corr_results[i][3]
                choice = get_user_choice_results(data_path, session, p, story)
                corr_results[i] += choice

        all_correlations += corr_results
        cleanSpace()
        print("finished session " + session)

    df = pd.DataFrame(all_correlations)
    df.columns = headers
    if (csv_out_path!= None):
        df.to_csv(csv_out_path, index=False)
    else:
        return df

