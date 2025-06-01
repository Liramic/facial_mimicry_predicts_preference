import pandas as pd
from Analysis.compute_cross_corr import get_headers_reading, get_headers_listening

feature_names = [f"cluster_{i}_corr" for i in range(16)]

def create_minus_coll(row, header):
    #if either is nan return nan
    if pd.isna(row[header + "_S1"]) or pd.isna(row[header + "_S2"]):
        return float('nan')
    return (row[header + "_S2"] - row[header + "_S1"])

def get_listener_choice(second_row):
    if getattr(second_row, "whoseReading") == 1:
        return getattr(second_row, "isChoiceB")
    return getattr(second_row, "isChoiceA")

def get_reader_choice(second_row):
    if getattr(second_row, "whoseReading") == 1:
        return getattr(second_row, "isChoiceA")
    return getattr(second_row, "isChoiceB")

def export_reading_to_r(csv_path):
    data = pd.read_csv(csv_path)
    data_odd = data.iloc[::2, :].reset_index(drop=True)
    data_even = data.iloc[1::2, :].reset_index(drop=True)

    new_df = pd.DataFrame()

    for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
        #story 1 in odd_row
        #story 2 in even_row

        new_row = {}
        new_row["session"] = getattr(odd_row, "session")
        new_row["story_1"] = getattr(odd_row, "key").split("_")[1].split(".")[0]
        new_row["story_2"] = getattr(even_row, "key").split("_")[1].split(".")[0]
        new_row["listener_choice"] = get_listener_choice(even_row)
        new_row["reader_choice"] = get_reader_choice(even_row)


        for header in [x for x in get_headers_reading() if "optimal" not in x]:
            new_row[header + "_S1"] = getattr(odd_row, header)
            new_row[header + "_S2"] = getattr(even_row, header)
            new_row["minus_" + header] = create_minus_coll(new_row, header)
        
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    new_df.to_csv(csv_path.replace(".csv", "_for_R.csv"), index = False)


def export_listening_to_r(csv_path):
    data = pd.read_csv(csv_path)
    new_df = pd.DataFrame()

    data = data.reset_index(drop=True)

    for p in ["A","B"]:
        df_p = data[data["participant"] == p]
        df_p = df_p.reset_index(drop=True)

        data_odd = df_p.iloc[::2, :].reset_index(drop=True)
        data_even = df_p.iloc[1::2, :].reset_index(drop=True)
        for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
        #story 1 in odd_row
        #story 2 in even_row
            new_row = {}
            new_row["session"] = getattr(odd_row, "session")
            new_row["story_1"] = getattr(odd_row, "key").split("_")[1].split(".")[0]
            new_row["story_2"] = getattr(even_row, "key").split("_")[1].split(".")[0]
            new_row["listener_choice"] = getattr(even_row, "choice") #get_listener_choice(even_row)
            new_row["is_other"] = getattr(even_row, "isOther")
            new_row["p"] = getattr(even_row, "participant")

            for header in get_headers_listening():
                new_row[header + "_S1"] = getattr(odd_row, header)
                new_row[header + "_S2"] = getattr(even_row, header)
                new_row["minus_" + header] = create_minus_coll(new_row, header)
            
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    new_df.to_csv(csv_path.replace(".csv", "_for_R.csv"), index = False)