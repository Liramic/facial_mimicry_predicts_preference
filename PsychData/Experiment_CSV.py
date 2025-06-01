import pandas as pd
import os

#['story' , 'whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

def convert_row(df, i, whose_reading):
    s1 = str(df["StoryOrder1"][i])
    s2 = str(df["StoryOrder2"][i])
    choiceA = int(df["UserANumberChoice"][i])
    choiceB = int(df["UserBNumberChoice"][i])
    rtA = str(df["UserA_choice.rt"][i])
    rtB = str(df["UserB_choice.rt"][i])
    isOther = "other" in str(df["AudioInstruction"][i]).lower()
    
    s1_arr = [s1, whose_reading, int(choiceA == 1), int(choiceB == 1), rtA, rtB, int(isOther), int(choiceA == choiceB)]
    s2_arr = [s2, whose_reading, int(choiceA == 2), int(choiceB == 2), rtA, rtB, int(isOther), int(choiceA == choiceB)]

    return s1_arr, s2_arr


def read_experiment_csv(data_path, session):
    uc_list = []
    session_folder = os.path.join(data_path, session)
    fname = [x for x in os.listdir(session_folder) if "TwoPeopleEmg" in x][0]
    csvPath = os.path.join(session_folder, fname)
    df = pd.read_csv(csvPath)

    whose_reading_arr = [1,2,1,2]
    for i in range(len(df)):
        if pd.isna(df["Trigger0"][i]):
            continue
        if ( "rs" in str.lower(df["Trigger0"][i])):
            whose_reading = whose_reading_arr.pop(0)
        elif ("as" in str.lower(df["Trigger0"][i])):
            whose_reading = 0
        else:
            continue #smile\frown\blink
        s1_arr, s2_arr = convert_row(df, i, whose_reading)
        uc_list.append(s1_arr)
        uc_list.append(s2_arr)
    return uc_list

def get_user_choice_results(data_path, session, p=None, story_num=None):
    #if p is None, return the regular results.
    #else: listening results
    res = read_experiment_csv(data_path, session)
    if p is None:
        return res
    
    correct_row = []
    #find the right row:
    for row in res:
        if row[0].split(".")[0] == story_num:
            correct_row = row
            break
    
    #return choice, rt, isOther
    if p == "A":
        return [correct_row[2],correct_row[4], correct_row[6]]
    return [correct_row[3],correct_row[5], correct_row[6]]
    