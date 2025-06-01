from EDF.EdfAnalyzer import EdfAnalyzer
import numpy as np
import os
import pyedflib
from dataclasses import dataclass
from clustering_by_hila.classifying_ica_components import wavelet_denoising, whiten, center
from scipy import signal

edf_path = r"path\to\actress\edf"
correct_value = -0.5 #based on EDF inspection.


def get_folder_path(p=edf_path):
    return os.path.dirname(p)

def open_w_if_exisis():
    name = "w.npy"
    try:
        #get folder path from the edf path:
        return np.load(os.path.join(get_folder_path(), name))
    except:
        return None

def open_whiten_if_exisis():
    try:
        return np.load(os.path.join(get_folder_path(), "white.npy"))
    except:
        return None
    
def save_w(w):
    name = "w.npy"
    np.save(os.path.join(get_folder_path(), name), w)

def is_my_annotation(annotation):
    s = str(annotation).lower()
    return "story_" in s

def extract_story_number(annotation):
    s = str(annotation).lower()
    return int(s.split("_")[1])

def parse_annotation(annotation):
    s = str(annotation).lower()
    #regex to extarct 15 and "start" from : "story_Slide15.PNG_start"
    s = s[len("story_slide"):]
    parts = s.split(".")
    story_number = parts[0]
    annotation_type = parts[1].split("_")[1]

    return story_number, annotation_type

@dataclass
class ActressChunk:
    story_number: str
    start: int
    end: int
    data: np.ndarray = None

def read_annotations(f, freq, downsampleWindowInMs, correction):
    annotations = f.readAnnotations()
    chunks = []
    N = len(annotations[0])
    for i in range(N):
        anotation  = [annotations[0][i], annotations[1][i], annotations[2][i]]
        if is_my_annotation(anotation[2]):
            timing = anotation[0] - correction
            timing = (timing * freq)/(4*downsampleWindowInMs) # 4 because original freq is 4000
            timing = int(timing)

            story_number, annotation_type = parse_annotation(anotation[2])
            if ( annotation_type == "start"):
                chunks.append(ActressChunk(story_number, timing, -1))
            if ( annotation_type == "end"):
                chunks[-1].end = timing
    return chunks


def get_listening_stories():
    return [10, 5, 12, 6, 11, 3, 8, 16]


def build_actress_chunks(chunks, downsampleWindowInMs):
    chosen_chunk_numbers = get_listening_stories()
    chunks = [chunk for chunk in chunks if int(chunk.story_number) in chosen_chunk_numbers]

    chunks_as_dict = {}
    for chunk in chunks:
        #taking the last instance:
        chunks_as_dict[chunk.story_number] = chunk

    for key in chunks_as_dict:
        print(f"{key}: len: {(chunks_as_dict[key].end - chunks_as_dict[key].start)*downsampleWindowInMs/1000}")

    return chunks_as_dict

def export_actress_chunked_data_with_dwt(downsampleWindowInMs=50, rms_size_ms=5):
    f = pyedflib.EdfReader(edf_path)
    Y, freq = EdfAnalyzer.readEdf(f)
    chunks = read_annotations(f, freq, downsampleWindowInMs, correct_value)
    chunks_as_dict = build_actress_chunks(chunks, downsampleWindowInMs)
    
    Y = wavelet_denoising(Y, freq, 'db15')
    Y, _ = center(Y)
    Y = whiten(Y)

    w = open_w_if_exisis(dwt=True)
    
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16, whiten=False)
        save_w(w, dwt=True)
    else:
        independentComponents = np.matmul(w,Y)

    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    hz = int(1000/downsampleWindowInMs)
    independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)
    
    return independentComponents, chunks_as_dict