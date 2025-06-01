import pyedflib
from scipy import signal
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt
from General.HelperFunctions import isMyAnnotation

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], analog=False, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        y = filtfilt(b,a,data)
        return y

def window_rms_single(a, window_size=800):
    # Square the values
    a_squared = np.squeeze(a**2)

    # Apply a moving window average to the squared values
    window = np.squeeze(np.ones((1,window_size)) / float(window_size))
    a_squared_convolved = np.convolve(a_squared, window, mode='same')

    # Calculate the square root of the averaged squared values
    return np.sqrt(a_squared_convolved)

def filter_signal(data, fs):
    # apply notch filter to remove 50Hz noise
    b, a = signal.iirnotch(50, 30, fs)
    data = signal.filtfilt(b, a, data, axis=1)
    # apply bandpass filter to remove high frequency noise
    b, a = signal.butter(4, [35, 350], fs=fs, btype='band')
    filtered_signal = signal.filtfilt(b, a, data, axis=1)
    # add notch filter of 200 hz
    b, a = signal.iirnotch(200, 30, fs)
    filtered_signal = signal.filtfilt(b, a, filtered_signal)
    # add notch filter of 100 hz
    b, a = signal.iirnotch(100, 30, fs)
    filtered_signal = signal.filtfilt(b, a, filtered_signal)
    return filtered_signal


def NotchFilterSignal(noisySignal, sampling_rate, removed_frequency=50.0, Q_Factor=30.0):
    # Design notch filter
    b_notch, a_notch = signal.iirnotch(removed_frequency, Q_Factor, sampling_rate)
    return signal.filtfilt(b_notch, a_notch, noisySignal)

def ExtractIdAndPos(s):
    splitted = s.split("_")
    return ( int(splitted[1]), splitted[2])

def extractAnnotation(s : str, timing):
    s = s.lower()
    splitted = s.split("_")
    a = Annotation()
    a.Time = timing
    if ( splitted[0] in ["smile", "angry", "blink"]):
        a.Type = splitted[0]
        a.TrialId = int(splitted[1])
        a.order = int(splitted[1])
        a.isStart = splitted[2] == "start"
        return a

    if (splitted[0][0] == "a"):
        a.Type = "Listening"
    elif ( splitted[0][0] == "r"):
        a.Type = "Reading"
    
    if(splitted[0][1] == "s"):
        a.isStart = True
    else:
        a.isStart = False
    
    a.order = int(splitted[1])
    a.TrialId = int(splitted[3])
    a.Story = splitted[5]
    return a

def GetCallibrationsTicks(chunks, calibration_state_to_analyze, num_repeats = 3):
    ticks = []
    for i in range(0,num_repeats):
        key = '%s__%d' % (calibration_state_to_analyze, i)
        chunk = chunks[key]
        ticks.append(int(chunk.Start.Time))
        ticks.append(int(chunk.End.Time))
    
    return ticks

class EdfAnalyzer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def readEdf(f : pyedflib.EdfReader, doButter = True):
        sampling_rate = int(f.getSampleFrequency(0))
        n = 16
        sigbufs = np.array([f.readSignal(i) for i in range(0,n)])
        #Filter before ICA
        sigbufs = NotchFilterSignal(sigbufs, sampling_rate)
        if (doButter):
            sigbufs = filter_signal(sigbufs, sampling_rate) #butter_bandpass_filter(sigbufs, lower_bound, 400.0, sampling_rate)
        
        return sigbufs, sampling_rate

    @staticmethod
    def getCallibrationTicks(chunks, calibration_state_to_analyze = "smile"):
        return GetCallibrationsTicks(chunks, calibration_state_to_analyze)

    @staticmethod
    def combine_callibration_ticks(chunks):
        states = ["smile", "angry", "blink"]
        for state in states:
            start = chunks[state + "__0"].Start.Time
            end = chunks[state + "__2"].End.Time
            for i in range(3):
                key = state + "__" + str(i)
                del chunks[key]
            chunks[state] = Chunk()
            chunks[state].Start = Annotation()
            chunks[state].Start.Time = start
            chunks[state].End = Annotation()
            chunks[state].End.Time = end
        return chunks

    @staticmethod
    def getAnnotationChunks(f : pyedflib.EdfReader, correctBy, RmsDownsampleWindow = 1):
        annotations = f.readAnnotations()
        file_sampling_rate = int(f.getSampleFrequency(0))
        startIndex = np.where(annotations[2] == "StartExperiment")[0][-1] + 1

        endIndex = len(annotations[2])
        startCorrectIdx = np.where(annotations[2] == "Smile_0_start")[0][0]
        startCorrectionTime = annotations[0][startCorrectIdx] - 60 # fix error by xTrodes. # __class__.startCorrectionTime
        
        chunks = dict()
        timing_in_seconds = 0
        index=""

        toWindowConvertionValue = int(RmsDownsampleWindow * (file_sampling_rate/1000))
        if (RmsDownsampleWindow == 1):
            toWindowConvertionValue = 1


        for i in range(startIndex, endIndex):
            annotation = str(annotations[2][i])
            
            if ( not isMyAnnotation(annotation)):
                continue
            
            #todo - no need to perform *4000 outside of this code.
            timing_in_seconds = annotations[0][i] - correctBy - startCorrectionTime # fix error by xTrodes. __class__.startCorrectionTime
            timing = timing_in_seconds*file_sampling_rate/toWindowConvertionValue
            
            extractedAnnotation = extractAnnotation(annotation, timing)
            index = "%s_%s_%d" % (extractedAnnotation.Type, extractedAnnotation.Story, extractedAnnotation.TrialId)
            if ( index in chunks):
                if (extractedAnnotation.isStart):
                    chunks[index].Start = extractedAnnotation
                else:
                    chunks[index].End = extractedAnnotation
            else:
                chunks[index] = Chunk()
                if(extractedAnnotation.isStart):
                    chunks[index].Start = extractedAnnotation
                else:
                    chunks[index].End = extractedAnnotation
        
        #get the end of the last index:
        trim_start = int(startCorrectionTime*file_sampling_rate)
        trim_end = trim_start + int((timing_in_seconds+5)*file_sampling_rate) # extra 5 seconds
        
        return chunks, trim_start, trim_end

    @staticmethod
    def RemoveRedundentData(x, trim_start, trim_end):
        return x[:, trim_start:trim_end]
    
    @staticmethod
    def Read(file_path, correctBy=0, downsample_window_in_ms=1,  doButter = True, forEEG = False):
        # Read EDF: if downsample_window_in_ms is 1, the user should downsample the data himself. annotations are handled here.
        f = pyedflib.EdfReader(file_path)
        sigbufs, freq = __class__.readEdf(f, doButter, forEEG)
        chunks, trim_start, trim_end = __class__.getAnnotationChunks(f, correctBy, downsample_window_in_ms)
        sigbufs = __class__.RemoveRedundentData(sigbufs, trim_start, trim_end)
        f.close()
        return sigbufs, chunks, freq
    
    @staticmethod
    def ICA(matrix, n_components=16, whiten=True):
        if (whiten):
            whiten = 'unit-variance'
        transformer = FastICA(n_components=n_components, random_state=0, whiten=whiten)
        x= np.transpose(transformer.fit_transform(np.transpose(matrix))) #FAST ICA
        w = transformer.components_
        return w, x

    @staticmethod
    def window_rms(a, window_size_in_ms=200, original_freq = 4000):
        factor = original_freq//1000
        factored_window_size = window_size_in_ms*factor
        if (a.shape[0] == 1 or len(a.shape) < 2 or a.shape[1] == 1):
            return window_rms_single(a, factored_window_size)
        return np.array([window_rms_single(x,factored_window_size) for x in a])

class Chunk:
    def __init__(self) -> None:
        self.Start = None
        self.End = None
        self.Data = None

    #todo: test these.
    def GyroStart_whenNoRmsOnGyro(self, emgRmsDownsampleWindow = 1):
        if emgRmsDownsampleWindow == 1:
            return self.Start.Time//4
        return int(self.Start.Time * emgRmsDownsampleWindow)
        
    def GyroEnd_whenNoRmsOnGyro(self, emgRmsDownsampleWindow = 1):
        if emgRmsDownsampleWindow == 1:
            return self.End.Time//4
        return int(self.End.Time * emgRmsDownsampleWindow)


class ExtendedChunk(Chunk):
    def __init__(self, chunk=None, isOther=False) -> None:
        super().__init__()
        if (chunk!=None):
            self.Start = chunk.Start
            self.End = chunk.End
            self.Data = chunk.Data
        
        self.isOther = isOther

class Annotation:
    def __init__(self) -> None:
        self.Type = ""
        self.Story = ""
        self.Reader = ""
        self.order = None
        self.TrialId = None
        self.isStart = False
        self.Time = None
