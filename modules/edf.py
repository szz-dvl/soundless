from io import BytesIO
from warnings import warn
import mne
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BadSamplingFreq(Exception):
    pass

class EdfParser:

    def __init__(self, file: BytesIO):
        mne.set_log_level(verbose="CRITICAL")
        self.annotations = None
        self.picks = None
        self.stopEvent = "Stopped_Analyzer_-_Sleep_Events"
        self.annotationsTags = { 
            "Sleep_stage_W" : ["Sleep_stage_4" ,"Sleep_stage_W"], 
            "Sleep_stage_N1": ["Sleep_stage_1", "Sleep_stage_N1"], 
            "Sleep_stage_N2": ["Sleep_stage_N2", "Sleep_stage_2"], 
            "Sleep_stage_N3": ["Sleep_stage_N3", "Sleep_stage_3"], 
            "Sleep_stage_R": ["Sleep_stage_REM", "Sleep_stage_R"] 
        }
        self.tagsToClass = { 
            "Sleep_stage_W" : 4, 
            "Sleep_stage_N1": 1, 
            "Sleep_stage_N2": 2, 
            "Sleep_stage_N3": 3, 
            "Sleep_stage_R": 0 
        }

        self.freqBands = {
            "delta": [0.5, 4.5],
            "theta": [4.5, 8.5],
            "alpha": [8.5, 11.5],
            "sigma": [11.5, 15.5],
            "beta": [15.5, 30],
        }

        # https://github.com/mne-tools/mne-python/pull/13156
        with tempfile.NamedTemporaryFile(suffix='.edf', delete_on_close=False, delete=False) as fp:
            fp.write(file.getbuffer())
            fp.seek(0)
            self.filename = fp.name
            self.edf = mne.io.read_raw_edf(fp.name, preload=True, infer_types=True, verbose="error")
            if self.edf.info["sfreq"] != 200.0:
                warn(f"Bad sampling frequency: {self.edf.info["sfreq"]}")
                #raise BadSamplingFreq(self.edf.info["sfreq"])
            # self.df = pd.DataFrame(self.edf.get_data().transpose(), columns=self.edf.ch_names)

    def getChannelTypes(self):
        return self.edf.get_channel_types()
    
    def __selectAnnotations(self, rawAnnotations: pd.DataFrame):
        measDate = self.edf.info['meas_date']
        measDay = measDate.strftime("%Y-%m-%d")
        
        times = rawAnnotations['time'].copy()
        times = measDay + " " + times
        times = pd.to_datetime(times, format='%Y-%m-%d %H:%M:%S', utc=True)
        
        # Add a day past twelve
        times[times.dt.hour < measDate.hour] = times[times.dt.hour < measDate.hour] + pd.Timedelta(days = 1)
        times = (times - measDate) / pd.Timedelta(seconds=1)
        
        events = rawAnnotations['event'].copy()
        aggregated = []

        for tag, candidates in self.annotationsTags.items():
            events.loc[events.isin(candidates)] = tag
            aggregated.append(tag)
        
        eventsIdxs = events.index[events.isin(aggregated)].tolist()

        idxs = rawAnnotations.index[rawAnnotations['event'] == self.stopEvent].tolist()

        if len(idxs) == 1:
            stopIndex = idxs[0]
        else:
            stopIndex = rawAnnotations['event'].shape[0]

        # Add one second offset trying to avoid truncation errors in crop()
        paddedDurations = rawAnnotations['duration'].astype(float) + 1
        offsets = times + paddedDurations
        offsetsIdxs = offsets.index[offsets > self.edf.duration].tolist()

        if len(offsetsIdxs) > 0:
            minOffset = min(offsetsIdxs)
            stopIndex = min([stopIndex, minOffset])

        # Discard events that are out of file boundaries
        eventsIdxs = list(filter(lambda x: x < stopIndex, eventsIdxs))
        
        events = events.iloc[eventsIdxs]
        times = times.iloc[eventsIdxs]
        durations = rawAnnotations['duration'].iloc[eventsIdxs].astype(float)

        return times, events, durations


    def setAnottations(self, rawAnnotations: pd.DataFrame):

        times, events, durations = self.__selectAnnotations(rawAnnotations)

        annotations = mne.Annotations(
            onset=times, 
            orig_time=None, 
            duration=durations, 
            description=events
        )

        self.edf.set_annotations(annotations, emit_warning=False)
        self.annotations = annotations
        self.tags = events
    
    def __getEventIds(self, present): 
        eventId = {}

        for event, label in self.tagsToClass.items():
            if label in present:
                eventId[event] = label

        return eventId

    def __getEpochs(self):
        events, _ = mne.events_from_annotations(self.edf, event_id=self.tagsToClass, chunk_duration=30.0)
        
        return mne.Epochs(
            raw=self.edf,
            events=events,
            event_id=self.__getEventIds(np.unique(events[:, -1])),
            tmin=0.0,
            tmax=30.0 - 1.0 / self.edf.info["sfreq"],
            baseline=None,
            on_missing="raise"
        )

    def featuresPerEvent(self, picks: pd.DataFrame):
        # https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html

        chann_names = picks["name"].to_list()
        epochs = self.__getEpochs()
        
        spectrum = epochs.compute_psd(picks=chann_names, fmin=0.5, fmax=30.0, method="multitaper").reorder_channels(chann_names)
        psds, freqs = spectrum.get_data(return_freqs=True)

        # Delete events where no presence/power is found in the given frequencies
        purged_psds = np.delete(psds, np.where(np.sum(psds, axis=-1) == 0)[0], axis=0)

        # Remove useless labels
        labels = np.delete(epochs.events[:, 2], np.where(np.sum(psds, axis=-1) == 0)[0])

        # Normalize the PSDs
        purged_psds /= np.sum(purged_psds, axis=-1, keepdims=True)

        X = []
        for fmin, fmax in self.freqBands.values():
            psds_band = purged_psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            X.append(psds_band.reshape(len(purged_psds), -1))

        return np.concatenate(X, axis=1), labels

    def crop(self, channelPicks: list):
        if self.annotations is None:
            raise Exception("Trying to crop a not annotated document") 
        picks = self.edf.pick(channelPicks).reorder_channels(channelPicks)
        return picks.crop_by_annotations()
    
    def getTags(self):
        return list(map(lambda t: self.tagsToClass[t], self.tags))

    def getInfo(self):
        return self.edf.info
    
    # https://github.com/mne-tools/mne-python/pull/13156
    def purge(self):
        os.remove(self.filename)
    
    def duration(self):
        return self.edf.duration

