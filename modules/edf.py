from io import BytesIO
import mne
import pandas as pd
import tempfile
import os

class BadAnnotations(Exception):
    pass

class EdfParser:

    def __init__(self, file: BytesIO):
        self.annotations = None
        self.stopEvent = "Stopped_Analyzer_-_Sleep_Events"
        self.annotationsTags = { 
            "Sleep_stage_W" : ["Sleep_stage_4" ,"Sleep_stage_W"], 
            "Sleep_stage_N1": ["Sleep_stage_1", "Sleep_stage_N1"], 
            "Sleep_stage_N2": ["Sleep_stage_N2", "Sleep_stage_2"], 
            "Sleep_stage_N3": ["Sleep_stage_N3", "Sleep_stage_3"], 
            "Sleep_stage_R": ["Sleep_stage_REM", "Sleep_stage_R"] 
        }

        # https://github.com/mne-tools/mne-python/pull/13156
        with tempfile.NamedTemporaryFile(suffix='.edf', delete_on_close=False, delete=False) as fp:
            fp.write(file.getbuffer())
            fp.seek(0)
            self.filename = fp.name
            self.edf = mne.io.read_raw_edf(fp.name, preload=True)
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

        offsets = times + rawAnnotations['duration'].astype(float)
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

        self.edf.set_annotations(annotations)
        self.annotations = annotations
        self.tags = events

    def crop(self, channel_picks: list):
        if self.annotations is None:
            raise Exception("Trying to crop a not annotated document")
        picks = self.edf.pick(channel_picks).reorder_channels(sorted(channel_picks))
        return picks.crop_by_annotations()
    
    def getTags(self):
        return self.tags

    def getInfo(self):
        return self.edf.info
    
    # https://github.com/mne-tools/mne-python/pull/13156
    def purge(self):
        os.remove(self.filename)
    
    def duration(self):
        return self.edf.duration

