import time
from datetime import timedelta

def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 1000
    return sec

class EpicSoundsRecord(object):
    def __init__(self, tup, sampling_rate=24000):
        self._index = str(tup[0])
        self._series = tup[1]
        self.sampling_rate = sampling_rate

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def video_id(self):
        return self._series['video_id']

    @property
    def annotation_id(self):
        return self._series['annotation_id']

    @property
    def start_audio_sample(self):
        return int(timestamp_to_sec(self._series["start_timestamp"]) * self.sampling_rate)

    @property
    def end_audio_sample(self):
        return int(timestamp_to_sec(self._series["stop_timestamp"]) * self.sampling_rate)

    @property
    def label(self):
        return self._series["class_id"] if "class_id" in self._series else 0

    @property
    def num_audio_samples(self):
        return self.end_audio_sample - self.start_audio_sample