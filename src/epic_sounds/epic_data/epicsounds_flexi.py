from curses import meta
import os
import pandas as pd
import torch
import h5py
import torch.utils.data
import numpy as np
from fvcore.common.file_io import PathManager

from . import logging as logging

from .build import DATASET_REGISTRY
from .epicsounds_record import EpicSoundsRecord

from .spec_augment import combined_transforms,combined_transforms_flexi
from .audio_loader_epicsounds import pack_audio_flexi
from . import utils as utils

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epicsounds_flexi(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC Sounds".format(mode)
        self.cfg = cfg
        self.mode = mode


        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        # self.audio_dataset = pickle.load(open(cfg.EPICSOUNDS.AUDIO_DATA_FILE, 'rb'))
        self.audio_dataset = None
        logger.info("Constructing EPIC Sounds flexi {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [os.path.join(self.cfg.EPICSOUNDS.ANNOTATIONS_DIR, self.cfg.EPICSOUNDS.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(self.cfg.EPICSOUNDS.ANNOTATIONS_DIR, self.cfg.EPICSOUNDS.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(self.cfg.EPICSOUNDS.ANNOTATIONS_DIR, self.cfg.EPICSOUNDS.TEST_LIST)]
        else:
            path_annotations_pickle = [os.path.join(self.cfg.EPICSOUNDS.ANNOTATIONS_DIR, file)
                                       for file in [self.cfg.EPICSOUNDS.TRAIN_LIST, self.cfg.EPICSOUNDS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert PathManager.exists(file), "{} dir not found".format(file)

        self._video_records = []
        self._temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    self._video_records.append(
                            EpicSoundsRecord(tup, self.cfg.AUDIO_DATA.SAMPLING_RATE)
                        )
                    self._temporal_idx.append(idx)
        assert (
                len(self._video_records) > 0
        ), "Failed to load Audio Annotations split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing audio annotations dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.audio_dataset is None:
            self.audio_dataset = h5py.File(self.cfg.EPICSOUNDS.AUDIO_DATA_FILE, 'r')

        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]

        spectrogram = pack_audio_flexi(
                self.cfg, 
                self.audio_dataset,
                self._video_records[index], 
                temporal_sample_index
            )
        # Normalization.
        spectrogram = spectrogram.float()
        if self.mode in ["train", "train+val"]:
            # Data augmentation.
            # C T F -> C F T
            spectrogram = spectrogram.permute(0, 2, 1)
            # SpecAugment
            spectrogram = combined_transforms_flexi(spectrogram,self.cfg)
            # C F T -> C T F
            spectrogram = spectrogram.permute(0, 2, 1)
        label = self._video_records[index].label
        # spectrogram = utils.pack_pathway_output(self.cfg, spectrogram)

        metadata = {
                "annotation_id": self._video_records[index].annotation_id
            }
        
        label_indices = np.zeros(44)
        label_indices[label] = 1.0
        label_indices = torch.FloatTensor(label_indices)
        # import ipdb; ipdb.set_trace()

        return spectrogram.transpose(-1,-2), label_indices

    def __len__(self):
        return len(self._video_records)
