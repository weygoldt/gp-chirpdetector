import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from thunderfish.dataloader import DataLoader


class ConfLoader:
    """
    Load configuration from yaml file as class attributes
    """

    def __init__(self, path: str) -> None:
        with open(path) as file:
            try:
                conf = yaml.safe_load(file)
                for key in conf:
                    setattr(self, key, conf[key])
            except yaml.YAMLError as error:
                raise error


class LoadData:
    """
    Attributes
    ----------
    data : DataLoader object containing raw data
    samplerate : sampling rate of raw data
    time : array of time for tracked fundamental frequency
    freq : array of fundamental frequency
    idx : array of indices to access time array
    ident : array of identifiers for each tracked fundamental frequency
    ids : array of unique identifiers exluding NaNs
    """

    def __init__(self, datapath: str) -> None:
        # load raw data
        self.datapath = datapath
        self.file = os.path.join(datapath, "traces-grid1.raw")
        if os.path.isfile(self.file) == False:
            self.raw = np.load(os.path.join(datapath, "raw.npy"))
            self.raw_rate = 20000.0
        else:
            self.raw = DataLoader(self.file, 60.0, 0, channel=-1)
            self.raw_rate = self.raw.samplerate

        # load wavetracker files
        self.time = np.load(datapath + "times.npy", allow_pickle=True)
        self.freq = np.load(datapath + "fund_v.npy", allow_pickle=True)
        self.powers = np.load(datapath + "sign_v.npy", allow_pickle=True)
        self.idx = np.load(datapath + "idx_v.npy", allow_pickle=True)
        self.ident = np.load(datapath + "ident_v.npy", allow_pickle=True)
        self.ids = np.unique(self.ident[~np.isnan(self.ident)])

    def __repr__(self) -> str:
        return f"LoadData({self.file})"

    def __str__(self) -> str:
        return f"LoadData({self.file})"


def make_outputdir(path: str) -> str:
    """
    Creates a new directory where the path leads if it does not already exist.

    Parameters
    ----------
    path : string
        path to the new output directory

    Returns
    -------
    string
        path of the newly created output directory
    """

    if os.path.isdir(path) == False:
        os.mkdir(path)
    return path
