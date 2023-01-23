import numpy as np

import os 

import numpy as np
import matplotlib.pyplot as plt 
from thunderfish.powerspectrum import decibel

from IPython import embed
from pandas import read_csv
from modules.logger import makeLogger
from modules.plotstyle import PlotStyle

ps = PlotStyle()

logger = makeLogger(__name__)


class Behavior:
    """Load behavior data from csv file as class attributes
        Attributes
    ----------
    behavior: 0: chasing onset, 1: chasing offset, 2: physical contact
    behavior_type:         
    behavioral_category:   
    comment_start:         
    comment_stop:          
    dataframe: pandas dataframe with all the data            
    duration_s:             
    media_file:            
    observation_date:      
    observation_id:        
    start_s: start time of the event in seconds               
    stop_s:  stop time of the event in seconds               
    total_length:          
    """

    def __init__(self, folder_path: str) -> None:
        

        LED_on_time_BORIS = np.load(os.path.join(folder_path, 'LED_on_time.npy'), allow_pickle=True)

        csv_filename = [f for f in os.listdir(folder_path) if f.endswith('.csv')][0]
        logger.info(f'CSV file: {csv_filename}')
        self.dataframe = read_csv(os.path.join(folder_path, csv_filename))

        self.chirps = np.load(os.path.join(folder_path, 'chirps.npy'), allow_pickle=True)
        self.chirps_ids = np.load(os.path.join(folder_path, 'chirps_ids.npy'), allow_pickle=True)

        self.ident = np.load(os.path.join(folder_path, 'ident_v.npy'), allow_pickle=True)
        self.idx = np.load(os.path.join(folder_path, 'idx_v.npy'), allow_pickle=True)
        self.freq = np.load(os.path.join(folder_path, 'fund_v.npy'), allow_pickle=True)
        self.time = np.load(os.path.join(folder_path, "times.npy"), allow_pickle=True)
        self.spec = np.load(os.path.join(folder_path, "spec.npy"), allow_pickle=True)    

        for k, key in enumerate(self.dataframe.keys()):
            key = key.lower() 
            if ' ' in key:
                key = key.replace(' ', '_')
                if '(' in key:
                    key = key.replace('(', '')
                    key = key.replace(')', '')
            setattr(self, key, np.array(self.dataframe[self.dataframe.keys()[k]]))
        
        last_LED_t_BORIS = LED_on_time_BORIS[-1]
        real_time_range = self.time[-1] - self.time[0]
        factor = 1.034141
        shift = last_LED_t_BORIS - real_time_range * factor
        self.start_s = (self.start_s - shift) / factor
        self.stop_s = (self.stop_s - shift) / factor


def correct_chasing_events(
    category: np.ndarray, 
    timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

    onset_ids = np.arange(
        len(category))[category == 0]
    offset_ids = np.arange(
        len(category))[category == 1]

    # Check whether on- or offset is longer and calculate length difference
    if len(onset_ids) > len(offset_ids):
        len_diff = len(onset_ids) - len(offset_ids)
        longer_array = onset_ids
        shorter_array = offset_ids
        logger.info(f'Onsets are greater than offsets by {len_diff}')
    elif len(onset_ids) < len(offset_ids):
        len_diff = len(offset_ids) - len(onset_ids)
        longer_array = offset_ids
        shorter_array = onset_ids
        logger.info(f'Offsets are greater than offsets by {len_diff}')
    elif len(onset_ids) == len(offset_ids):
        logger.info('Chasing events are equal')
        return category, timestamps


    # Correct the wrong chasing events; delete double events
    wrong_ids = []
    for i in range(len(longer_array)-(len_diff+1)):
        if (shorter_array[i] > longer_array[i]) & (shorter_array[i] < longer_array[i+1]):
            pass
        else:
            wrong_ids.append(longer_array[i])
            longer_array = np.delete(longer_array, i)
        
    category = np.delete(
        category, wrong_ids)
    timestamps = np.delete(
        timestamps, wrong_ids)
    return category, timestamps



def main(datapath: str):
    # behabvior is pandas dataframe with all the data
    bh = Behavior(datapath)
    # chirps are not sorted in time (presumably due to prior groupings)
    # get and sort chirps and corresponding fish_ids of the chirps
    chirps = bh.chirps[np.argsort(bh.chirps)]
    chirps_fish_ids = bh.chirps_ids[np.argsort(bh.chirps)]
    category = bh.behavior
    timestamps = bh.start_s
    # Correct for doubles in chasing on- and offsets to get the right on-/offset pairs
    # Get rid of tracking faults (two onsets or two offsets after another)
    category, timestamps = correct_chasing_events(category, timestamps)

    

    
    pass




if __name__ == '__main__':

    # Path to the data
    datapath = '../data/mount_data/2020-05-13-10_00/'
    main(datapath)