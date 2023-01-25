import numpy as np

import os 

import numpy as np
from IPython import embed


from pandas import read_csv
from modules.logger import makeLogger
from modules.datahandling import causal_kde1d, acausal_kde1d


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
        self.chirps_ids = np.load(os.path.join(folder_path, 'chirp_ids.npy'), allow_pickle=True)

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

    wrong_bh = np.arange(len(category))[category!=2][:-1][np.diff(category[category!=2])==0]
    if onset_ids[0] > offset_ids[0]:
        offset_ids = np.delete(offset_ids, 0)
        help_index = offset_ids[0]
        wrong_bh = np.append(wrong_bh[help_index])

    category = np.delete(category, wrong_bh)
    timestamps = np.delete(timestamps, wrong_bh)

    # Check whether on- or offset is longer and calculate length difference
    if len(onset_ids) > len(offset_ids):
        len_diff = len(onset_ids) - len(offset_ids)
        logger.info(f'Onsets are greater than offsets by {len_diff}')
    elif len(onset_ids) < len(offset_ids):
        len_diff = len(offset_ids) - len(onset_ids)
        logger.info(f'Offsets are greater than onsets by {len_diff}')
    elif len(onset_ids) == len(offset_ids):
        logger.info('Chasing events are equal')
    
    return category, timestamps


def event_triggered_chirps(
    event: np.ndarray, 
    chirps:np.ndarray,
    time_before_event: int,
    time_after_event: int,
    dt: float,
    width: float,
    )-> tuple[np.ndarray, np.ndarray, np.ndarray]:

    event_chirps = []   # chirps that are in specified window around event
    centered_chirps = []    # timestamps of chirps around event centered on the event timepoint

    for event_timestamp in event:
        start = event_timestamp - time_before_event
        stop = event_timestamp + time_after_event
        chirps_around_event = [c for c in chirps if (c >= start) & (c <= stop)]
        event_chirps.append(chirps_around_event)
        if len(chirps_around_event) == 0:
            continue
        else: 
            centered_chirps.append(chirps_around_event - event_timestamp)
    
    time = np.arange(-time_before_event, time_after_event, dt)
    
    # Kernel density estimation with some if's
    if len(centered_chirps) == 0:
        centered_chirps = np.array([])
        centered_chirps_convolved = np.zeros(len(time))
    else:
        centered_chirps = np.concatenate(centered_chirps, axis=0)   # convert list of arrays to one array for plotting
        centered_chirps_convolved = (acausal_kde1d(centered_chirps, time, width)) / len(event)

    return event_chirps, centered_chirps, centered_chirps_convolved

