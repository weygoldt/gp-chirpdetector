import os 

import numpy as np
import matplotlib.pyplot as plt 

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
        self.time = np.load(os.path.join(folder_path, "times.npy"), allow_pickle=True)
        csv_filename = [f for f in os.listdir(folder_path) if f.endswith('.csv')][0] # check if there are more than one csv file
        self.dataframe = read_csv(os.path.join(folder_path, csv_filename))
        self.chirps = np.load(os.path.join(folder_path, 'chirps.npy'), allow_pickle=True)
        self.chirps_ids = np.load(os.path.join(folder_path, 'chirps_ids.npy'), allow_pickle=True)

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
  

"""
1 - chasing onset
2 - chasing offset
3 - physical contact event

temporal encpding needs to be corrected ... not exactly 25FPS.

### correspinding python code ###

    factor = 1.034141
    LED_on_time_BORIS = np.load(os.path.join(folder_path, 'LED_on_time.npy'), allow_pickle=True)
    last_LED_t_BORIS = LED_on_time_BORIS[-1]
    real_time_range = times[-1] - times[0]
    shift = last_LED_t_BORIS - real_time_range * factor

    data = pd.read_csv(os.path.join(folder_path, file[1:-7] + '.csv'))
    boris_times = data['Start (s)']
    data_times = []

    for Cevent_t in boris_times:
        Cevent_boris_times = (Cevent_t - shift) / factor
        data_times.append(Cevent_boris_times)

    data_times = np.array(data_times)
    behavior = data['Behavior']
"""


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
    centered_chirps = np.concatenate(centered_chirps, axis=0)   # convert list of arrays to one array for plotting

    # Kernel density estimation
    time = np.arange(-time_before_event, time_after_event, dt)
    centered_chirps_convolved = (acausal_kde1d(centered_chirps, time, width)) / len(event)

    return event_chirps, centered_chirps, centered_chirps_convolved


def main(datapath: str):

    # behavior is pandas dataframe with all the data
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

    # split categories
    chasing_onsets = timestamps[category == 0]
    chasing_offsets = timestamps[category == 1]
    physical_contacts = timestamps[category == 2]

    chasing_durations = []
    # Calculate chasing duration to evaluate a nice time window for kernel density estimation
    for onset, offset in zip(chasing_onsets, chasing_offsets):
        duration = offset - onset
        chasing_durations.append(duration)

    fig, ax = plt.subplots()
    ax.boxplot(chasing_durations)
    plt.show()
    plt.close()

    # Get fish ids
    fish_ids = np.unique(chirps_fish_ids)

    # # Associate chirps to individual fish
    # fish1 = chirps[chirps_fish_ids == fish_ids[0]]
    # fish2 = chirps[chirps_fish_ids == fish_ids[1]]
    # fish = [len(fish1), len(fish2)]

    # Define time window for chirp around event analysis
    time_before_event = 30
    time_after_event = 60
    dt = 0.01
    width = 1

    #### Loop crashes at concatenate in function ####
    for i in range(len(fish_ids)):
        fish = fish_ids[i]
        chirps_temp = chirps[chirps_fish_ids == fish]
        print(fish)

        ##### Chirps around events #####
        time = np.arange(-time_before_event, time_after_event, dt)

        # Chirps around chasing onsets
        _, centered_chasing_onset_chirps, cc_chasing_onset_chirps = event_triggered_chirps(chasing_onsets, chirps_temp, time_before_event, time_after_event, dt, width)
        # Chirps around chasing offsets
        _, centered_chasing_offset_chirps, cc_chasing_offset_chirps = event_triggered_chirps(chasing_offsets, chirps_temp, time_before_event, time_after_event, dt, width)
        # Chirps around physical contacts
        _, centered_physical_chirps, cc_physical_chirps = event_triggered_chirps(physical_contacts, chirps_temp, time_before_event, time_after_event, dt, width)

        fig, ax = plt.subplots(1, 3, figsize=(50 / 2.54, 15 / 2.54), constrained_layout=True, sharey='all')
        offset = [0.25]
        ax[0].set_xlabel('Time[s]')
        # Plot chasing onsets
        ax[0].set_ylabel('Chirp rate [Hz]')
        ax[0].plot(time, cc_chasing_onset_chirps, color='tab:blue')
        ax0 = ax[0].twinx()
        ax0.eventplot(np.array([centered_chasing_onset_chirps]), lineoffsets=offset, linelengths=0.1, colors=['tab:green'])
        ax0.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
        ax0.set_yticklabels([])
        ax0.set_yticks([])
        # Plot chasing offets
        ax[1].set_xlabel('Time[s]')
        ax[1].plot(time, cc_chasing_offset_chirps, color='tab:blue')
        ax1 = ax[1].twinx()
        ax1.eventplot(np.array([centered_chasing_offset_chirps]), lineoffsets=offset, linelengths=0.1, colors=['tab:purple'])
        ax1.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        # Plot physical contacts
        ax[2].set_xlabel('Time[s]')
        ax[2].plot(time, cc_physical_chirps, color='tab:blue')
        ax2 = ax[2].twinx()
        ax2.eventplot(np.array([centered_physical_chirps]), lineoffsets=offset, linelengths=0.1, colors=['tab:red'])
        ax2.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        plt.show()

    ### Plots:
    # 1. All recordings, all fish, all chirps
        # One CTC, one PTC
    # 2. All recordings, only winners
        # One CTC, one PTC
    # 3. All recordings, all losers
        # One CTC, one PTC

    
    embed()
    exit()



if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-05-13-10_00/'
    main(datapath)
