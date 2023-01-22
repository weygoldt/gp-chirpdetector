import os 

import numpy as np
import matplotlib.pyplot as plt 

from IPython import embed
from pandas import read_csv
from modules.logger import makeLogger
from scipy.ndimage import gaussian_filter1d

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
    time_after_event: int
    )-> tuple[np.ndarray, np.ndarray]:



    event_chirps = []   # chirps that are in specified window around event
    centered_chirps = []    # timestamps of chirps around event centered on the event timepoint

    for event_timestamp in event:
        start = event_timestamp - time_before_event    # timepoint of window start
        stop = event_timestamp + time_after_event    # timepoint of window ending
        chirps_around_event = [c for c in chirps if (c >= start) & (c <= stop)]     # get chirps that are in a -5 to +5 sec window around event
        event_chirps.append(chirps_around_event)
        if len(chirps_around_event) == 0:
            continue
        else: 
            centered_chirps.append(chirps_around_event - event_timestamp)
    centered_chirps = np.concatenate(centered_chirps, axis=0)   # convert list of arrays to one array for plotting

    return event_chirps, centered_chirps


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
    chasing_onset = timestamps[category == 0]
    chasing_offset = timestamps[category == 1]
    physical_contact = timestamps[category == 2]

    # First overview plot
    fig1, ax1 = plt.subplots()
    ax1.scatter(chirps, np.ones_like(chirps), marker='*', color='royalblue', label='Chirps')
    ax1.scatter(chasing_onset, np.ones_like(chasing_onset)*2, marker='.', color='forestgreen', label='Chasing onset')
    ax1.scatter(chasing_offset, np.ones_like(chasing_offset)*2.5, marker='.', color='firebrick', label='Chasing offset')
    ax1.scatter(physical_contact, np.ones_like(physical_contact)*3, marker='x', color='black', label='Physical contact')
    plt.legend()
    # plt.show()
    plt.close()

    # Get fish ids
    fish_ids = np.unique(chirps_fish_ids)

    ##### Chasing triggered chirps CTC #####
    # Evaluate how many chirps were emitted in specific time window around the chasing onset events

    # Iterate over chasing onsets (later over fish)
    time_around_event = 5    # time window around the event in which chirps are counted, 5 = -5 to +5 sec around event

    #### Loop crashes at concatenate in function ####
    for i in range(len(fish_ids)):
        fish = fish_ids[i]
        chirps = chirps[chirps_fish_ids == fish]
        print(fish)

        chasing_chirps, centered_chasing_chirps = event_triggered_chirps(chasing_onset, chirps, time_around_event, time_around_event)
        physical_chirps, centered_physical_chirps = event_triggered_chirps(physical_contact, chirps, time_around_event, time_around_event)

        # Kernel density estimation ???
        # centered_chasing_chirps_convolved = gaussian_filter1d(centered_chasing_chirps, 5)
        
        # centered_chasing = chasing_onset[0] - chasing_onset[0]   ## get the 0 timepoint for plotting; set one chasing event to 0
        offsets = [0.5, 1]
        fig4, ax4 = plt.subplots(figsize=(20 / 2.54, 12 / 2.54), constrained_layout=True)
        ax4.eventplot(np.array([centered_chasing_chirps, centered_physical_chirps]), lineoffsets=offsets, linelengths=0.25, colors=['g', 'r'])
        ax4.vlines(0, 0, 1.5, 'tab:grey', 'dashed', 'Timepoint of event')
        # ax4.plot(centered_chasing_chirps_convolved)
        ax4.set_yticks(offsets)
        ax4.set_yticklabels(['Chasings', 'Physical \n contacts'])
        ax4.set_xlabel('Time[s]')
        ax4.set_ylabel('Type of event')
        plt.show()

    # Associate chirps to inidividual fish
    fish1 = chirps[chirps_fish_ids == fish_ids[0]]
    fish2 = chirps[chirps_fish_ids == fish_ids[1]]
    fish = [len(fish1), len(fish2)]

    #### Chirp counts per fish general #####
    fig2, ax2 = plt.subplots()
    x = ['Fish1', 'Fish2']
    width = 0.35
    ax2.bar(x, fish, width=width)
    ax2.set_ylabel('Chirp count')
    # plt.show()
    plt.close()

 
    ##### Count chirps emitted during chasing events and chirps emitted out of chasing events #####
    chirps_in_chasings = []
    for onset, offset in zip(chasing_onset, chasing_offset):
        chirps_in_chasing = [c for c in chirps if (c > onset) & (c < offset)]
        chirps_in_chasings.append(chirps_in_chasing)

    # chirps out of chasing events
    counts_chirps_chasings = 0
    chasings_without_chirps = 0
    for i in chirps_in_chasings:
        if i:
            chasings_without_chirps += 1
        else:
            counts_chirps_chasings += 1

    # chirps in chasing events
    fig3 , ax3 = plt.subplots()
    ax3.bar(['Chirps in chasing events',  'Chasing events without Chirps'], [counts_chirps_chasings, chasings_without_chirps], width=width)
    plt.ylabel('Count')
    # plt.show()
    plt.close()  

    # comparison between chasing events with and without chirps


    
    embed()
    exit()



if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-05-13-10_00/'
    main(datapath)
