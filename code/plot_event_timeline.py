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

    # split categories
    chasing_onset = (timestamps[category == 0]/ 60) /60
    chasing_offset = (timestamps[category == 1]/ 60) /60
    physical_contact = (timestamps[category == 2] / 60) /60

    all_fish_ids = np.unique(chirps_fish_ids)
    fish1_id = all_fish_ids[0]
    fish2_id = all_fish_ids[1]
    # Associate chirps to inidividual fish
    fish1 = (chirps[chirps_fish_ids == fish1_id] / 60) /60
    fish2 = (chirps[chirps_fish_ids == fish2_id] / 60) /60
    fish1_color = ps.red
    fish2_color = ps.orange

    fig, ax = plt.subplots(4, 1, figsize=(10, 5), height_ratios=[0.5, 0.5, 0.5, 6], sharex=True)
    # marker size 
    s = 200
    ax[0].scatter(physical_contact, np.ones(len(physical_contact)), color='firebrick', marker='|', s=s)
    ax[1].scatter(chasing_onset, np.ones(len(chasing_onset)), color='green', marker='|', s=s )
    ax[2].scatter(fish1, np.ones(len(fish1))-0.25, color=fish1_color, marker='|', s=s)
    ax[2].scatter(fish2, np.zeros(len(fish2))+0.25, color=fish2_color, marker='|', s=s)
   

    freq_temp = bh.freq[bh.ident==fish1_id]
    time_temp = bh.time[bh.idx[bh.ident==fish1_id]]
    ax[3].plot((time_temp/ 60) /60, freq_temp, color=fish1_color)

    freq_temp = bh.freq[bh.ident==fish2_id]
    time_temp = bh.time[bh.idx[bh.ident==fish2_id]]
    ax[3].plot((time_temp/ 60) /60, freq_temp, color=fish2_color)

    #ax[3].imshow(decibel(bh.spec), extent=[bh.time[0]/60/60, bh.time[-1]/60/60, 0, 2000], aspect='auto', origin='lower')

        # Hide grid lines
    ax[0].grid(False)
    ax[0].set_frame_on(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ps.hide_ax(ax[0])


    ax[1].grid(False)
    ax[1].set_frame_on(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ps.hide_ax(ax[1])

    ax[2].grid(False)
    ax[2].set_frame_on(False)
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    ps.hide_ax(ax[2])



    ax[3].axvspan(0, 3, 0, 5, facecolor='grey', alpha=0.5)
    ax[3].set_xticks(np.arange(0, 6.1, 0.5))

    labelpad = 40
    ax[0].set_ylabel('Physical contact', rotation=0, labelpad=labelpad)
    ax[1].set_ylabel('Chasing events', rotation=0, labelpad=labelpad)
    ax[2].set_ylabel('Chirps', rotation=0, labelpad=labelpad)
    ax[3].set_ylabel('EODf')

    ax[3].set_xlabel('Time [h]')

    plt.show()
    embed()

    # plot chirps


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-05-13-10_00/'
    main(datapath)
