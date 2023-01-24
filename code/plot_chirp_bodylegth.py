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

    woring_bh = np.arange(len(category))[category!=2][:-1][np.diff(category[category!=2])==0]
    if onset_ids[0] > offset_ids[0]:
        offset_ids = np.delete(offset_ids, 0)
        help_index = offset_ids[0]
        woring_bh = np.append(woring_bh, help_index)

    category = np.delete(category, woring_bh)
    timestamps = np.delete(timestamps, woring_bh)

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



def main(datapath: str):

    foldernames = [datapath + x + '/'  for x in os.listdir(datapath) if os.path.isdir(datapath+x)]
    path_to_csv = ('/').join(foldernames[0].split('/')[:-2]) + '/order_meta.csv'
    meta_id = read_csv(path_to_csv)
    meta_id['recording'] = meta_id['recording'].str[1:-1]
    
    chirps_winner = []
    chirps_loser = []

    for foldername in foldernames:
        # behabvior is pandas dataframe with all the data
        if foldername == '../data/mount_data/2020-05-12-10_00/':
            continue
        bh = Behavior(foldername)
        # chirps are not sorted in time (presumably due to prior groupings)
        # get and sort chirps and corresponding fish_ids of the chirps
        category = bh.behavior
        timestamps = bh.start_s
        # Correct for doubles in chasing on- and offsets to get the right on-/offset pairs
        # Get rid of tracking faults (two onsets or two offsets after another)
        category, timestamps = correct_chasing_events(category, timestamps)

        folder_name = foldername.split('/')[-2]
        winner_row = meta_id[meta_id['recording'] == folder_name]
        winner = winner_row['winner'].values[0].astype(int)
        winner_fish1 = winner_row['fish1'].values[0].astype(int)
        winner_fish2 = winner_row['fish2'].values[0].astype(int)
        if winner == winner_fish1:
            winner_fish_id = winner_row['rec_id1'].values[0]
            loser_fish_id = winner_row['rec_id2'].values[0]
        elif winner == winner_fish2:
            winner_fish_id = winner_row['rec_id2'].values[0]
            loser_fish_id = winner_row['rec_id1'].values[0]
        else:
            continue

        print(foldername)
        all_fish_ids = np.unique(bh.chirps_ids)
        chirp_loser = len(bh.chirps[bh.chirps_ids == loser_fish_id])
        chirp_winner = len(bh.chirps[bh.chirps_ids == winner_fish_id])
        chirps_winner.append(chirp_winner)
        chirps_loser.append(chirp_loser)


        fish1_id = all_fish_ids[0]
        fish2_id = all_fish_ids[1]
        print(winner_fish_id)
        print(all_fish_ids)


    fig, ax = plt.subplots()
    ax.boxplot([chirps_winner, chirps_loser])
    
    ax.set_xticklabels(['winner', 'loser'])
    ax.set_ylabel('Chirpscount per trial')
    plt.show()

    

    embed()
    exit()



if __name__ == '__main__':

    # Path to the data
    datapath = '../data/mount_data/'
    
    main(datapath)