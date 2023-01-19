import os 

import numpy as np
from IPython import embed
from pandas import read_csv
import matplotlib.pyplot as plt 




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

def main(datapath: str):

    # behavior is pandas dataframe with all the data
    bh = Behavior(datapath)
    
    # chirps are not sorted in time (presumably due to prior groupings)
    # get and sort chirps and corresponding fish_ids of the chirps
    chirps = bh.chirps[np.argsort(bh.chirps)]
    chirps_ids = bh.chirps_ids[np.argsort(bh.chirps)]
    category = bh.behavior
    timestamps = bh.start_s

    # split categories
    chasing_onset = timestamps[category == 0]
    chasing_offset = timestamps[category == 1]
    physical_contact = timestamps[category == 2]

    # Physical contact-triggered chirps (PTC) mit Rasterplot
    # Wahrscheinlichkeit von Phys auf Ch und vice versa
    # Chasing-triggered chirps (CTC) mit Rasterplot
    # Wahrscheinlichkeit von Chase auf Ch und vice versa

    # First overview plot
    fig, ax = plt.subplots()
    ax.scatter(chirps, np.ones_like(chirps), marker='*', color='royalblue', label='Chirps')
    ax.scatter(chasing_onset, np.ones_like(chasing_onset)*2, marker='.', color='forestgreen', label='Chasing onset')
    ax.scatter(chasing_offset, np.ones_like(chasing_offset)*2.5, marker='.', color='firebrick', label='Chasing offset')
    ax.scatter(physical_contact, np.ones_like(physical_contact)*3, marker='x', color='black', label='Physical contact')
    plt.legend()
    plt.show()

    embed()
    exit()


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-05-13-10_00/'
    main(datapath)
