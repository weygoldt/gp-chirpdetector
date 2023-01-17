from pathlib import Path

import numpy as np
from IPython import embed
from pandas import read_csv




class Behavior:
    """Load behavior data from csv file as class attributes
        Attributes
    ----------
    behavior_type:         
    behavioral_category:   
    comment_start:         
    comment_stop:          
    dataframe: pandas dataframe with all the data            
    duration_s:             
    media_file:            
    observation_date:      
    observation_id:        
    start_s:               
    stop_s:                
    total_length:          
    """

    def __init__(self, datapath: str) -> None:
        csv_file = str(sorted(Path(datapath).glob('**/*.csv'))[0])
        self.dataframe = read_csv(csv_file, delimiter=',')
        for key in self.dataframe:
            if ' ' in key:
                key = key.replace(' ', '_')
                if '(' in key:
                    key = key.replace('(', '')
                    key = key.replace(')', '')
            key = key.lower()  
            setattr(self, key, np.array(self.dataframe[key]))
        
        embed()
        


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
    # behabvior is pandas dataframe with all the data
    behavior = Behavior(datapath)


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-03-13-10_00/'
    main(datapath)
