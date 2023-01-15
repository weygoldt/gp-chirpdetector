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
                new_key = key.replace(' ', '_')
                if '(' in new_key:
                    new_key = new_key.replace('(', '')
                    new_key = new_key.replace(')', '')
            new_key = new_key.lower()  
            setattr(self, new_key, np.array(self.dataframe[key]))


def main(datapath: str):
    # behabvior is pandas dataframe with all the data
    behavior = Behavior(datapath)


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-03-13-10_00/'
    main(datapath)
