from pathlib import Path
import os 
import numpy as np
from IPython import embed

from pandas import read_csv



class Behavior:
    def __init__(self, datapath: str):
        csv_file = str(sorted(Path(datapath).glob('**/*.csv'))[0])
        self.dataframe = read_csv(csv_file, delimiter=',')

        embed()


def main(datapath:str):
    behavior = Behavior(datapath)


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-03-13-10_00/'
    main(datapath)
