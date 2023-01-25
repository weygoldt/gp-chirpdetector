import numpy as np

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from thunderfish.powerspectrum import decibel

from IPython import embed
from pandas import read_csv
from modules.logger import makeLogger
from modules.plotstyle import PlotStyle
from modules.behaviour_handling import Behavior, correct_chasing_events
from modules.datahandling import flatten


ps = PlotStyle()

logger = makeLogger(__name__)


def main(datapath: str):

    foldernames = [
        datapath + x + '/' for x in os.listdir(datapath) if os.path.isdir(datapath+x)]
    time_precents = []
    chirps_percents = []
    for foldername in foldernames:
        # behabvior is pandas dataframe with all the data
        if foldername == '../data/mount_data/2020-05-12-10_00/':
            continue
        bh = Behavior(foldername)

        category = bh.behavior
        timestamps = bh.start_s
        # Correct for doubles in chasing on- and offsets to get the right on-/offset pairs
        # Get rid of tracking faults (two onsets or two offsets after another)
        category, timestamps = correct_chasing_events(category, timestamps)

        chasing_onset = timestamps[category == 0]
        chasing_offset = timestamps[category == 1]
        if len(chasing_onset) != len(chasing_offset):
            embed()

        chirps_in_chasings = []
        for onset, offset in zip(chasing_onset, chasing_offset):
            chirps_in_chasing = [c for c in bh.chirps if (c > onset) & (c < offset)]
            chirps_in_chasings.append(chirps_in_chasing)

        try:
            time_chasing = np.sum(chasing_offset[chasing_offset<3*60*60] - chasing_onset[chasing_onset<3*60*60])
        except:
            time_chasing = np.sum(chasing_offset[chasing_offset<3*60*60] - chasing_onset[chasing_onset<3*60*60][:-1])


        time_chasing_percent = (time_chasing/(3*60*60))*100
        chirps_chasing = np.asarray(flatten(chirps_in_chasings))
        chirps_chasing_new = chirps_chasing[chirps_chasing<3*60*60]
        chirps_percent = (len(chirps_chasing_new)/len(bh.chirps))*100

        time_precents.append(time_chasing_percent)
        chirps_percents.append(chirps_percent)
    
    fig, ax = plt.subplots(1, 1, figsize=(14*ps.cm, 10*ps.cm))

    ax.boxplot([time_precents, chirps_percents])
    ax.set_xticklabels(['Time Chasing', 'Chirps in Chasing'])
    ax.set_ylabel('Percent')
    ax.scatter(np.ones(len(time_precents))*1.25, time_precents, color=ps.white)
    ax.scatter(np.ones(len(chirps_percents))*1.75, chirps_percents, color=ps.white)
    plt.savefig('../poster/figs/chirps_in_chasing.pdf')
    plt.show()


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/'
    main(datapath)

        
