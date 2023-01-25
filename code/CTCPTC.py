import os 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from tqdm import tqdm
from IPython import embed
from pandas import read_csv
from modules.logger import makeLogger
from modules.plotstyle import PlotStyle
from modules.datahandling import flatten
from modules.behaviour_handling import Behavior, correct_chasing_events, event_triggered_chirps

logger = makeLogger(__name__)
ps = PlotStyle()

#### Goal: CTC & PTC for each winner and loser and for all winners and loser ####
