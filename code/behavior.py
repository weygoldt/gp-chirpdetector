from pathlib import Path
from pandas import read_csv


class Behavior:
    def __init__(self, datapath: str) -> None:
        csv_file = str(sorted(Path(datapath).glob('**/*.csv'))[0])
        self.dataframe = read_csv(csv_file, delimiter=',')


def main(datapath: str):
    # behabvior is pandas dataframe with all the data
    behavior = Behavior(datapath)


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/2020-03-13-10_00/'
    main(datapath)
