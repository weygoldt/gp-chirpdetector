import os
import pandas as pd
import numpy as np
from chirpdetection import chirpdetection
from IPython import embed

# check rec ../data/mount_data/2020-03-25-10_00/ starting at 3175


def get_valid_datasets(dataroot):
    datasets = sorted(
        [
            name
            for name in os.listdir(dataroot)
            if os.path.isdir(os.path.join(dataroot, name))
        ]
    )

    valid_datasets = []
    for dataset in datasets:
        path = os.path.join(dataroot, dataset)
        csv_name = "-".join(dataset.split("-")[:3]) + ".csv"

        if os.path.exists(os.path.join(path, csv_name)) is False:
            continue

        if os.path.exists(os.path.join(path, "ident_v.npy")) is False:
            continue

        ident = np.load(os.path.join(path, "ident_v.npy"))
        number_of_fish = len(np.unique(ident[~np.isnan(ident)]))
        if number_of_fish != 2:
            continue

        valid_datasets.append(dataset)

    datapaths = [
        os.path.join(dataroot, dataset) + "/" for dataset in valid_datasets
    ]

    return datapaths, valid_datasets


def main(datapaths):
    for path in datapaths:
        chirpdetection(path, plot="show")


if __name__ == "__main__":
    dataroot = "../data/mount_data/"

    datapaths, valid_datasets = get_valid_datasets(dataroot)

    recs = pd.DataFrame(columns=["recording"], data=valid_datasets)
    recs.to_csv("../recs.csv", index=False)
    # datapaths = ['../data/mount_data/2020-03-25-10_00/']
    main(datapaths)

# window 1524 + 244 in dataset index 4 is nice example
