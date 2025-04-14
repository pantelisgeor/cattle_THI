import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import glob
import xarray as xr
import numpy as np
import gc
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import argparse


def getDat(model) -> pd.DataFrame:
    # Construct the path to the directory where data is stored
    path_ = f"data/{model}"
    # List the netcdf files in the directory
    files_ = glob.glob1(path_, "*.nc")
    # List the year and scenarios for each
    if model == "ERA5":
        scenarios_ = ["historical" for x in files_]
        years_ = [int(x.split("_")[1].split(".")[0]) for x in files_]
    else:
        years_ = [int(x.split("_")[0]) for x in files_]
        scenarios_ = [x.split("_")[1].split(".")[0] for x in files_]
    paths_ = [f"{path_}/{x}" for x in files_]
    df_ = pd.DataFrame(
        {"model": model, "year": years_, "scenario": scenarios_, "path": paths_}
    )
    return df_


def combModel(model, dfDat, pathSave="combined"):
    print(model)
    if os.path.isfile(f"{pathSave}/{model}.nc"):
        return None
    dat_ = dfDat.loc[dfDat.model == model].reset_index(drop=True)
    ds = xr.Dataset()
    for i in tqdm(range(dat_.shape[0])):
        dat = dat_.iloc[i, :]
        ds_ = xr.open_dataset(dat.path)
        ds_ = ds_.expand_dims(model=[dat.model], scenario=[dat.scenario])
        ds = ds.merge(ds_)
        del i, dat, ds_
        gc.collect()
    ds.to_netcdf(
        f"{pathSave}/{model}.nc",
        encoding={
            "num_occ": {
                "dtype": np.int16,
                "zlib": True,
                "complevel": 6,
                "fletcher32": True,
                "_FillValue": -999,
            },
            "max_dur": {
                "dtype": np.int16,
                "zlib": True,
                "complevel": 6,
                "fletcher32": True,
                "_FillValue": -999,
            },
            "event_hrs": {
                "dtype": np.int16,
                "zlib": True,
                "complevel": 6,
                "fletcher32": True,
                "_FillValue": -999,
            },
            "mean_dur": {
                "dtype": np.float32,
                "zlib": True,
                "complevel": 6,
                "fletcher32": True,
                "_FillValue": -999,
            },
        },
    )
    return None


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="THI load.")
    parser.add_argument("--pathDat", type=str, help="Path to directory where THI data is stored.")
    parser.add_argument("--pathLSM", type=str, help="Path to directory where LSM data is stored (CMIP6).")
    args = parser.parse_args()
    # Paths
    pathDat = args.pathDat
    os.chdir(pathDat)

    # List the datasets in data directory
    models_ = [x.split("/")[1] for x in glob.glob("data/*/", recursive=True)]
    
    # List all the datasets
    dfDat = pd.concat([getDat(x) for x in models_])
    dfDat = dfDat.sort_values(by=["model", "year", "scenario"]).reset_index(drop=True)

    process_map(partial(combModel, dfDat=dfDat), models_, max_workers=3, chunksize=1)

    # And combine them all in one (except the ERA5 dataset)
    if not os.path.isfile("heat_waves.nc"):
        ds = xr.Dataset()
        for mod_ in tqdm(models_):
            if mod_ == "ERA5":
                continue
            ds = ds.merge(xr.open_dataset(f"combined/{mod_}.nc"))

        # Read the land sea mask
        lsm = xr.open_dataset(args.pathLSM)
        lsm = xr.where(lsm > 0, 1, np.nan)
        # Merge with ds to apply it
        ds = ds.fillna(0)
        ds = ds.merge(lsm)
        ds["num_occ"] = ds["num_occ"] * ds["lsm"]
        ds["mean_dur"] = ds["mean_dur"] * ds["lsm"]
        ds["event_hrs"] = ds["event_hrs"] * ds["lsm"]
        ds["max_dur"] = ds["max_dur"] * ds["lsm"]
        # FInally drop the land sea mask variable
        ds = ds.drop_vars("lsm")

        # Save it
        ds.to_netcdf(
            "heat_waves.nc",
            encoding={
                "num_occ": {
                    "dtype": np.int16,
                    "zlib": True,
                    "complevel": 6,
                    "fletcher32": True,
                    "_FillValue": -999,
                },
                "max_dur": {
                    "dtype": np.int16,
                    "zlib": True,
                    "complevel": 6,
                    "fletcher32": True,
                    "_FillValue": -999,
                },
                "event_hrs": {
                    "dtype": np.int16,
                    "zlib": True,
                    "complevel": 6,
                    "fletcher32": True,
                    "_FillValue": -999,
                },
                "mean_dur": {
                    "dtype": np.float32,
                    "zlib": True,
                    "complevel": 6,
                    "fletcher32": True,
                    "_FillValue": -999,
                },
            },
        )
