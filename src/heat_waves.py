import os
import glob
import gc
import xarray as xr
import pandas as pd
import numpy as np
from functools import partial
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import argparse


# =============================== FUNCTIONS =============================== #
def getDat(path_models, model):
    # Construct the path to the model's data directory
    path_ = f"{path_models}/{model}"
    # List the netcdf files
    files_ = glob.glob1(path_, "*.nc")
    # Put the data in a dataframe
    df = pd.DataFrame({
        "model": model,
        "year": [int(x.split("_")[0]) for x in files_],
        "scenario": [x.split("_")[1].split(".")[0] for x in files_],
        "filename": [f"{path_models}/{model}/{x}" for x in files_]
    })
    return df.sort_values(by=["year", "scenario"], ascending=True)


def listERA(pathERA):
    files_ = glob.glob1(pathERA, "*.nc")
    df = pd.DataFrame({
        "model": "ERA5",
        "year": [int(x.split("_")[2]) for x in files_],
        "filename": [f"{pathERA}/{x}" for x in files_]})
    return df.sort_values(by=["year"]).reset_index(drop=True)


# ========================================================================= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Climate model to process.")
    parser.add_argument("--year", type=int,
                        help="Year to process.")
    parser.add_argument("--scenario", type=str,
                        help="Climate scenario.")
    parser.add_argument("--pathERA", type=str,
                        help="Path to the ERA5 data.")
    parser.add_argument("--pathTarget", type=str,
                        help="Path to where the processed data will be saved.")
    parser.add_argument("--pathDat", type=str, help="Path to the THI hourly data.")
    parser.add_argument("--lsm_CMIP6", type=str, help="Path to the CMIP6 land sea mask.")
    parser.add_argument("--lsm_ERA5", type=str, help="Path to the ERA5 land sea mask.")
    args = parser.parse_args()
    print(f"----\n{args.model} - {args.scenario} - {args.year}\n----")

    path_target = args.pathTarget
    os.chdir(path_target)

    # Path to data with hours above threshold per day
    pathDat = args.pathDat
    # List the subdirectories (model names)
    models = [x.split("/")[-2] for x in glob.glob(f"{pathDat}/*/",
                                                  recursive=True)]
    # CMIP6 land sea mask (derived from ERA5 lsm)
    if args.model == "ERA5":
        lsm = xr.open_dataset(args.lsm_ERA5)\
            .rename({"longitude": "lon", "latitude": "lat"})\
            .squeeze().drop_vars("time")
    else:
        lsm = xr.open_dataset(args.lsm_CMIP6)
    # Make the land sea mask a binary mask
    lsm["lsm"] = xr.where(lsm.lsm > 0, 1, np.nan)

    if args.model == "ERA5":
        # List the data
        dfDat = listERA(pathERA=args.pathERA)
        path_ = dfDat.loc[dfDat.year == args.year].filename.values[0]
        pathSave = "data/ERA5"
        os.makedirs(pathSave, exist_ok=True)
        if os.path.isfile(f"{pathSave}/ERA5_{args.year}.nc"):
            exit()
    else:
        # Get all the data for the models
        dfDat = pd.concat([getDat(path_models=pathDat, model=model) for model
                           in models])

        path_ = dfDat.loc[(dfDat.model == args.model) &
                          (dfDat.scenario == args.scenario) &
                          (dfDat.year == args.year)].filename.values[0]
        # Construct the path to save the processed netcdf and
        # test if it's already there
        pathSave = f"data/{path_.split('/')[-2]}"
        os.makedirs(pathSave, exist_ok=True)
        if os.path.isfile(f"{pathSave}/{path_.split('/')[-1]}"):
            exit()

    #  Read dataset and load it into memory to speed up following calculations
    if args.model == "ERA5":
        ds = xr.open_dataset(path_).drop_vars(["t2m", "rh", "t2mC"])\
            .rename({"longitude": "lon", "latitude": "lat"}).load()
    else:
        ds = xr.open_dataset(path_).load()
    # Convert the land sea mask to dataframe and only keep the land coordiantes
    coords = lsm.to_dataframe().reset_index(drop=False).dropna()\
        .groupby(["lat", "lon"], as_index=False)\
        .size().drop("size", axis=1)

    def getOccurrences(coord_, thr_=74, length_occ=48, ds=ds):
        """
        Approach description:

        - Shift and Compare: We use the .shift() method to compare 
        the current value with the previous one. Whenever there's a change, 
        a new group starts.
        - Grouping: Using .cumsum() with the comparison result creates a unique 
        identifier for each sequence of consecutive '1's or '0's.
        - Filtering: We filter the groups to find those with more than 48 '1's.
        - Calculate Results: Finally, we calculate the number of such 
        occurrences and their mean duration.
        """
        # Subset the xarray for the specified coordinates
        df_ = ds.sel(lon=coord_[1], lat=coord_[0])\
            .to_dataframe()\
            .reset_index(drop=False)
        # Apply the THI threshold
        df_ = df_.assign(THI_=np.where(df_.THI > thr_, 1, 0))
        # Identify where the value changes
        df_['group'] = (df_['THI_'] != df_['THI_'].shift()).cumsum()
        # Count the size of each group
        group_sizes = df_[df_['THI_'] == 1].groupby('group').size()
        # Filter groups with more than 48 consecutive 1s
        long_sequences = group_sizes[group_sizes > length_occ]
        # Number of occurrences
        num_occurrences = long_sequences.count()
        # Mean duration of these occurrences
        mean_duration = long_sequences.mean()
        # Duration of longest occurrence
        max_duration = long_sequences.max()
        # Total duration of events
        event_hrs = long_sequences.sum()
        # Terminate function
        return pd.DataFrame({
            "lat": [coord_[0]], "lon": [coord_[1]],
            "time": [pd.to_datetime(str(df_.time.max()).split(" ")[0])],
            "num_occ": [num_occurrences], "mean_dur": [mean_duration],
            "max_dur": [max_duration], "event_hrs": [event_hrs]})

    # Run the getOccurrences function in parallel
    heatWaves = pd.concat(process_map(getOccurrences, coords.values,
                                      max_workers=128,
                                      chunksize=coords.shape[0]//512))
    heatWaves = heatWaves.reset_index(drop=True)
    # Merge with the land sea mask coordinates and convert it to xarray
    ds_ = lsm.merge(heatWaves.set_index(["lat", "lon", "time"]).to_xarray())\
        .drop_vars("lsm")

    # Save it
    if args.model == 'ERA5':
        ds_.to_netcdf(f"{pathSave}/ERA5_{args.year}.nc",
                      encoding={"num_occ": {"dtype": np.int16,
                                            "zlib": True,
                                            "complevel": 6,
                                            "fletcher32": True,
                                            "_FillValue": -999},
                                "max_dur": {"dtype": np.int16,
                                            "zlib": True,
                                            "complevel": 6,
                                            "fletcher32": True,
                                            "_FillValue": -999},
                                "event_hrs": {"dtype": np.int16,
                                              "zlib": True,
                                              "complevel": 6,
                                              "fletcher32": True,
                                              "_FillValue": -999},
                                "mean_dur": {"dtype": np.float32,
                                             "zlib": True,
                                             "complevel": 6,
                                             "fletcher32": True,
                                             "_FillValue": -999}})
    else:
        ds_.to_netcdf(f"{pathSave}/{path_.split('/')[-1]}",
                      encoding={"num_occ": {"dtype": np.int16,
                                            "zlib": True,
                                            "complevel": 6,
                                            "fletcher32": True,
                                            "_FillValue": -999},
                                "max_dur": {"dtype": np.int16,
                                            "zlib": True,
                                            "complevel": 6,
                                            "fletcher32": True,
                                            "_FillValue": -999},
                                "event_hrs": {"dtype": np.int16,
                                              "zlib": True,
                                              "complevel": 6,
                                              "fletcher32": True,
                                              "_FillValue": -999},
                                "mean_dur": {"dtype": np.float32,
                                             "zlib": True,
                                             "complevel": 6,
                                             "fletcher32": True,
                                             "_FillValue": -999}})
