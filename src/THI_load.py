import os
import glob
import pandas as pd
import xarray as xr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import gc
import argparse


# =============================== FUNCTIONS =============================== #
def getDat(path_models, model):
    # Construct the path to the model's data directory
    path_ = f"{path_models}/{model}"
    # List the netcdf files
    files_ = glob.glob1(path_, "*.nc")
    # Put the data in a dataframe
    df = pd.DataFrame(
        {
            "model": model,
            "year": [int(x.split("_")[0]) for x in files_],
            "scenario": [x.split("_")[1].split(".")[0] for x in files_],
            "filename": [f"{path_models}/{model}/{x}" for x in files_],
        }
    )
    return df.sort_values(by=["year", "scenario"], ascending=True)


def convert360_180(_ds):
    """
    convert longitude from 0-360 to -180 -- 180 deg
    """
    # check if already
    attrs = _ds["lon"].attrs
    if _ds["lon"].min() >= 0:
        with xr.set_options(keep_attrs=True):
            _ds.coords["lon"] = (_ds["lon"] + 180) % 360 - 180
        _ds = _ds.sortby("lon")
    return _ds


def getArea(df, model, year, scenario, thr_, convert_lon=False):
    """
    Calculates the total THI load per day (THI above threshold)
    """
    # Read the netcdf file
    ds = xr.open_dataset(
        df.loc[
            (df.model == model) & (df.scenario == scenario) & (df.year == year)
        ].filename.item()
    )
    # Subtract the thr_ value from the THI variable
    ds["THI"] = ds.THI - thr_
    # Set all values below 0 to 0
    ds_ = xr.where(ds.THI < 0, 0, ds.THI)
    # Calculate the sum per day
    ds_ = ds_.resample(time="1D").sum()
    gc.collect()
    ds_ = ds_.to_dataset(name="THI")
    del ds
    # Adjust lon values to make sure they are within (-180, 180)
    if convert_lon:
        ds_ = convert360_180(ds_)
    # Add the model name and scenario as dimensions
    ds_ = ds_.expand_dims(model=[model], scenario=[scenario], threshold=[thr_])
    return ds_


def thrAll(model, scenario, df, path_save_):
    """
    Calculate the area under the curve of the THI values for each day,
    that is the total THI units above a threshold for each day.
    """
    print(model)
    if not os.path.isdir(path_save_):
        os.mkdir(path_save_)
    path_save_ = f"{path_save_}/{model}"
    if not os.path.isdir(path_save_):
        os.mkdir(path_save_)
    for year in tqdm(range(2020, 2101, 1)):
        if os.path.isfile(f"{path_save_}/{scenario}_{year}.nc"):
            print("Skipping. . .")
            continue
        print("Year: {} - model: {}".format(year, model))
        for thr_ in [68.8, 84]:
            if thr_ == 68.8:
                ds = getArea(
                    df=df, model=model, year=year, scenario=scenario, thr_=thr_
                )
            else:
                ds = ds.merge(
                    getArea(df=df, model=model, year=year, scenario=scenario, thr_=thr_)
                )
        # Save it
        ds.to_netcdf(
            f"{path_save_}/{scenario}_{year}.nc",
            encoding={
                "THI": {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 6,
                    "fletcher32": True,
                    "_FillValue": -999.0,
                }
            },
        )
        del ds
        gc.collect()


# ========================================================================= #
if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description="THI load calculation")
    parser.add_argument("--pathDat", help="Path to the data directory", type=str)
    parser.add_argument("--pathTarget", type=str, help="Path to the target directory")
    args = parser.parse_args()

    # Paths
    path_save = args.pathTarget
    os.makedirs(path_save, exist_ok=True)
    path_models = args.pathDat
    os.chdir(path_models)

    # List the subdirectories (model names)
    models = [x.split("/")[-2] for x in glob.glob(f"{path_models}/*/", recursive=True)]

    # Get all the data
    df = pd.concat([getDat(path_models, model) for model in models])
    df = df.reset_index(drop=True)

    # Process all the datasets
    for scenario in df.scenario.unique():
        for model in df.model.unique():
            thrAll(model, scenario, df, path_save_=path_save)
