"""
Calculate the number of hours above the defined threshold for each day/
grid cell for a given model/scenario combination. Do the same for all
the years available and save it. The netcdf is saved in int16 type and
compressed to save space.
"""

import os
import glob
import pandas as pd
import xarray as xr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import argparse
import gc

# Input argument
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float)
parser.add_argument("--scenario", type=str, choices=["ssp245", "ssp585"])
parser.add_argument(
    "--pathTHI", type=str, help="Path to directory where THI projections are stored."
)
parser.add_argument(
    "--pathTarget", type=str, help="Path to directory where target data is stored."
)
parser.add_argument(
    "--pathERA", type=str, help="Path to directory where ERA5 data is stored."
)
args = parser.parse_args()

path_models = args.pathTHI
path_save = args.pathTarget
os.chdir(path_models)
os.makedirs(path_save, exist_ok=True)

# List the subdirectories (model names)
models = [x.split("/")[-2] for x in glob.glob(f"{path_models}/*/", recursive=True)]


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


def checkMissing(df):
    # List the unique combinations in each model/scenario
    df_ = df.groupby(["model", "scenario"], as_index=False).nunique()
    # Print the ones that are complete (81 entries)
    print("Complete:")
    print(f"{df_.loc[df_.year == 81]}\n")
    # Print the missing years for the rest
    df_ = df_.loc[df_.year < 81].reset_index(drop=True)
    df_missing = pd.DataFrame()
    for i in range(df_.shape[0]):
        mod, scen = df_.iloc[i, :2]
        # Get a list of the unique years in this model/scenario
        years_ = df.loc[(df.model == mod) & (df.scenario == scen)].year.unique()
        # missing years
        years_missing = []
        for yr in range(2020, 2101):
            if yr not in years_:
                years_missing.append(yr)
        df_missing = pd.concat(
            [
                df_missing,
                pd.DataFrame(
                    {"model": mod, "scenario": scen, "years": list(years_missing)}
                ),
            ]
        )
    return df_missing


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


def getThr(df, model, scenario, year, thr_, adjust_lon=False):
    """
    Function to get the number of hours per grid cell per day for a given
    year of a used defined model and scenario
    """
    # Read the netcdf file
    ds = xr.open_dataset(
        df.loc[
            (df.model == model) & (df.scenario == scenario) & (df.year == year)
        ].filename.item()
    )
    # Assign a boolean value based on whether THI is above the
    # specified THI threshold
    ds_ = xr.where(ds.THI >= thr_, 1, 0)
    # Calculate the sum per day
    ds_ = ds_.resample(time="1D").sum()
    gc.collect()
    ds_ = ds_.to_dataset(name="THI")
    del ds
    if adjust_lon:
        # Adjust lon values to make sure they are within (-180, 180)
        ds_ = convert360_180(ds_)
    # Add the model name and scenario as dimensions
    ds_ = ds_.expand_dims(model=[model], scenario=[scenario], threshold=[thr_])
    return ds_


def thrAll(model, scenario, df, thr_, path_save):
    """
    Calculate the number of hours above the defined threshold for each day/
    gric cell for a given model/scenario combination. Do the same for all
    the years available and save it. The netcdf is saved in int16 type and
    compressed to save space.
    """
    # Directory to save data
    path_save_ = f"{path_save}/{str(thr_).replace('.', '')}"
    if not os.path.isdir(path_save_):
        os.mkdir(path_save_)
    path_save_ = f"{path_save_}/{model}"
    if not os.path.isdir(path_save_):
        os.mkdir(path_save_)
    for year in tqdm(range(2020, 2101, 1)):
        if os.path.isfile(f"{path_save_}/{scenario}_{year}.nc"):
            continue
        try:
            ds = getThr(df, model, scenario, year, thr_)
        except Exception as e:
            print(f"\n{model}-{year}-{scenario} has failed because of \n{e}")
            continue
        # Save it
        ds.to_netcdf(
            f"{path_save_}/{scenario}_{year}.nc",
            encoding={
                "THI": {
                    "dtype": "int16",
                    "zlib": True,
                    "complevel": 5,
                    "fletcher32": True,
                    "_FillValue": -999,
                }
            },
        )


def listERA(
    pathERA=args.pathERA,
):
    files_ = glob.glob1(pathERA, "*.nc")
    df = pd.DataFrame(
        {
            "model": "ERA5",
            "year": [int(x.split("_")[2]) for x in files_],
            "filename": [f"{pathERA}/{x}" for x in files_],
        }
    )
    return df.sort_values(by=["year"]).reset_index(drop=True)


def getThrERA(ds, model, scenario, year, thr_, adjust_lon=False):
    """
    Function to get the number of hours per grid cell per day for a given
    year of a used defined model and scenario
    """
    # specified THI threshold
    ds_ = xr.where(ds.THI >= thr_, 1, 0)
    # Calculate the sum per day
    ds_ = ds_.resample(time="1D").sum()
    gc.collect()
    ds_ = ds_.to_dataset(name="THI")
    del ds
    if adjust_lon:
        # Adjust lon values to make sure they are within (-180, 180)
        ds_ = convert360_180(ds_)
    # Add the model name and scenario as dimensions
    ds_ = ds_.expand_dims(model=[model], scenario=[scenario], threshold=[thr_])
    return ds_


def thrAllERA(model="ERA5", scenario="historical", path_save=path_save):
    """
    Calculate the number of hours above the defined threshold for each day/
    gric cell for a given model/scenario combination. Do the same for all
    the years available and save it. The netcdf is saved in int16 type and
    compressed to save space.
    """
    df = listERA()
    for year in tqdm(range(1980, 2019, 1)):
        # Read the netcdf file
        dsERA = (
            xr.open_dataset(df.loc[df.year == year].filename.item())
            .drop_vars(["t2m", "rh", "t2mC"])
            .load()
        )
        for thr_ in [68.8, 84.0]:
            # Directory to save data
            path_save_ = f"{path_save}/{str(thr_).replace('.', '')}"
            if not os.path.isdir(path_save_):
                os.mkdir(path_save_)
            path_save_ = f"{path_save_}/{model}"
            if not os.path.isdir(path_save_):
                os.mkdir(path_save_)
            if os.path.isfile(f"{path_save_}/{scenario}_{year}.nc"):
                continue
            try:
                print(path_save)
                print(path_save_, "\n-----------")
                ds = getThrERA(
                    ds=dsERA, model="ERA5", scenario="historical", year=year, thr_=thr_
                )
            except Exception as e:
                print(f"\n{model}-{year} has failed because of \n{e}")
                continue
            # Save it
            ds.to_netcdf(
                f"{path_save_}/historical_{year}.nc",
                encoding={
                    "THI": {
                        "dtype": "int16",
                        "zlib": True,
                        "complevel": 6,
                        "fletcher32": True,
                        "_FillValue": -999,
                    }
                },
            )
            del ds
            gc.collect()
        del dsERA
        gc.collect()


# ========================================================================= #
if __name__ == "__main__":
    # Get all the data
    df = pd.concat([getDat(path_models, model) for model in models])
    df = df.reset_index(drop=True)

    thr = args.threshold
    scenario = args.scenario
    # Process all the datasets
    # print(type(args.thr))
    print(f"Processing THI data for {thr} threshold ({scenario}).")
    models_ = df.model.unique()
    func_ = partial(thrAll, scenario=scenario, df=df, thr_=thr, path_save=path_save)
    process_map(func_, models_, max_workers=2)
