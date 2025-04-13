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
        "year": [int(x.split("_")[1].split(".")[0]) for x in files_],
        "scenario": [x.split("_")[0] for x in files_],
        "filename": [f"{path_models}/{model}/{x}" for x in files_]
    })
    return df.sort_values(by=["year", "scenario"], ascending=True)


def processYear(year, model, scenario, df, pathDat=pathDat):
    # Construct the path to the directory to save the data
    path_ = df.loc[(df.model == model) & (df.year == year) &
                   (df.scenario == scenario)].filename.item()
    pathSave = f"{str(thr_).replace('.', '')}/{model}"
    os.makedirs(pathSave, exist_ok=True)
    pathSave = f"{pathSave}/{path_.split('/')[-1]}"
    if os.path.isfile(pathSave):
        return None
    # Read the dataset
    ds = xr.open_dataset(path_).load()
    # Assing 1 if at least 1 hr is above threshold and 0 otherwise
    ds["THI"] = xr.where(ds.THI > 0, 1, 0)
    # Save it
    ds.to_netcdf(pathSave, encoding={"THI": {"dtype": "int16",
                                             "zlib": True, "complevel": 5,
                                             "fletcher32": True,
                                             "_FillValue": -999}})
    del ds
    gc.collect()
    return None


def getModel(model, df_):
    if os.path.isfile(f"yearly/{model}_thr_{str(thr_).replace('.', '')}.nc"):
        return None
    # Subset the data (paths_) for the model
    dats_ = df_.loc[df_.model == model].filename.values
    ds = xr.Dataset()
    for f_ in tqdm(dats_):
        ds_ = xr.open_dataset(f_)
        ds_["THI"] = xr.where(ds_["THI"] > 0, 1, 0)
        ds_ = ds_.resample(time="1Y").sum()
        ds = ds.merge(ds_)
        del ds_
        gc.collect()
    # Save it
    ds.to_netcdf(f"yearly/{model}_thr_{str(thr_).replace('.', '')}.nc",
                 encoding={"THI": {"dtype": "int32",
                                   "zlib": True, "complevel": 5,
                                   "fletcher32": True,
                                   "_FillValue": -999}})
    del ds
    gc.collect()
    return None


# ===============================  =============================== #
if __name__ == "__main__":

    os.chdir("/nvme/h/pgeorgiades/data_p143/cattle_heat/THI_paper/" +
         "predictions_analysis/at_least_1_hr")

    # Path to data with hours above threshold per day
    pathDat = "../hrs_above_thr"
    
    # Merge with land sea mask
    # Land sea mask
    lsm = xr.open_dataset("../ERA5_lsm.nc").rename({"longitude": "lon",
                                                    "latitude": "lat"})
    # Make the land sea mask a binary mask
    lsm["lsm"] = xr.where(lsm.lsm > 0, 1, np.nan)

    # Define the threshold
    thr_ = 68.8

    # Get all the data for the models
    df = getDat(f"{pathDat}/{str(thr_).replace('.', '')}", "ERA5")
    df = df.reset_index(drop=True)

    model = "ERA5"

    print(f"Processing {model} - scenario: historical")
    # Make the processYear partial
    process_map(partial(processYear, model=model,
                        scenario="historical", df=df),
                np.arange(1980, 2019, 1),
                max_workers=20, chunksize=1)

    # Define the threshold
    thr_ = 84.0

    # Get all the data for the models
    df = getDat(f"{pathDat}/{str(thr_).replace('.', '')}", "ERA5")
    df = df.reset_index(drop=True)

    print(f"Processing {model} - scenario: historical")
    # Make the processYear partial
    process_map(partial(processYear, model=model,
                        scenario="historical", df=df),
                np.arange(1980, 2019, 1),
                max_workers=10, chunksize=1)

    # ---------------------------------------------------------------------- #
    # List all the datasets for a given threshold and combine
    # them in a single netcdf file
    os.chdir("/nvme/h/pgeorgiades/data_p143/cattle_heat/THI_paper/" +
             "predictions_analysis/at_least_1_hr")

    thr_ = 68.8
    # List the subdirectories (model names)
    # Get all the data for the models
    df_ = getDat(f"{pathDat}/{str(thr_).replace('.', '')}", "ERA5")
    df_ = df_.reset_index(drop=True)
    # for model in models:
    #     print(model, " ", thr_)
    #     getModel(model, df_)
    getModel(model="ERA5", df_=df_)

    thr_ = 84.0
    # List the subdirectories (model names)
    df_ = getDat(f"{pathDat}/{str(thr_).replace('.', '')}", "ERA5")
    df_ = df_.reset_index(drop=True)
    # for model in models:
    #     print(model, " ", thr_)
    #     getModel(model, df_)
    getModel(model="ERA5", df_=df_)

    if not os.path.isfile("atLeast1hr_ERA.nc"):
        # List all the netcdf files in the directory
        files_ = glob.glob1(f"{os.getcwd()}/yearly", "ERA5*.nc")
        # Read them all and combine them in one
        for f_ in tqdm(files_):
            if f_ == files_[0]:
                ds = xr.open_dataset(f"yearly/{f_}")
            else:
                ds = ds.merge(xr.open_dataset(f"yearly/{f_}"))
            gc.collect()

        del files_

        # Apply the land sea mask to the data
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})\
            .merge(lsm.squeeze().drop_vars("time"))
        ds["THI"] = ds["THI"] * ds["lsm"]
        ds = ds.drop_vars("lsm")

        ds.to_netcdf("atLeast1hr_ERA.nc",
                     encoding={"THI": {"dtype": "int32",
                                       "zlib": True, "complevel": 6,
                                       "fletcher32": True,
                                       "_FillValue": -999}})
    else:
        ds = xr.open_dataset("atLeast1hr_ERA.nc")

    if not os.path.isfile("atLeast1hr_ERA.parquet"):
        # Calculate the yearly mean (globally) and convert to dataframe
        dfYear = ds.mean(dim=["lat", "lon"])\
            .to_dataframe()\
            .reset_index(drop=False)
        # Save it
        dfYear.to_parquet("atLeast1hr_ERA.parquet", compression="gzip")
        dfYear.to_csv("atLeast1hr_ERA.csv", index=False)
    else:
        dfYear = pd.read_parquet("atLeast1hr_ERA.parquet")
