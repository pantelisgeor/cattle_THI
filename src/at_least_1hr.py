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

parser = argparse.ArgumentParser()
parser.add_argument("--pathTarget", type=str, help="Path to directory where target data will be stored.")
parser.add_argument("--pathDat", type=str, help="Path to directory where the hours above threshold data is stored.")
parser.add_argument("--pathLSM", type=str, help="Path to directory where the CMIP6 land sea mask is stored.")
args = parser.parse_args()

# Path to target directory
path_target = args.pathTarget
# Path to data with hours above threshold per day
pathDat = args.pathDat
os.makedirs(path_target, exist_ok=True)
os.chdir(path_target)


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
    ds = xr.open_dataset(path_)
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
    for f_ in tqdm(dats_):
        if f_ == dats_[0]:
            ds = xr.open_dataset(f_).resample(time="1Y").sum()
        else:
            ds = ds.merge(xr.open_dataset(f_).resample(time="1Y").sum())
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
    # Merge with land sea mask
    # Land sea mask
    lsm = xr.open_dataset(args.pathLSM)
    # Make the land sea mask a binary mask
    lsm["lsm"] = xr.where(lsm.lsm > 0, 1, np.nan)

    # Define the threshold
    thr_ = 68.8
    # List the subdirectories (model names)
    models = [x.split("/")[-2] for x
              in glob.glob(f"{pathDat}/{str(thr_).replace('.', '')}/*/",
                           recursive=True)]

    # Get all the data for the models
    df = pd.concat([getDat(f"{pathDat}/{str(thr_).replace('.', '')}", model)
                    for model in models])
    df = df.reset_index(drop=True)

    for model_ in models:
        for scenario in ["ssp245", "ssp585"]:
            print(f"Processing {model_} - scenario: {scenario}")
            # Make the processYear partial
            func_ = partial(processYear, model=model_,
                            scenario=scenario, df=df)
            process_map(func_, np.arange(2020, 2101, 1),
                        max_workers=20, chunksize=1)

    # Define the threshold
    thr_ = 84.0
    # List the subdirectories (model names)
    models = [x.split("/")[-2] for x
              in glob.glob(f"{pathDat}/{str(thr_).replace('.', '')}/*/",
                           recursive=True)]

    # Get all the data for the models
    df = pd.concat([getDat(f"{pathDat}/{str(thr_).replace('.', '')}", model)
                    for model in models])
    df = df.reset_index(drop=True)

    for model_ in models:
        for scenario in ["ssp245", "ssp585"]:
            print(f"Processing {model_} - scenario: {scenario}")
            # Make the processYear partial
            func_ = partial(processYear, model=model_,
                            scenario=scenario, df=df)
            process_map(func_, np.arange(2020, 2101, 1),
                        max_workers=40, chunksize=1)

    # List all the datasets for a given threshold and combine
    # them in a single netcdf file
    os.chdir("/nvme/h/pgeorgiades/data_p143/cattle_heat/THI_paper/" +
             "predictions_analysis/at_least_1_hr")

    thr_ = 68.8
    # List the subdirectories (model names)
    models = [x.split("/")[-2] for x
              in glob.glob(f"{pathDat}/{str(thr_).replace('.', '')}/*/",
                           recursive=True)]
    df_ = pd.concat([getDat(path_models=f"{str(thr_).replace('.', '')}",
                            model=model) for model in models])
    # for model in models:
    #     print(model, " ", thr_)
    #     getModel(model, df_)
    process_map(partial(getModel, df_=df_), models, max_workers=4,
                chunksize=1)

    thr_ = 84.0
    # List the subdirectories (model names)
    models = [x.split("/")[-2] for x
              in glob.glob(f"{pathDat}/{str(thr_).replace('.', '')}/*/",
                           recursive=True)]
    df_ = pd.concat([getDat(path_models=f"{str(thr_).replace('.', '')}",
                            model=model) for model in models])
    # for model in models:
    #     print(model, " ", thr_)
    #     getModel(model, df_)
    process_map(partial(getModel, df_=df_), models, max_workers=3,
                chunksize=1)

    if not os.path.isfile("atLeast1hr.nc"):
        # List all the netcdf files in the directory
        files_ = glob.glob1(f"{os.getcwd()}/yearly", "*.nc")
        # Read them all and combine them in one
        for f_ in tqdm(files_):
            if f_ == files_[0]:
                ds = xr.open_dataset(f"yearly/{f_}")
            else:
                ds = ds.merge(xr.open_dataset(f"yearly/{f_}"))
            gc.collect()

        del files_

        # Apply the land sea mask to the data
        ds = ds.merge(lsm)
        ds["THI"] = ds["THI"] * ds["lsm"]
        ds = ds.drop_vars("lsm")

        ds.to_netcdf("atLeast1hr.nc",
                     encoding={"THI": {"dtype": "int32",
                                       "zlib": True, "complevel": 6,
                                       "fletcher32": True,
                                       "_FillValue": -999}})
    else:
        ds = xr.open_dataset("atLeast1hr.nc")

    if not os.path.isfile("atLeast1hr.parquet"):
        # Calculate the yearly mean (globally) and convert to dataframe
        dfYear = ds.mean(dim=["lat", "lon"])\
            .to_dataframe()\
            .reset_index(drop=False)
        # Save it
        dfYear.to_parquet("atLeast1hr.parquet", compression="gzip")
        dfYear.to_csv("atLeast1hr.csv", index=False)
    else:
        dfYear = pd.read_parquet("atLeast1hr.parquet")
