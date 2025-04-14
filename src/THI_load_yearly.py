import os
import xarray as xr
import pandas as pd
import numpy as np
import glob
import gc
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
import argparse


# ================================================================ #
def getDat(path_models, model):
    # Construct the path to the model's data directory
    path_ = f"{path_models}/{model}"
    # List the netcdf files
    files_ = glob.glob1(path_, "*.nc")
    # Put the data in a dataframe
    df = pd.DataFrame(
        {
            "model": model,
            "year": [int(x.split("_")[1].split(".")[0]) for x in files_],
            "scenario": [x.split("_")[0] for x in files_],
            "filename": [f"{path_models}/{model}/{x}" for x in files_],
        }
    )
    return df.sort_values(by=["year", "scenario"], ascending=True)


def getModel(model, df_):
    if os.path.isfile(f"yearly/{model}.nc"):
        return None
    # Subset the data (paths_) for the model
    dats_ = df_.loc[df_.model == model].filename.values
    for f_ in tqdm(dats_):
        if f_ == dats_[0]:
            ds = xr.open_dataset(f_).resample(time="1Y").mean()
        else:
            ds = ds.merge(xr.open_dataset(f_).resample(time="1Y").mean())
        gc.collect()
    # Save it
    ds.to_netcdf(
        f"yearly/{model}.nc",
        encoding={
            "THI": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 5,
                "fletcher32": True,
                "_FillValue": -999,
            }
        },
    )
    del ds
    gc.collect()
    return None


# ================================================================ #
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Get the yearly THI data for all models"
    )
    parser.add_argument("--pathDat", type=str, help="Path to the THI load data.")
    parser.add_argument("--pathLSM", type=str, help="Path to the CMIP6 land sea mask.")
    args = parser.parse_args()

    # Paths
    pathDat = args.pathDat
    os.chdir(pathDat)
    models = [x.split("/")[-2] for x in glob.glob("*/", recursive=True)]
    os.makedirs("yearly", exist_ok=True)

    df = pd.DataFrame()
    for mod_ in models:
        if mod_ == "yearly":
            continue
        df = pd.concat([df, getDat(os.getcwd(), mod_)])

    process_map(
        partial(getModel, df_=df), [df.model.unique()], max_workers=3, chunksize=1
    )

    # Combine the yearly datasets into one
    for i, f_ in enumerate(glob.glob1("yearly/", "*.nc")):
        print(i)
        if i == 0:
            ds = xr.open_dataset(f"yearly/{f_}")
        else:
            ds = ds.merge(xr.open_dataset(f"yearly/{f_}"))

    # Merge with land sea mask
    # Land sea mask
    lsm = xr.open_dataset(args.pathLSM)
    # Make the land sea mask a binary mask
    lsm["lsm"] = xr.where(lsm.lsm > 0, 1, np.nan)
    ds = ds.merge(lsm)
    ds["THI"] = ds["THI"] * ds["lsm"]
    ds = ds.drop_vars("lsm")

    # Save it
    ds.to_netcdf(
        "THI_load.nc",
        encoding={
            "THI": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 6,
                "fletcher32": True,
            }
        },
    )

    # Get the global means and convert to dataframe
    df = ds.mean(dim=["lat", "lon"]).to_dataframe().reset_index(drop=False)
    # And save them in parquet and csv format
    df.to_parquet("THI_load_ERA.parquet", index=False, compression="gzip")
