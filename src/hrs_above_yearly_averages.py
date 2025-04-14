import os
import gc
import xarray as xr
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pathDat",
    type=str,
    help="Path to directory where the hours above threshold data is stored.",
)
args = parser.parse_args()

os.chdir(args.pathDat)
# Land sea mask
lsm = xr.open_dataset("CMIP6_lsm.nc")
# Make the land sea mask a binary mask
lsm["lsm"] = xr.where(lsm.lsm > 0, 1, np.nan)


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
    return df.sort_values(by=["year", "scenario"], ascending=True).reset_index(
        drop=True
    )


def processModel(model, df, lsm):
    if os.path.isfile(f"yearly/{model}.nc"):
        return None
    # Get the filenames for this model
    dats_ = df.loc[df.model == model].reset_index(drop=True)
    for f_ in tqdm(dats_.filename.values):
        if f_ == dats_.filename.values[0]:
            ds = xr.open_dataset(f_).resample(time="1Y").mean()
        else:
            ds = ds.merge(xr.open_dataset(f_).resample(time="1Y").mean())
        gc.collect()
    # Merge with land sea mask
    ds = ds.merge(lsm)
    ds["THI"] = ds["THI"] * ds["lsm"]
    ds = ds.drop_vars("lsm")
    # Save it
    ds.to_netcdf(
        f"yearly/{model}.nc",
        encoding={
            "THI": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 6,
                "fletcher32": True,
            }
        },
    )
    del ds
    gc.collect()
    return None


# ================================================================ #
if __name__ == "__main__":
    df = pd.DataFrame()

    for thr_ in ["688", "840"]:
        models = [x.split("/")[-2] for x in glob.glob(f"{thr_}/*/", recursive=True)]
        for mod_ in models:
            if mod_ == "yearly":
                continue
            df = pd.concat([df, getDat(f"{os.getcwd()}/{thr_}", mod_)])

    del thr_, mod_
    gc.collect()
    df = df.reset_index(drop=True)

    # Process all the datasets in parallel
    for model in models:
        print(model)
        processModel(model=model, df=df, lsm=lsm)

    if not os.path.isfile("hrs_above_thr.nc"):
        # Read them all and combine them into one xarray dataset
        ds = xr.Dataset()
        for f_ in tqdm(glob.glob("yearly/*.nc")):
            ds = ds.merge(xr.open_dataset(f_))

        # Save it
        ds.to_netcdf(
            "hrs_above_thr.nc",
            encoding={
                "THI": {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 6,
                    "fletcher32": True,
                }
            },
        )
    else:
        ds = xr.open_dataset("hrs_above_thr.nc")

    if not os.path.isfile("hrs_above_thr.parquet"):
        # Calculate the global mean and concert to dataframe
        df = ds.mean(dim=["lat", "lon"]).to_dataframe().reset_index(drop=False)
        # Save it as parquet and csv
        df.to_parquet("hrs_above_thr.parquet", compression="gzip")
        df.to_csv("hrs_above_thr.csv", index=False)
    else:
        df = pd.read_parquet("hrs_above_thr.parquet")
