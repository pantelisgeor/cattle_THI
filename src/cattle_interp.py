import xarray as xr
import rioxarray as rxr
import os
import argparse


def convert_longitude(ds, lon="longitude"):
    """
    Convert longitude from 0-360 to -180 to 180 format in an xarray dataset

    Args:
        ds (xarray.Dataset): Input dataset with longitude coordinates

    Returns:
        xarray.Dataset: Dataset with converted longitude coordinates
    """
    # Select longitudes > 180 and subtract 360 to convert to negative values
    ds.coords[lon] = (ds.coords[lon] + 180) % 360 - 180

    # Sort the dataset by longitude
    return ds.sortby(lon)


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pathDat", type=str, help="Path to the cattle data directory"
    )
    parser.add_argument(
        "--testDat", type=str, help="Path to a test dataset from the THI projections."
    )
    args = parser.parse_args()

    os.chdir(args.pathDat)
    
    # Read a CMIP6 dataset to interpolate using it's dimensions
    ds_era = convert_longitude(
        xr.open_dataset(args.testDat),
        lon="lon",
    )

    # Read the cattle dataset
    if not os.path.isfile("cattle.nc"):
        ds_ = (
            rxr.open_rasterio("5_Ct_2010_Da.tif")
            .rename({"x": "lon", "y": "lat"})
            .sel(band=1)
            .squeeze()
            .drop_vars(["band", "spatial_ref"])
        )
        ds = convert_longitude(ds_, lon="lon")
        # Index the cattle dataset using the latitude limits of the cmip6 datasaet
        ds = ds.sel(lat=slice(ds_era.lat.max().item(), ds_era.lat.min().item()))
    else:
        ds = convert_longitude(xr.open_dataset("cattle.nc"), lon="lon")
        # Index the cattle dataset using the latitude limits of the cmip6 datasaet
        ds = ds.sel(lat=slice(ds_era.lat.min().item(), ds_era.lat.max().item()))

    # Interpolate based on the cmip6 dataset
    ds_ = ds.interp_like(ds_era, method="slinear")
    
    if isinstance(ds_, xr.DataArray):
        ds_ = ds_.to_dataset(name="cattle")
        ds_["cattle"] = xr.where(ds_["cattle"] < 0, 0, ds_["cattle"])

    ds_.to_netcdf(
        "cattle_interp.nc",
        encoding={"cattle": {"zlib": True, "complevel": 6, "dtype": "float32"}},
    )
