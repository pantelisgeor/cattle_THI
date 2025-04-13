import cdsapi
import os
import argparse


def download_era5(year: int, variable: str, pathTarget: str) -> None:
    # Var short name
    var_ = "2t" if variable == "2m_temperature" else "2d"
    # Target path
    target = f"{pathTarget}/{var_}"
    os.makedirs(target, exist_ok=True)
    # File name
    target = f"{target}/ERA5_{year}_MET-VARIABLES_global_var_{variable}.nc"
    # Check if file already exists
    if os.path.isfile(target):
        print(f"File {target} already exists")
        return None
    # Download data
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": ["1990"],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(0, 24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    client = cdsapi.Client()
    r = client.retrieve(dataset, request)
    r.download(target)
    return None


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Download ERA5 data")
    parser.add_argument(
        "--path", type=str, default="data/ERA5", help="Path to save the data"
    )
    args = parser.parse_args()
    # Variables list
    variables = ["2m_dewpoint_temperature", "2m_temperature"]
    # Years list
    years = range(1990, 2020, 1)
    for variable in variables:
        for year in years:
            download_era5(year, variable, pathTarget=args.path)


if __name__ == "__main__":
    main()
