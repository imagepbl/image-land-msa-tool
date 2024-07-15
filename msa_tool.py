"""
IMAGE-Land MSA tool v1.0 - Geanderson Ambrósio - ambrosiog@pbl.nl

The IMAGE-Land MSA tool calculates the index Mean Species Abundance, for plants (uses 3 out of 3 pressures) and warm-blooded vertebrates (uses 2 out of 5 pressures), based on outputs from IMAGE model.
The tool is based on GLOBIO 4 MSA implementation, which is more detailed at the grid level, but takes more time to run.  
Even though the code calculate MSA for vertebrates, it  only account for two pressures while the original MSA vertebrates from GLOBIO accounts for five pressures.
We validated results for plants and vertebrates and agreed that the MSA tool is a good estimator for MSA plants, but not for MSA vertebrates.
Please do not use the MSA tool to calculate MSA for vertebrates.

The file parameters.ini controls the main settings for this tool, like scenarios and years for which you want the tool to run.
One specific settings is still defined in this source code: see file_list in the function compute_share.
"""
from pathlib import Path
import pandas as pd
import json
import pym
import xarray as xr
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import dask
import os.path
import os
import configparser
import datetime
import shutil
import functools
import time
import traceback as tb

start_time = time.time()
print("Starting script at", time.strftime("%H:%M:%S", time.localtime()))

print("Importing packages")
"""
Importing required libraries
"""

AREA_REGION_INPUT = os.path.join("input", "")


class DataPath:
    """
    Class based on the function get_paths from previous msa_tool version.
    Deals with paths for input files from scenario folder, input files from working directory and output folder.
    """

    def __init__(self, input):
        scen_name = os.path.basename(input)
        project_name = os.path.basename(os.path.dirname(input))

        self._scenario_input = os.path.join(input, "netcdf", "")
        self._output = os.path.join(
            "msa_tool_output", project_name, scen_name, "")
        self._area_region_input = AREA_REGION_INPUT
        self._gbuiltup_path = None

    @property
    def scenario_input(self):
        return self._scenario_input

    @property
    def output(self):
        return self._output

    @property
    def area_region_input(self):
        return self._area_region_input

    @property
    def gbuiltup_path(self):
        if self._gbuiltup_path is None:
            return self._area_region_input
        return self._gbuiltup_path


class Parameters:
    """
    Class based on the function load_parameters from previous msa_tool version.
    Deals with parameters in the file parameters.ini
    """

    def __init__(self, filename="parameters.ini"):
        config = configparser.ConfigParser()
        config.read(filename)

        self._input_path = config["MSA_TOOL"]["input_path"]
        self._list_of_projects = [
            proj.lstrip()
            for proj in config["MSA_TOOL"]["list_of_projects"][1:-1].split(",")
            if proj != ""
        ]
        self._list_of_scen = [
            scen.lstrip()
            for scen in config["MSA_TOOL"]["list_of_scen"][1:-1].split(",")
            if scen != ""
        ]
        self._years = [
            int(year) for year in config["MSA_TOOL"]["years"][1:-1].split(",")
        ]
        self._make_figures = config.getboolean("MSA_TOOL", "make_figures")
        self._force_compute_files = config.getboolean(
            "MSA_TOOL", "force_compute_files")
        self._clear_cache = config.getboolean("MSA_TOOL", "clear_cache")
        self._species_names = [
            s.title().lstrip()
            for s in config.get("MSA_TOOL", "species_names")[1:-1].split(",")
        ]

    @property
    def input_path(self):
        return self._input_path

    @property
    def list_of_projects(self):
        return self._list_of_projects

    @property
    def list_of_scen(self):
        return self._list_of_scen

    @property
    def years(self):
        return self._years

    @property
    def make_figures(self):
        return self._make_figures

    @property
    def force_compute_files(self):
        return self._force_compute_files

    @property
    def clear_cache(self):
        return self._clear_cache

    @property
    def species_names(self):
        return self._species_names


print("Loading functions")
selected_years = []


def load_dataset(file_list: list, data_path, **kwargs_sel):
    """
    Function to load multiple datasets at once with chunks in time to optimize memory usage
    """

    @functools.lru_cache(maxsize=1)
    def load_ref_dataset_for_coords(path):
        """
        Load the dataset to be used as reference for the coordinates.

        Notice lru_cache is used to avoid loading the same dataset multiple times.
        """
        return (
            xr.open_dataset(path, engine="netcdf4")[["latitude", "longitude"]]
            .astype(np.float32)
            .compute()
        )

    def pre_processing(dataset):
        """
        Function to harmonize coordinates among datasets and select timesteps
        """
        if "lat" in dataset.coords:
            dataset = dataset.rename(dict(lat="latitude"))
        if "lon" in dataset.coords:
            dataset = dataset.rename(dict(lon="longitude"))
        if "N" in dataset.coords:
            dataset = dataset.rename(dict(N="NFORMID"))

        ### WARNING ####
        ### it is not possible to run a case selecting just one time step ###
        ### SEE WARNING IN parameters.ini ###
        if "time" in dataset.coords:
            if dataset.time.size == 1:
                dataset = dataset.squeeze().drop("time")

        # Adjusting the values for the coordinates latitude and longitude to be the same as GREG.NC
        ref_dataset = load_ref_dataset_for_coords(
            f"{data_path.area_region_input}GREG.NC"
        )

        if "latitude" in dataset.coords:
            # print("Latitude of dataset are not the same as GREG.NC, replacing it")
            dataset = dataset.assign_coords(latitude=ref_dataset.latitude.data)
        if "longitude" in dataset.coords:
            # print("Longitude of dataset are not the same as GREG.NC, replacing it")
            dataset = dataset.assign_coords(
                longitude=ref_dataset.longitude.data)
        return dataset.transpose(*sorted(dataset.dims))

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        msa_dataset = xr.open_mfdataset(
            paths=file_list,
            chunks=dict(time="auto", latitude="auto", longitude="auto"),
            # chunks=dict(time="auto"),
            join="right",
            preprocess=pre_processing,
            engine="netcdf4",
        )

    if not kwargs_sel and selected_years:
        kwargs_sel["time"] = [
            datetime.date(year, 1, 1)
            for year in selected_years
            if year in msa_dataset.time.dt.year.values
        ]

    return msa_dataset.sel(**kwargs_sel)


land_use_type_names = [  # Land use types based on GLOBIO4 classification and extended to include bioenergy biomass and carbon plantation.
    "Food/feed cropland - Intense use",
    "Food/feed cropland - Minimal use",
    "Pasture - Intense use",
    "Pasture - Minimal use",
    "Other plantation",
    "Secondary vegetation",
    "Urban",
    "Natural land",
    "Bioenergy cropland - Intense use",
    "Bioenergy cropland - Minimal use",
    "Bioenergy plantation",
    "Carbon plantation",
]


def compute_share(data_path, parameters):
    """
    Function to map share and area of land_use_type from GLOBIO4 based on Netcdf outputs from IMAGE-land
    Mapping is defined below for each land_use_type (i = index of land_use_type)
    """
    result_file = f"{data_path.output}share.nc"

    if not need_to_compute(result_file, parameters):
        return

    print("    Calculating share of land use types")

    file_list = [
        f"{data_path.scenario_input}GLCT.NC",
        f"{data_path.gbuiltup_path}",  # WARNING ####
        ### GBUILTUP.NC is not available in Biodiv_Post2020 runs. ###
        ### For Biodiv_Post2020, set area_region_input instead of scenario_input, and you will use a GBUILTUP file for SSP2 which is in the folder input. ###
        f"{data_path.scenario_input}GFERTILIZER.NC",
        f"{data_path.area_region_input}GAreaCellNoWater.NC",
        f"{data_path.scenario_input}GFRAC.NC",
        f"{data_path.scenario_input}GFORMAN.NC",
        f"{data_path.area_region_input}GREG.NC",
    ]

    msa_dataset = load_dataset(file_list, data_path).assign_coords(
        land_use_type=land_use_type_names
    )

    # Create a new variables to store share and area for each land_use_type
    # it is possible to make a better memory usage by creating a dataset for each calculation where each array is the data of a unique land use type
    # later I could use concat  to join them into one single dataset identifying land_use_type by coordinates (as done for pressures)
    msa_dataset["share"] = xr.DataArray(
        np.float32(0.0),  # Initial value of the array
        coords=[  # Coordinates of the array
            msa_dataset.latitude,
            msa_dataset.longitude,
            msa_dataset.time,
            msa_dataset.land_use_type,
        ],
    ).chunk(dict(time="auto", latitude="auto", longitude="auto"))
    msa_dataset["area"] = xr.DataArray(
        np.float32(0.0),  # Initial value of the array
        coords=[  # Coordinates of the array
            msa_dataset.latitude,
            msa_dataset.longitude,
            msa_dataset.time,
            msa_dataset.land_use_type,
        ],
    ).chunk(dict(time="auto", latitude="auto", longitude="auto"))

    ### Urban ###
    i = 6  # Urban - Urban area can occur over any GLCT
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = msa_dataset["GBUILTUP"]
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["GBUILTUP"] / msa_dataset[
        "GAreaCellNoWater"
    ].where(msa_dataset["GAreaCellNoWater"] > 0)
    # non_urban_area to be used with GFRAC
    msa_dataset["non_urban_area"] = (
        msa_dataset["GAreaCellNoWater"] - msa_dataset["GBUILTUP"]
    )

    ### Pasture - Intense Use ###
    i = 2  # Pasture - Intense use - GFRAC must be "grass", GLCT must be 1
    # Area
    for grass in [b"grass                                             ", b"grass"]:
        if grass in msa_dataset["GFRAC"]["NGFBFC"].values:
            break
    msa_dataset["area"][dict(land_use_type=i)] = (
        msa_dataset["GFRAC"].sel(NGFBFC=grass) * msa_dataset["non_urban_area"]
    ).where(msa_dataset["GLCT"] == 1)
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Pasture - Minimal use ###
    i = 3  # Pasture - Minimal use - GFRAC must be "grass", GLCT must be 2
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = (
        msa_dataset["GFRAC"].sel(NGFBFC=grass) * msa_dataset["non_urban_area"]
    ).where(msa_dataset["GLCT"] == 2)
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Food/feed Crop ###
    ### Share of all food/feed cropland to be later divided between intense and minimal ###
    # Loop to get the sum of GFRAC related to food/feed cropland share
    NGFBFC_list = [
        i
        for i in range(msa_dataset.NGFBFC.size)
        if i not in [0, 13, 17, 18, 19, 20, 21, 34]
    ]
    msa_dataset["non_urban_crop_share"] = (
        msa_dataset["GFRAC"].isel(NGFBFC=NGFBFC_list).fillna(0.0).sum("NGFBFC")
    )

    ### Crop - Intense use ###
    # Conditions for intense use
    # GLCT must be 1 OR 6, GFRAC "non_urban_crop_share" must exist, GFERTILIZER must be > 10
    GFRAC = (msa_dataset["non_urban_crop_share"] > 0) & (
        msa_dataset["GFERTILIZER"].isel(NFERTSMT=0) > 10.0
    )
    GLCT = (msa_dataset["GLCT"] == 1) | (msa_dataset["GLCT"] == 6)

    ### Food/feed crop - Intense use ###
    i = 0  # Food/feed cropland - Intense use
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = xr.where(
        GFRAC & GLCT,  # condition
        msa_dataset["non_urban_area"]
        * msa_dataset["non_urban_crop_share"],  # where True
        np.NaN,  # where False
    )
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Food/feed crop - Minimal use ###
    i = 1  # Crop - Minimal use - GFRAC approach
    # GLCT must be 1 OR 6, GFRAC "non_urban_crop_share" must exist, GFERTILIZER must be <= 10
    # Conditions
    GFRAC = (msa_dataset["non_urban_crop_share"] > 0) & (
        msa_dataset["GFERTILIZER"].isel(NFERTSMT=0) <= 10.0
    )
    GLCT = (msa_dataset["GLCT"] == 1) | (msa_dataset["GLCT"] == 6)
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = xr.where(
        GFRAC & GLCT,  # condition
        msa_dataset["non_urban_area"]
        * msa_dataset["non_urban_crop_share"],  # where True
        np.NaN,  # where False
    )
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Bioenergy Crop ###
    ### Share of all bioenergy cropland to be later divided between intense and minimal ###
    # Loop to get the sum of GFRAC related to bioenergy cropland share
    NGFBFC_list = [i for i in range(msa_dataset.NGFBFC.size) if i in [
        17, 18, 19, 21]]
    msa_dataset["non_urban_crop_share_bioenergy"] = (
        msa_dataset["GFRAC"].isel(NGFBFC=NGFBFC_list).fillna(0.0).sum("NGFBFC")
    )

    ### Bioenergy Crop - Intense use ###
    i = 8  # Crop - Intense use
    # GLCT must be 1 OR 6, GFRAC "non_urban_crop_share_bioenergy" must exist, GFERTILIZER must be > 10
    # Conditions
    GFRAC = (msa_dataset["non_urban_crop_share_bioenergy"] > 0) & (
        msa_dataset["GFERTILIZER"].isel(NFERTSMT=0) > 10.0
    )
    GLCT = (msa_dataset["GLCT"] == 1) | (msa_dataset["GLCT"] == 6)
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = xr.where(
        GFRAC & GLCT,  # condition
        msa_dataset["non_urban_area"]
        * msa_dataset["non_urban_crop_share_bioenergy"],  # where True
        np.NaN,  # where False
    )
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Bioenergy Crop - Minimal use ###
    i = 9  # Crop - Minimal use - GFRAC approach
    # GLCT must be 1 OR 6, GFRAC "non_urban_crop_share" must exist, GFERTILIZER must be <= 10
    # Conditions
    GFRAC = (msa_dataset["non_urban_crop_share_bioenergy"] > 0) & (
        msa_dataset["GFERTILIZER"].isel(NFERTSMT=0) <= 10.0
    )
    GLCT = (msa_dataset["GLCT"] == 1) | (msa_dataset["GLCT"] == 6)
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = xr.where(
        GFRAC & GLCT,  # condition
        msa_dataset["non_urban_area"]
        * msa_dataset["non_urban_crop_share_bioenergy"],  # where True
        np.NaN,  # where False
    )
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### All types of Plantation ###
    ### Share and area of non-bioenergy plantation, bioenergy plantation and other plantation from GFRAC to be later added to plantation from GLCT ###
    # Loop to get the sum of GFRAC related to plantation share
    msa_dataset["non_urban_plantation_share"] = xr.zeros_like(
        msa_dataset["non_urban_area"]
    )
    # Plantation (share GFRAC)
    for specific_plantation in {14 - 1, 35 - 1}:
        msa_dataset["non_urban_plantation_share"] += (
            msa_dataset["GFRAC"].isel(NGFBFC=specific_plantation).fillna(0.0)
        )
    # Plantation (area GFRAC)
    msa_dataset["plantation_area_gfrac"] = (
        msa_dataset["non_urban_area"] *
        msa_dataset["non_urban_plantation_share"]
    ).where(msa_dataset["non_urban_plantation_share"] > 0.0)

    msa_dataset["non_urban_plantation_share_bioenergy"] = xr.zeros_like(
        msa_dataset["non_urban_area"]
    )

    # Bioenergy Plantation (share GFRAC)
    for specific_plantation in {
        21 - 1,
    }:
        msa_dataset["non_urban_plantation_share_bioenergy"] += (
            msa_dataset["GFRAC"].isel(NGFBFC=specific_plantation).fillna(0.0)
        )
    # Bioenergy Plantation (area GFRAC)
    msa_dataset["plantation_area_gfrac_bioenergy"] = (
        msa_dataset["non_urban_area"]
        * msa_dataset["non_urban_plantation_share_bioenergy"]
    ).where(msa_dataset["non_urban_plantation_share_bioenergy"] > 0.0)

    ### Other plantation ###
    i = 4  # Plantation - conditions for part 1 and 2 below
    # Share
    # Part 1 - GLCT must be 4 OR 5, GFORMAN must be 3
    msa_dataset["share"][dict(land_use_type=i)] = (
        xr.where(  # Part 1 (from GLCT and GFRAC)
            np.logical_and(
                np.logical_or(msa_dataset["GLCT"] ==
                              4, msa_dataset["GLCT"] == 5),
                msa_dataset["GFORMAN"].isel(NFORMID=0) == 3,
            ),  # condition
            msa_dataset["non_urban_area"]
            / msa_dataset["GAreaCellNoWater"].where(
                msa_dataset["GAreaCellNoWater"] > 0
            ),  # where True
            0.0,  # where False
        )
        # Part 2 - Adding plantation_area_gfrac to Part 1 (can occur on any GLCT)
    ).fillna(0.0) + (
        msa_dataset["plantation_area_gfrac"]
        / msa_dataset["GAreaCellNoWater"].where(
            msa_dataset["GAreaCellNoWater"] > 0.0, 0.0
        )
    ).fillna(
        0.0
    )
    # Add fillna to solve the problem with plantation 2021-10-07T17:15:31.011Z
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = msa_dataset["share"].isel(
        land_use_type=i
    ) * msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Bioenergy Plantation ###
    i = 10  # Plantation - conditions for part 1 and 2 below
    # Share
    # plantation_area_gfrac_bioenergy
    msa_dataset["share"][dict(land_use_type=i)] = (
        msa_dataset["plantation_area_gfrac_bioenergy"]
        / msa_dataset["GAreaCellNoWater"].where(
            msa_dataset["GAreaCellNoWater"] > 0.0, 0.0
        )
    ).fillna(0.0)
    # Add fillna to solve the problem with plantation 2021-10-07T17:15:31.011Z
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = msa_dataset["share"].isel(
        land_use_type=i
    ) * msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Carbon Plantation ###
    i = 11  # Carbon Plantation - GLCT = 3
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = (
        (
            msa_dataset["non_urban_area"]
            - msa_dataset["plantation_area_gfrac"].fillna(0.0)
            - msa_dataset["plantation_area_gfrac_bioenergy"].fillna(0.0)
        )
        / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)
    ).where(msa_dataset["GLCT"] == 3, 0.0)
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = msa_dataset["share"].isel(
        land_use_type=i
    ) * msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Secondary Vegetation ###
    i = 5  # Secondary Vegetation - conditions for part 1 and 2 below
    # Share
    # Part 1 - GLCT has to be 4 OR 5, GFORMAN has to be DIFFERENT from 3
    msa_dataset["share"][
        dict(land_use_type=i)
    ] = xr.where(  # Part 1 -> This part has data!
        np.logical_and(
            np.logical_or(msa_dataset["GLCT"] == 4, msa_dataset["GLCT"] == 5),
            msa_dataset["GFORMAN"].isel(NFORMID=0).data != 3,
        ),  # condition
        (
            (
                msa_dataset["non_urban_area"]
                - msa_dataset["plantation_area_gfrac"].fillna(0.0)
                - msa_dataset["plantation_area_gfrac_bioenergy"].fillna(0.0)
            )
            / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)
        ),  # Where True
        0.0,  # np.nan, # where false
    )
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = msa_dataset["share"].isel(
        land_use_type=i
    ) * msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Natural Land ###
    i = 7  # Natural land - GLCT has to be between [7, 20]
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = xr.where(
        np.logical_and(
            7.0 <= msa_dataset["GLCT"], msa_dataset["GLCT"] <= 20.0
        ),  # condition
        msa_dataset["GAreaCellNoWater"]
        - msa_dataset["GBUILTUP"]
        - msa_dataset["plantation_area_gfrac_bioenergy"].fillna(0.0)
        - msa_dataset["plantation_area_gfrac"].fillna(0.0),  # Where True
        0.0,  # where false
    )
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    # Saving share as netcdf
    # From DataArray to Dataset to Netcdf
    msa_dataset["share"].to_dataset().to_netcdf(result_file)

    msa_dataset.share.fillna(0.0).where(
        msa_dataset.GREG != 27
    ).sum(  # GREG != 27 to exclude Greenland
        "land_use_type"
    ).where(
        msa_dataset.GAreaCellNoWater > 0.0
    ).plot.hist(
        bins=51, log=False
    )  # GAreaCellNoWater > 0.0 to exclude ocean
    plt.savefig(f"{data_path.output}sum_share_land_use_type.png")

    msa_dataset.share.fillna(0.0).where(
        msa_dataset.GREG != 27
    ).sum(  # GREG != 27 to exclude Greenland
        "land_use_type"
    ).where(
        msa_dataset.GAreaCellNoWater > 0.0
    ).plot.hist(
        bins=51, log=True
    )  # GAreaCellNoWater > 0.0 to exclude ocean
    plt.savefig(f"{data_path.output}sum_share_land_use_type_log.png")

    # Deleting unnecessary arrays in Dataset
    msa_dataset.close()


def compute_msa_lu(data_path, parameters):
    """
    Calculate MSA Land Use
    """
    result_file = f"{data_path.output}msa_grid_lu.nc"

    if not need_to_compute(result_file, parameters):
        return

    print("    Calculating Land Use MSA")

    file_list = [f"{data_path.output}share.nc"]

    # Add coordinates for species (plants and vertebrates)
    msa_dataset = load_dataset(file_list, data_path)
    # Add msa_lu discrete values
    msa_dataset["input_msa"] = xr.DataArray(
        np.array(
            [
                [
                    0.13,
                    0.13,
                    0.19,
                    0.25,
                    0.29,
                    0.55,
                    0.31,
                    1,
                    0.13,
                    0.13,
                    0.29,
                    0.55,
                ],  # Land use MSA for Plants in the same order as land_use_type
                [
                    0.36,
                    0.54,
                    0.50,
                    0.36,
                    0.58,
                    0.62,
                    0.26,
                    1,
                    0.36,
                    0.54,
                    0.58,
                    0.62,
                ],  # Land use MSA for Vertebrates in the same order as land_use_type
            ]
        ),
        coords=dict(
            land_use_type=msa_dataset["land_use_type"],
            specie=["Plants", "Vertebrates"],
        ),
        dims=["specie", "land_use_type"],
    ).sel(specie=parameters.species_names)

    # Calculating Land Use MSA based on ResponseRelationships_GLOBIO4.xlsx
    msa_dataset["msa_lu"] = msa_dataset["share"] * msa_dataset["input_msa"]
    msa_dataset["msa_lu_max"] = msa_dataset["share"] * \
        xr.ones_like(msa_dataset["input_msa"])
    msa_dataset["msa_lu_food_feed_crop"] = msa_dataset["msa_lu"].where(msa_dataset["msa_lu"].land_use_type.isin(
        ["Food/feed cropland - Intense use", "Food/feed cropland - Minimal use"]), msa_dataset["msa_lu_max"]).sum("land_use_type")
    msa_dataset["msa_lu_pasture"] = msa_dataset["msa_lu"].where(msa_dataset["msa_lu"].land_use_type.isin(
        ["Pasture - Intense use", "Pasture - Minimal use"]), msa_dataset["msa_lu_max"]).sum("land_use_type")
    msa_dataset["msa_lu_bioenergy"] = msa_dataset["msa_lu"].where(msa_dataset["msa_lu"].land_use_type.isin(
        ["Bioenergy cropland - Intense use", "Bioenergy cropland - Minimal use", "Bioenergy plantation"]), msa_dataset["msa_lu_max"]).sum("land_use_type")
    msa_dataset["msa_lu_carbon_plantation"] = msa_dataset["msa_lu"].where(
        msa_dataset["msa_lu"].land_use_type.isin(["Carbon plantation"]), msa_dataset["msa_lu_max"]).sum("land_use_type")
    msa_dataset["msa_lu_other"] = msa_dataset["msa_lu"].where(msa_dataset["msa_lu"].land_use_type.isin(
        ["Other plantation", "Secondary vegetation", "Urban", "Natural land"]), msa_dataset["msa_lu_max"]).sum("land_use_type")

    msa_dataset["msa_lu_total"] = (
        msa_dataset["msa_lu_food_feed_crop"]
        * msa_dataset["msa_lu_pasture"]
        * msa_dataset["msa_lu_bioenergy"]
        * msa_dataset["msa_lu_carbon_plantation"]
        * msa_dataset["msa_lu_other"]
    )
    msa_dataset["msa_lu_sum"] = msa_dataset["msa_lu"].sum("land_use_type")

    # Saving Netcdf file
    # output_dataset.to_netcdf(result_file)
    msa_dataset[["msa_lu_sum", "msa_lu_food_feed_crop", "msa_lu_pasture", "msa_lu_bioenergy",
                 "msa_lu_carbon_plantation", "msa_lu_other", "msa_lu_total"]].to_netcdf(result_file)


def compute_msa_cc(data_path, parameters):
    """
    Calculating MSA Climate Change
    """

    result_file = f"{data_path.output}msa_grid_cc.nc"

    if not need_to_compute(result_file, parameters):
        return

    print(
        "    Calculating Climate Change MSA and copying temperature file to output folder"
    )

    file_list = [f"{data_path.output}share.nc"]
    msa_dataset = load_dataset(file_list, data_path)
    path_to_temp = os.path.join(
        data_path.scenario_input, "..", "output", "climate_impacts"
    )
    if Path(os.path.join(path_to_temp, "TemperatureMAGICC.OUT")).is_file():
        file_in = pym.read_mym(os.path.join(
            path_to_temp, "TemperatureMAGICC.OUT"))
        temperature_file = os.path.join(path_to_temp, "TemperatureMAGICC.OUT")
    elif Path(
        os.path.join(path_to_temp, "..", "..",
                     "I2RT/exchange", "TemperatureMAGICC.OUT")
    ).is_file():
        file_in = pym.read_mym(
            os.path.join(
                path_to_temp, "..", "..", "I2RT/exchange", "TemperatureMAGICC.OUT"
            )
        )
        temperature_file = os.path.join(
            path_to_temp, "..", "..", "I2RT/exchange", "TemperatureMAGICC.OUT"
        )
    # elif Path(os.path.join(path_to_temp, "TEMPERATURE.OUT")).is_file():
    #     file_in = pym.read_mym(os.path.join(path_to_temp, "TEMPERATURE.OUT"))
    #     temperature_file = os.path.join(path_to_temp, "TEMPERATURE.OUT")
    else:
        raise Exception(
            f"There is no temperature file for {scen} at {data_path.input}")

    with open("temperature_log.txt", "a") as f:
        f.write(
            f"{time.strftime('%H:%M:%S', time.localtime())} - {proj} - {scen} - {temperature_file}"
        )

    shutil.copyfile(
        temperature_file,
        os.path.join(data_path.output, os.path.basename(temperature_file)),
    )

    # Adjusting to make sure that GMTI is set to zero when CC is off (get_clfbopt(data_path) == 0)
    if get_clfbopt(data_path) == 0:
        # file_in[0][:, 0] = file_in[0][:, 0] * 0.0
        file_in[0][:, 0] *= 0.0
    #########

    cc_dataArray = xr.DataArray(
        file_in[0][:, 0],  # Initial value of the array
        dims=["time"],
        coords=dict(
            time=np.array(
                [datetime.date(year, 1, 1) for year in file_in[1]], dtype=np.datetime64
            )
        ),
    )

    msa_dataset["GMTI"] = cc_dataArray.sel(time=msa_dataset.time)

    # function to define msa for plants and vertebrates based on GMTI
    def temperature_to_msa(GMTI_delta, intercept, GMTI):
        return 1.0 / (1.0 + np.exp(-(intercept + GMTI * GMTI_delta)))

    param = dict(
        Plants=dict(intercept=2.86685, GMTI=-0.46705),
        Vertebrates=dict(intercept=3.21349, GMTI=-0.3622),
    )

    # Calculating msa and saving into one variable with two coordinates (plants, vertebrates) using xr.concat
    msa_dataset["msa_cc"] = xr.concat(
        [
            temperature_to_msa(
                msa_dataset["GMTI"], param[value]["intercept"], param[value]["GMTI"]
            )
            for value in parameters.species_names
        ],
        "specie",
    ).assign_coords(specie=parameters.species_names)

    # Saving msa_cc as Netcdf file
    msa_dataset[["msa_cc", "GMTI"]].to_netcdf(result_file)


def compute_msa_n(data_path, parameters):
    """
    Calculate MSA Nitrogen deposition
    """

    result_file = f"{data_path.output}msa_grid_n.nc"

    if not need_to_compute(result_file, parameters):
        return

    print("    Calculating Nitrogen deposition MSA")

    file_list = [
        f"{data_path.output}share.nc",
        f"{data_path.scenario_input}GNDEP.NC",
    ]
    msa_dataset = load_dataset(file_list, data_path)

    # function to define msa for plants and vertebrates based on GNDEP
    def nitrogen_to_msa(nitrogen):
        return 1.0 / (
            1.0
            + np.exp(-(2.24485 - 0.74762 * np.log10(nitrogen.where(nitrogen > 0.0))))
        )

    # Calculating msa
    msa_dataset["GNDEP_to_msa"] = nitrogen_to_msa(msa_dataset["GNDEP"] / 100.0)

    # Creating a new DataArray to populate it later in each coordinate
    msa_dataset["msa"] = xr.DataArray(
        np.nan,  # Initial value of the array
        coords=[  # Coordinates of the array
            msa_dataset.latitude,
            msa_dataset.longitude,
            msa_dataset.time,
            msa_dataset.land_use_type,
        ],
    )

    # saving msa into one variable for each land use type accounted for Nitrogen deposition:
    # Secondary Vegetation, Plantation, Natural Land
    for i in [4, 5, 7, 11]:  # added 11 after changes in land use types
        msa_dataset["msa"][dict(land_use_type=i)] = msa_dataset[
            "GNDEP_to_msa"
        ] * msa_dataset["share"].isel(land_use_type=i)

    # As Nitrogen deposition does not influence msa in crop, pasture and urban
    # the share of these land_use_types is multiplied by msa = 1 (highest msa)
    for i in [0, 1, 2, 3, 6, 8, 9, 10]:  # added 8, 9, 10 after changes in land use types
        msa_dataset["msa"][dict(land_use_type=i)] = 1.0 * msa_dataset["share"].isel(
            land_use_type=i
        )

    msa_n = xr.Dataset()

    # Summing msa over land_use_type and saving to one single variable
    if "Plants" in parameters.species_names:
        msa_n["Plants"] = msa_dataset["msa"].sum("land_use_type")
    if "Vertebrates" in parameters.species_names:
        msa_n["Vertebrates"] = xr.ones_like(msa_n["Plants"])

    # Saving to file - two different arrays being saved as one dataset with species coordinate to identify them
    msa_n.to_array(dim="specie", name="msa_n").to_dataset(
    ).to_netcdf(result_file)


def compute_pressure_impact(data_path, parameters):
    result_file = f"{data_path.output}pressure_impact.nc"

    if not need_to_compute(result_file, parameters):
        return

    print("    Calculating pressure impact at grid and region level")

    file_list = [
        f"{data_path.output}msa_grid_n.nc",
        f"{data_path.output}msa_grid_lu.nc",
        f"{data_path.output}msa_grid_cc.nc",
        f"{data_path.output}msa_grid.nc",
    ]

    msa_dataset = load_dataset(file_list, data_path).fillna(0.0)

    # Adjusting to make sure that the msa CC is 1 (which means zero CC impact on MSA) when CC is off (get_clfbopt(data_path) == 0)
    if get_clfbopt(data_path) == 0:
        msa_dataset["msa_cc"] *= 0.0
        msa_dataset["msa_cc"] += 1.0
    #########

    # Eq (2) from GLOBIO4 paper

    ### Original GLOBIO equation ###
    # tmp_sum_msa = (
    #     (1.0 - msa_dataset["msa_cc"])
    #     + (1.0 - msa_dataset["msa_lu_total"])
    #     + (1.0 - msa_dataset["msa_n"])
    # )

    ### Adjusted MSA tool equation to account for the decomposition of land use types ###
    tmp_sum_msa_lu = (
        (1.0 - msa_dataset["msa_lu_food_feed_crop"])
        + (1.0 - msa_dataset["msa_lu_pasture"])
        + (1.0 - msa_dataset["msa_lu_bioenergy"])
        + (1.0 - msa_dataset["msa_lu_carbon_plantation"])
        + (1.0 - msa_dataset["msa_lu_other"])
        + (1.0 - msa_dataset["msa_cc"])
        + (1.0 - msa_dataset["msa_n"])
    )

    # Calculating Eq2 in different arrays and saving them to one dataset with pressure as coordinate (concat function)
    msa_dataset["P"] = xr.concat(
        [
            (1.0 - msa_dataset[x]) *
            (1.0 - msa_dataset[msa_type]) / denominator
            for x, msa_type, denominator in [
                ("msa_cc", "msa", tmp_sum_msa_lu),
                ("msa_lu_food_feed_crop", "msa", tmp_sum_msa_lu),
                ("msa_lu_pasture", "msa", tmp_sum_msa_lu),
                ("msa_lu_bioenergy", "msa", tmp_sum_msa_lu),
                ("msa_lu_carbon_plantation", "msa", tmp_sum_msa_lu),
                ("msa_lu_other", "msa", tmp_sum_msa_lu),
                ("msa_n", "msa", tmp_sum_msa_lu),
                ("msa_lu_total", "msa", tmp_sum_msa_lu)
            ]
        ],
        "pressure",
    ).assign_coords(
        pressure=[
            "Climate change",
            "Land use - food/feed crop",
            "Land use - pasture",
            "Land use - bioenergy",
            "Land use - carbon plantation",
            "Land use - other",
            "Nitrogen deposition",
            "Land use"
        ]
    )

    return compute_regional_values(
        msa_dataset["P"].fillna(0.0), "pressure_impact", data_path
    )


def compute_overall_msa(data_path, parameters):
    """
    Calculating Overall MSA
    """

    result_file = f"{data_path.output}msa_grid.nc"

    if not need_to_compute(result_file, parameters):
        return

    print("    Calculating overall MSA at grid and region level")

    file_list = [
        f"{data_path.output}msa_grid_n.nc",
        f"{data_path.output}msa_grid_lu.nc",
        f"{data_path.output}msa_grid_cc.nc",
    ]

    msa_dataset = load_dataset(file_list, data_path).fillna(0.0)

    # Adjusting to make sure that the msa CC is 1 (which means zero CC impact on MSA) when CC is off (get_clfbopt(data_path) == 0)
    if get_clfbopt(data_path) == 0:
        msa_dataset["msa_cc"] *= 0.0
        msa_dataset["msa_cc"] += 1.0
    #########

    # Eq (1) from GLOBIO4 paper

    ### Original GLOBIO equation ###
    # Calculating overall MSA at grid level
    # msa_dataset["msa"] = (
    #     msa_dataset["msa_cc"]
    #     * msa_dataset["msa_lu_total"]
    #     * msa_dataset["msa_n"]
    # )

    ### Adjusted MSA tool equation to account for the decomposition of land use types ###
    msa_dataset["msa"] = (
        msa_dataset["msa_cc"]
        * msa_dataset["msa_lu_food_feed_crop"]
        * msa_dataset["msa_lu_pasture"]
        * msa_dataset["msa_lu_bioenergy"]
        * msa_dataset["msa_lu_carbon_plantation"]
        * msa_dataset["msa_lu_other"]
        * msa_dataset["msa_n"]
    )

    # Saving overall MSA as Netcdf
    # Save share as netcdf
    # From DataArray to Dataset (with only one variable per file) to netcdf
    msa_dataset["msa"].to_dataset().to_netcdf(result_file)

    # return msa_dataset["msa"]

    return compute_regional_values(
        msa_dataset["msa"].fillna(0.0), "msa_region", data_path
    )


def compute_regional_values(dataArray, file_prefix, data_path):
    greg_flags = {
        "Low income regions": {7, 8, 9, 18, 21, 22, 25, 26},
        "Medium income regions": {3, 4, 5, 6, 10, 12, 13, 14, 15, 17, 20},
        "High income regions": {1, 2, 11, 16, 19, 23, 24},
    }

    greg_flags["Low-medium income regions"] = (
        greg_flags["Low income regions"] | greg_flags["Medium income regions"]
    )

    compute_regional_values_helper(
        dataArray=dataArray,
        file_prefix=file_prefix,
        data_path=data_path,
        region_mapper_path=f"{data_path.area_region_input}GREG.NC",
        array_name="GREG",
        output_prefix="greg",
        region_name_mapper={  # IMAGE Regions
            1: "Canada",
            2: "USA",
            3: "Mexico",
            4: "Rest Central America",
            5: "Brazil",
            6: "Rest South America",
            7: "Northern Africa",
            8: "Western Africa",
            9: "Eastern Africa",
            10: "Southern Africa",
            11: "OECD Europe",
            12: "Eastern Europe",
            13: "Turkey",
            14: "Ukraine +",
            15: "Asia-Stan",
            16: "Russia +",
            17: "Middle East",
            18: "India +",
            19: "Korea",
            20: "China +",
            21: "Southeastern Asia",
            22: "Indonesia +",
            23: "Japan",
            24: "Oceania",
            25: "Rest S.Asia",
            26: "Rest S.Africa",
        },
        flags=greg_flags,
    )

    compute_regional_values_helper(
        dataArray=dataArray,
        file_prefix=file_prefix,
        data_path=data_path,
        region_mapper_path=f"{data_path.area_region_input}GPOTVEG.nc",
        array_name="GPOTVEG",
        output_prefix="gpotveg",
        region_name_mapper={
            # 1: "agricultural land",
            # 2: "extensive grassland",
            # 3: "carbon plantation",
            # 4: "regrowth forest abandoning",
            # 5: "regrowth forest timber",
            # 6: "biofuels",
            7: "ice",
            8: "tundra",
            9: "wooded tundra",
            10: "Boreal forest",
            11: "Cool conifer forest",
            12: "Temp. mixed forest",
            13: "Temp. decid. forest",
            14: "Warm mixed forest",
            15: "grassland/steppe",
            16: "hot desert",
            17: "scrubland",
            18: "savanna",
            19: "tropical woodland",
            20: "tropical forest",
        },
        flags={
            "Trop. forest/woodland": {19, 20},
            "Grass./scrub./savanna": {15, 17, 18},
            "Grass./scrub.": {15, 17},
            "Scrub./savanna": {17, 18},
            "Grass./savanna": {15, 18},
        },
    )


def compute_regional_values_helper(
    dataArray,
    file_prefix,
    data_path,
    region_mapper_path,
    array_name,
    region_name_mapper,
    flags,
    output_prefix,
):
    file_list = [
        f"{data_path.output}share.nc",
        f"{data_path.area_region_input}GAreaCellNoWater.NC",
        region_mapper_path,
    ]

    msa_dataset = load_dataset(file_list, data_path).fillna(0.0)

    msa_region = xr.Dataset()

    # Calculating Region level MSA
    # IMAGE Regions
    for number, name in region_name_mapper.items():
        msa_region[name] = (
            dataArray.where(msa_dataset[array_name] == number)
            .weighted(msa_dataset["GAreaCellNoWater"].fillna(0.0))
            .mean(["longitude", "latitude"])
        )

    # Calculating agregated regions
    for name, numbers in flags.items():
        msa_region[name] = (
            dataArray.where(msa_dataset[array_name].isin(list(numbers)))
            .weighted(msa_dataset["GAreaCellNoWater"].fillna(0.0))
            .mean(["longitude", "latitude"])
        )

    if array_name == "GREG":
        # World
        msa_region["World"] = dataArray.weighted(
            msa_dataset["GAreaCellNoWater"].fillna(0)
        ).mean(["longitude", "latitude"])

        TROPICAL_REF = 23.43634

        # Tropical region
        tropical_flag = (msa_dataset.latitude <= TROPICAL_REF) & (
            msa_dataset.latitude >= (-TROPICAL_REF)
        )
        msa_region["Tropical region"] = (
            dataArray.where(tropical_flag)
            .weighted(msa_dataset["GAreaCellNoWater"].fillna(0))
            .mean(["longitude", "latitude"])
        )

        # Subtropical region
        subtropical_flag = (msa_dataset.latitude > TROPICAL_REF) | (
            msa_dataset.latitude < (-TROPICAL_REF)
        )
        msa_region["Subtropical region"] = (
            dataArray.where(subtropical_flag)
            .weighted(msa_dataset["GAreaCellNoWater"].fillna(0))
            .mean(["longitude", "latitude"])
        )

    msa_region = msa_region.to_array(dim="region", name=file_prefix)

    for specie in msa_region.specie.data:
        # Save as csv for species
        dataframe = (
            msa_region.sel(specie=specie).drop(
                "specie").to_dataframe().unstack("time")
        )
        dataframe.to_csv(
            f"{data_path.output}{output_prefix}_{file_prefix}_{specie.lower()}.csv",
            header=msa_region.time.dt.year.data,
        )

    msa_region.to_dataset().to_netcdf(
        f"{data_path.output}{output_prefix}_{file_prefix}.nc"
    )

    return msa_region


def make_figures(data_path, parameters):
    for var in ["greg", "gpotveg"]:
        file_list = [
            f"{data_path.output}share.nc",
            f"{data_path.output}msa_grid.nc",
            f"{data_path.output}msa_grid_cc.nc",
            f"{data_path.output}{var}_pressure_impact.nc",
            f"{data_path.output}{var}_msa_region.nc",
            f"{data_path.area_region_input}GAreaCellNoWater.NC",
        ]

        dataset = load_dataset(file_list, data_path).fillna(0.0)

        # Bar charts - pressures and MSA - region level
        for time in range(dataset.time.size):
            for specie in range(-1, dataset.specie.size):
                if specie == -1 and dataset.specie.size == 1:
                    continue

                if specie == -1:
                    ds = (
                        dataset.isel(time=time)
                        .mean("specie")
                        .assign_coords(specie="Plants and vertebrates average")
                    )
                else:
                    ds = dataset.isel(specie=specie, time=time)

                # ds = ds.sortby(ds.pressure_impact.sum("pressure")) #  ds = ds.sortby(1 - ds.pressure_impact.sum("pressure")) # commented out to have the default sort of IMAGE Regions instead of sorting by msa value

                with plt.style.context("seaborn-talk"):
                    fig, ax = plt.subplots()

                    bottom = xr.zeros_like(ds.msa_region)

                    plt.bar(
                        ds.region, ds.msa_region, label="MSA region", bottom=bottom
                    )  # could have been ax.bar
                    bottom += ds.msa_region

                    for p in ds.pressure[::-1]:
                        plt.bar(
                            ds.region,
                            ds.pressure_impact.sel(pressure=p),
                            label=f"{p.data}",
                            bottom=bottom,
                        )
                        bottom += ds.pressure_impact.sel(pressure=p)

                    title = f"{ds.time.dt.year.data}, {ds.specie.data}"
                    ax.set_title(title)

                    ax.legend(
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        title="MSA & Pressures",
                    )
                    ax.xaxis.set_ticks(ds.region.data)
                    ax.tick_params(axis="x", labelrotation=90)
                    plt.tight_layout()
                    fig.savefig(
                        f"{data_path.output}{var}_pressure_and_msa_{title}.pdf")

        plt.close("all")

        # Diff Maps t_final - t_initial
        for specie in range(-1, dataset.specie.size):
            if specie == -1 and dataset.specie.size == 1:
                continue

            if specie == -1:
                ds = dataset.mean("specie").assign_coords(
                    specie="plants and vertebrates average"
                )
            else:
                ds = dataset.isel(specie=specie)

            with plt.style.context("seaborn-talk"):
                da = ds["msa"].isel(time=-1) - ds["msa"].isel(time=0)

                da.fillna(0.0).where(ds.GAreaCellNoWater > 0.0).plot(
                    rasterized=True,
                    x="longitude",
                    y="latitude",
                    vmin=-0.3,
                    vmax=0.3,
                    levels=9,
                    cmap="PiYG",
                )
                title = f"Change in {ds.specie.data} MSA ({dataset.isel(time=-1).time.dt.year.data}-{dataset.isel(time=0).time.dt.year.data})"
                plt.title(title)

                plt.savefig(f"{data_path.output}{var}_{title}.png", dpi=200)
                # plt.show()
                plt.close("all")


def get_paths(input=""):
    scen_name = os.path.basename(input)
    project_name = os.path.basename(os.path.dirname(input))
    return dict(
        scenario_input=os.path.join(input, "netcdf", ""),
        output=os.path.join(project_name, scen_name, ""),
        area_region_input=AREA_REGION_INPUT,
    )


def create_output_folder(data_path, parameters):
    os.makedirs(data_path.output, exist_ok=True)


def need_to_compute(filename, parameters):
    if os.path.isfile(filename) and not parameters.force_compute_files:
        return False
    else:
        return True


def get_clfbopt(data_path):
    filename = os.path.join(data_path.scenario_input,
                            "..", "dat", "options.dat")

    if os.path.isfile(filename):
        with open(filename, "r") as f:
            file_content = "[section]\n" + f.read()

        config = configparser.ConfigParser()
        config.read_string(file_content)

        return config.getint("section", "clfbopt")
    else:
        return 1


def clear_cache(data_path):
    file_list = [
        f"{data_path.output}share.nc",
        f"{data_path.output}msa_grid.nc",
        f"{data_path.output}msa_grid_cc.nc",
        f"{data_path.output}msa_grid_lu.nc",
        f"{data_path.output}msa_grid_n.nc",
    ]

    for file in file_list:
        os.remove(file)


def get_scenario_properties(data_path, scen, df):
    data = {"name": scen}
    data.update(**{col: df[col][scen] for col in df.columns})

    with open(f"{data_path.output}info.json", "w") as f:
        json.dump(df.loc[scen].to_json(), f, ensure_ascii=True, indent=4)

    data_path._gbuiltup_path = (
        f"{data_path.area_region_input}{data['SSPurban_IMAGEversion']}/GBUILTUP.NC"
    )

    return data_path


def get_excel():
    df = pd.read_excel(
        f"{AREA_REGION_INPUT}project_scen_features_msa_gmti.xlsx",
        sheet_name="scen_attributes",
        index_col="Scen name",
        usecols=[
            "Scen name",
            "Project",
            "SSPurban_IMAGEversion",
            "CC impacts",
            "2 degree GMTI",
            "30% PAs",
            "Sustainable Demand",
            "Sustainable Supply",
        ],
        dtype={"CC mitigation": bool},
    )

    return df


def get_proj_scen(df):
    project_names = list(set(df["Project"]))
    scen_names = list(set(df.index))
    # scen_names = list(set(df.index))

    return project_names, scen_names


if __name__ == "__main__":
    parameters = Parameters()
    selected_years = parameters.years

    df = get_excel()
    list_project_excel, list_scen_excel = get_proj_scen(df)

    if parameters.list_of_projects and parameters.list_of_scen:
        project_runs = parameters.list_of_projects
        scen_runs = parameters.list_of_scen

    else:
        project_runs = list_project_excel
        scen_runs = list_scen_excel

    for proj in project_runs:
        print(f"########## {proj} ##########")

        for scen in scen_runs:
            path_in = os.path.join(parameters.input_path, proj, scen)

            if not os.path.exists(path_in):
                continue

            try:
                print(
                    f"========== Starting {scen} at",
                    time.strftime("%H:%M:%S", time.localtime()),
                    "==========",
                )

                data_path = DataPath(path_in)
                create_output_folder(data_path, parameters)
                data_path = get_scenario_properties(data_path, scen, df)
                print(f"{scen} properties were taken from the Excel input file!")
                compute_share(data_path, parameters)
                compute_msa_lu(data_path, parameters)
                compute_msa_cc(data_path, parameters)
                compute_msa_n(data_path, parameters)
                compute_overall_msa(data_path, parameters)
                compute_pressure_impact(data_path, parameters)

                if parameters.make_figures:
                    print("    Making figures")
                    make_figures(data_path, parameters)

                if parameters.clear_cache:
                    clear_cache(data_path)

            except Exception as e:
                # print(f"exception found: {e}")
                print("".join(tb.format_exception(None, e, e.__traceback__)))

    end_time = time.time()
    print("Finished script at", time.strftime("%H:%M:%S", time.localtime()))
    print(f"Elapsed time: {(end_time - start_time)/60.0:.2f} min")
