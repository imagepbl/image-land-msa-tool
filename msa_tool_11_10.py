import time
start_time = time.time()
print("Starting script at", time.strftime("%H:%M:%S", time.localtime()))

print("Importing packages")
"""
Importing required libraries
"""
import datetime
import configparser
import os
import os.path
import dask
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr
import pym

class DataPath:
    '''
    Class based on the function get_paths from previous msa_tool version.
    Deals with paths for input files from scenario folder, input files from working directory and output folder.
    '''

    def __init__(self, input):
        scen_name = os.path.basename(input)
        project_name = os.path.basename(os.path.dirname(input))

        self._scenario_input = os.path.join(input, "netcdf", "")
        self._output=os.path.join(project_name, scen_name, "")
        self._area_region_input=os.path.join("input", "")
    
    @property
    def scenario_input(self):
        return self._scenario_input
    @property
    def output(self):
        return self._output
    @property
    def area_region_input(self):
        return self._area_region_input

class Parameters:
    '''
    Class based on the function load_parameters from previous msa_tool version.
    Deals with parameters in the file parameters.ini
    '''

    def __init__(self, filename="parameters.ini"):

        config = configparser.ConfigParser()
        config.read(filename)

        self._input_path=config["MSA_TOOL"]["input_path"]
        self._list_of_scen=[
            scen.lstrip() for scen in config["MSA_TOOL"]["list_of_scen"][1:-1].split(",")
        ]
        self._years=[int(year) for year in config["MSA_TOOL"]["years"][1:-1].split(",")]
        self._make_figures=config.getboolean("MSA_TOOL", "make_figures")
        self._force_compute_files=config.getboolean("MSA_TOOL", "force_compute_files")

        # self._get_clfbopt(self)

    # def _get_clfbopt(self):
    #     filename = os.path.join(data_path.get("scenario_input"), "..", "dat", "options.dat")

    #     self._clfbopt = 0

    #     if os.path.isfile(filename):

    #         with open(filename, "r") as f:
    #             file_content = "[section]\n" + f.read()

    #         config = configparser.ConfigParser()
    #         config.read_string(file_content)
        
    #         self._clfbopt = config.getint("section", "clfbopt")

    @property
    def input_path(self):
        return self._input_path
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
    # @property
    # def clfbopt(self):
    #     return self._clfbopt

print("Loading functions")
selected_years = []

def load_dataset(file_list: list, data_path, **kwargs_sel):
    """
    Function to load multiple datasets at once with chunks in time to optimize memory usage
    """

    def pre_processing(dataset):
        """
        Function to harmonize coordinates among datasets and select timesteps
        """
        if "lat" in dataset.coords:
            dataset = dataset.rename(dict(lat="latitude"))
        if "lon" in dataset.coords:
            dataset = dataset.rename(dict(lon="longitude"))
        # it is not possible to run a case selecting just one time step - SEE WARNING IN parameters.ini
        if "time" in dataset.coords:
            if dataset.time.size == 1:
                dataset = dataset.squeeze().drop("time")
        if "latitude" in dataset.coords and "longitude" in dataset.coords:
            if dataset.latitude.dtype != "float32" or dataset.longitude.dtype != "float32":
                ref_for_coords = f"{data_path.scenario_input}GLCT.NC"
                ref_dataset = xr.open_dataset(ref_for_coords, engine="netcdf4")
                dataset = dataset.assign_coords(
                    latitude=ref_dataset.latitude.astype(np.float32).data,
                    longitude=ref_dataset.longitude.astype(np.float32).data,
                )
                ref_dataset.close()
        return dataset.transpose(*sorted(dataset.dims))

    if not kwargs_sel and selected_years:
        kwargs_sel["time"] = [datetime.date(year, 1, 1) for year in selected_years]

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        msa_dataset = xr.open_mfdataset(
            paths=file_list,
            chunks=dict(time="auto", latitude="auto", longitude="auto"),
            #chunks=dict(time="auto"),
            join="right",
            preprocess=pre_processing,
            engine="netcdf4",
        ).sel(**kwargs_sel)
    return msa_dataset

land_use_type_names = [  # Land use types based on GLOBIO4 classification
    "Cropland - Intense use",
    "Cropland - Minimal use",
    "Pasture - Intense use",
    "Pasture - Minimal use",
    "Plantation",
    "Secondary vegetation",
    "Urban",
    "Natural land",
]

region_names = {  # IMAGE Regions
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
    27: "Low income regions",
    28: "Medium income regions",
    29: "High income regions",
    30: "World",
}


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
        f"{data_path.area_region_input}GBUILTUP.NC", #GBUILTUP.NC is not available in Biodiv_Post2020 runs. 
        # For Biodiv_Post2020, set area_region_input instead of scenario_input, and you will use a GBUILTUP file for SSP2 which is in the folder input
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
        0.0,  # Initial value of the array
        coords=[  # Coordinates of the array
            msa_dataset.latitude,
            msa_dataset.longitude,
            msa_dataset.time,
            msa_dataset.land_use_type,
        ],
    ).chunk(
        dict(
            time="auto",
            latitude="auto",
            longitude="auto"
        )
    )
    msa_dataset["area"] = xr.DataArray(
        0.0,  # Initial value of the array
        coords=[  # Coordinates of the array
            msa_dataset.latitude,
            msa_dataset.longitude,
            msa_dataset.time,
            msa_dataset.land_use_type,
        ],
    ).chunk(
        dict(
            time="auto",
            latitude="auto",
            longitude="auto"
        )
    )

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
    msa_dataset["area"][dict(land_use_type=i)] = (
        msa_dataset["GFRAC"].sel(
            NGFBFC=b"grass                                             "
        )
        * msa_dataset["non_urban_area"]
    ).where(msa_dataset["GLCT"] == 1)
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Pasture - Minimal use ###
    i = 3  # Pasture - Minimal use - GFRAC must be "grass", GLCT must be 2
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = (
        msa_dataset["GFRAC"].sel(
            NGFBFC=b"grass                                             "
        )
        * msa_dataset["non_urban_area"]
    ).where(msa_dataset["GLCT"] == 2)
    # Share
    msa_dataset["share"][dict(land_use_type=i)] = msa_dataset["area"].isel(
        land_use_type=i
    ) / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Crop ###
    ### Share of all cropland to be later divided between intense and minimal ###
    # Loop to get the sum of GFRAC related to cropland share
    NGFBFC_list = [
        i for i in range(msa_dataset.NGFBFC.size) if i not in [0, 13, 20, 34]
    ]
    msa_dataset["non_urban_crop_share"] = (
        msa_dataset["GFRAC"].isel(NGFBFC=NGFBFC_list).fillna(0.0).sum("NGFBFC")
    )

    ### Crop - Intense use ###
    i = 0  # Crop - Intense use
    # GLCT must be 1 OR 6, GFRAC "non_urban_crop_share" must exist, GFERTILIZER must be > 10
    # Conditions
    GFRAC = (msa_dataset["non_urban_crop_share"] > 0) & (
        msa_dataset["GFERTILIZER"].isel(NFERTSMT=0) > 10.0
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

    ### Crop - Minimal use ###
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

    ### Plantation ###
    ### Share and area of plantation from GFRAC to be later added to plantation from GLCT ###
    # Loop to get the sum of GFRAC related to plantation share
    msa_dataset["non_urban_plantation_share"] = xr.zeros_like(
        msa_dataset["non_urban_area"]
    )
    # Plantation (share GFRAC)
    for specific_plantation in {14 - 1, 21 - 1, 35 - 1}:
        msa_dataset["non_urban_plantation_share"] += msa_dataset["GFRAC"].isel(
            NGFBFC=specific_plantation
        ).fillna(0.0)
    # Plantation (area GFRAC)
    msa_dataset["plantation_area_gfrac"] = (
        msa_dataset["non_urban_area"] * msa_dataset["non_urban_plantation_share"]
    ).where(msa_dataset["non_urban_plantation_share"] > 0.0)

    ### Plantation ###
    i = 4  # Plantation - conditions for part 1 and 2 below
    # Share
    # Part 1 - GLCT must be 4 OR 5, GFORMAN must be 3
    msa_dataset["share"][dict(land_use_type=i)] = (
        xr.where(  # Part 1 (from GLCT and GFRAC)
            np.logical_and(
                np.logical_or(msa_dataset["GLCT"] == 4, msa_dataset["GLCT"] == 5),
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
        / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0.0, 0.0)
    ).fillna(0.0)
    # Add fillna to solve the problem with plantation 2021-10-07T17:15:31.011Z
    # Area
    msa_dataset["area"][dict(land_use_type=i)] = msa_dataset["share"].isel(
        land_use_type=i
    ) * msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)

    ### Secondary Vegetation ###
    i = 5  # Secondary Vegetation - conditions for part 1 and 2 below
    # Share
    # Part 1 - GLCT has to be 4 OR 5, GFORMAN has to be DIFFERENT from 3
    msa_dataset["share"][dict(land_use_type=i)] = (
        xr.where(  # Part 1 -> This part has data!
            np.logical_and(
                np.logical_or(msa_dataset["GLCT"] == 4, msa_dataset["GLCT"] == 5),
                msa_dataset["GFORMAN"].isel(NFORMID=0).data != 3,
            ),  # condition
            (
                (
                    msa_dataset["non_urban_area"]
                    - msa_dataset["plantation_area_gfrac"].fillna(0.0)
                )
                / msa_dataset["GAreaCellNoWater"].where(
                    msa_dataset["GAreaCellNoWater"] > 0
                )
            ),  # Where True
            0.0,  # np.nan, # where false
        )
        # Part 2 - GLCT has to be 3
    ) + (
        (
            (
                msa_dataset["non_urban_area"]
                - msa_dataset["plantation_area_gfrac"].fillna(0.0)
            )
            / msa_dataset["GAreaCellNoWater"].where(msa_dataset["GAreaCellNoWater"] > 0)
        ).where(msa_dataset["GLCT"] == 3, 0.0)
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
    species_names = ["Plants", "Vertebrates"]
    msa_dataset = load_dataset(file_list, data_path).assign_coords(specie=species_names)
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
                ],  # Land use MSA for Vertebrates in the same order as land_use_type
            ]
        ),
        coords=dict(
            land_use_type=msa_dataset["land_use_type"],specie=msa_dataset["specie"],
        ),
        dims=["specie", "land_use_type"],
    )

    # Calculating Land Use MSA based on ResponseRelationships_GLOBIO4.xlsx
    msa_dataset["msa_lu"] = (msa_dataset["share"] * msa_dataset["input_msa"]).sum(
        "land_use_type"
    )
    # Saving msa_lu as Netcdf file
    msa_dataset["msa_lu"].to_dataset().to_netcdf(result_file)


def compute_msa_cc(data_path, parameters):
    """
    Calculating MSA Climate Change
    """

    result_file = f"{data_path.output}msa_grid_cc.nc"

    if not need_to_compute(result_file, parameters):
        return
    
    print("    Calculating Climate Change MSA")

    file_list = [f"{data_path.output}share.nc"]
    msa_dataset = load_dataset(file_list, data_path)
    file_in = pym.read_mym(
        os.path.join(
            data_path.scenario_input,
            "..",
            "output",
            "climate_impacts",
            "TEMPERATURE.OUT",
        )
    )
    ######### Ajustes para plotar sem a temperatura
    if get_clfbopt(data_path) == 0:
        #file_in[0][:, 0] = file_in[0][:, 0] * 0.0
        file_in[0][:, 0] *= 0.0
    #########
    # try using np.squeeze to make code more readable
    # review later for better understanding
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
            temperature_to_msa(msa_dataset["GMTI"], value["intercept"], value["GMTI"])
            for value in param.values()
        ],
        "specie",
    ).assign_coords(specie=list(param.keys()))

    # Saving msa_cc as Netcdf file
    msa_dataset["msa_cc"].to_dataset().to_netcdf(result_file)


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
            # msa_dataset.specie,
        ],
    )  

    # saving msa into one variable for each land use type accounted for Nitrogen deposition:
    # Secondary Vegetation, Plantation, Natural Land
    for i in [4, 5, 7]:
        msa_dataset["msa"][dict(land_use_type=i)] = msa_dataset[
            "GNDEP_to_msa"
        ] * msa_dataset["share"].isel(land_use_type=i)

    # As Nitrogen deposition does not influence msa in crop, pasture and urban
    # the share of these land_use_types is multiplied by msa = 1 (highest msa)
    for i in [0, 1, 2, 3, 6]:
        msa_dataset["msa"][dict(land_use_type=i)] = 1.0 * msa_dataset["share"].isel(
            land_use_type=i
        )

    msa_n = xr.Dataset()

    # Summing msa over land_use_type and saving to one single variable
    msa_n["Plants"] = msa_dataset["msa"].sum("land_use_type")
    msa_n["Vertebrates"] = xr.ones_like(msa_n["Plants"])

    # Saving to file - two different arrays beeing saved as one dataset with species coordinate to identify them
    msa_n.to_array(dim="specie", name="msa_n").to_dataset().to_netcdf(
        result_file
    )


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

    ######### Ajustes para plotar sem a temperatura
    if get_clfbopt(data_path) == 0:
        msa_dataset["msa_cc"] *= 0.0
        msa_dataset["msa_cc"] += 1.0
    #########

    ### Eq (2) from GLOBIO4 paper
    tmp_sum_msa = (
        (1.0 - msa_dataset["msa_cc"])
        + (1.0 - msa_dataset["msa_lu"])
        + (1.0 - msa_dataset["msa_n"])
    )
    #Calculating Eq2 in different arrays and saving them to one dataset with pressure as coordinate (concat function)
    msa_dataset["P"] = xr.concat(
        [
            (1.0 - msa_dataset[x]) * (1.0 - msa_dataset["msa"]) / tmp_sum_msa
            for x in ["msa_cc", "msa_lu", "msa_n"]
        ],
        "pressure",
    ).assign_coords(pressure=["Climate change", "Land use", "Nitrogen deposition"])

    # return msa_dataset["P"]

    return compute_regional_values(msa_dataset["P"].fillna(0.0), "pressure_impact", data_path)


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

    ######### Ajustes para plotar sem a temperatura
    if get_clfbopt(data_path) == 0:
        msa_dataset["msa_cc"] *= 0.0
        msa_dataset["msa_cc"] += 1.0
    #########

    # Eq (1) from GLOBIO4 paper
    # Calculating overall MSA at grid level
    msa_dataset["msa"] = (
        msa_dataset["msa_cc"] * msa_dataset["msa_lu"] * msa_dataset["msa_n"]
    )

    # Saving overall MSA as Netcdf
    # Save share as netcdf
    # From DataArray to Dataset (with only one variable per file) to netcdf
    msa_dataset["msa"].to_dataset().to_netcdf(result_file)

    # return msa_dataset["msa"]

    return compute_regional_values(msa_dataset["msa"].fillna(0.0), "msa_region", data_path)


def compute_regional_values(dataArray, file_prefix, data_path):

    file_list = [
        f"{data_path.output}share.nc",
        f"{data_path.area_region_input}GREG.NC",
        f"{data_path.area_region_input}GAreaCellNoWater.NC",
    ]

    msa_dataset = load_dataset(file_list, data_path).fillna(0.0)

    msa_dataset = msa_dataset.assign_coords(
        specie=dataArray.specie, region=list(region_names.values())
    )

    # Adding development level as boolean in coordinates
    low_income_flag = xr.zeros_like(msa_dataset.GREG, dtype=bool)
    for i in (7, 8, 9, 18, 21, 22, 25, 26):
        low_income_flag = xr.where(
            msa_dataset.GREG == i,  # Condition
            True,  # Where True
            low_income_flag,  # Where False
        )

    medium_income_flag = xr.zeros_like(msa_dataset.GREG, dtype=bool)
    for i in (3, 4, 5, 6, 10, 12, 13, 14, 15, 17, 20):
        medium_income_flag = xr.where(
            msa_dataset.GREG == i,  # Condition
            True,  # Where True
            medium_income_flag,  # Where False
        )

    high_income_flag = xr.zeros_like(msa_dataset.GREG, dtype=bool)
    for i in (1, 2, 11, 16, 19, 23, 24):
        high_income_flag = xr.where(
            msa_dataset.GREG == i,  # Condition
            True,  # Where True
            high_income_flag,  # Where False
        )

    msa_region = xr.Dataset()

    # Calculating Region level MSA
    # IMAGE Regions
    for i in range(26):
        msa_region[region_names[i + 1]] = (
            dataArray.where(msa_dataset["GREG"] == i + 1)
            .weighted(msa_dataset["GAreaCellNoWater"].fillna(0.0))
            .mean(["longitude", "latitude"])
        )

    # Low income regions
    msa_region[region_names[27]] = (
        dataArray.where(low_income_flag)
        .weighted(msa_dataset["GAreaCellNoWater"].fillna(0))
        .mean(["longitude", "latitude"])
    )

    # Medium income regions
    msa_region[region_names[28]] = (
        dataArray.where(medium_income_flag)
        .weighted(msa_dataset["GAreaCellNoWater"].fillna(0))
        .mean(["longitude", "latitude"])
    )

    # High income regions
    msa_region[region_names[29]] = (
        dataArray.where(high_income_flag)
        .weighted(msa_dataset["GAreaCellNoWater"].fillna(0))
        .mean(["longitude", "latitude"])
    )

    # World
    msa_region[region_names[30]] = dataArray.weighted(
        msa_dataset["GAreaCellNoWater"].fillna(0)
    ).mean(["longitude", "latitude"])

    msa_region = msa_region.to_array(dim="region", name=file_prefix)

    # Save as csv for species = plants
    dataframe = msa_region.isel(specie=0).drop("specie").to_dataframe().unstack("time")
    dataframe.to_csv(
        f"{data_path.output}{file_prefix}_plants.csv",
        header=msa_dataset.time.dt.year.data,
    )

    # Save as csv for species = vertebrates
    dataframe = msa_region.isel(specie=-1).drop("specie").to_dataframe().unstack("time")
    dataframe.to_csv(
        f"{data_path.output}{file_prefix}_vertebrates.csv",
        header=msa_dataset.time.dt.year.data,
    )

    msa_region.to_dataset().to_netcdf(f"{data_path.output}{file_prefix}.nc")

    return msa_region


def make_figures(data_path, parameters):
    file_list = [
        f"{data_path.output}share.nc",
        f"{data_path.output}msa_grid.nc",
        f"{data_path.output}msa_grid_cc.nc",
        f"{data_path.output}pressure_impact.nc",
        f"{data_path.output}msa_region.nc",
        f"{data_path.area_region_input}GAreaCellNoWater.NC",
    ]

    dataset = load_dataset(file_list, data_path).fillna(0.0)

    # Bar charts - pressures and MSA - region level
    ######### Setting for ploting CC pressure as pressure or add it to the MSA (as requested for Navigate)
    # if get_clfbopt(data_path) == 0:
    #     dataset["msa_region"] = dataset["msa_region"] + dataset["pressure_impact"].sel(
    #         pressure="Climate change", drop=True
    #     )

    #     dataset = dataset.sel(pressure=["Land use", "Nitrogen deposition"])
    #########

    for time in range(dataset.time.size):
        for specie in range(-1, dataset.specie.size):

            if specie == -1:
                ds = (
                    dataset.isel(time=time)
                    .mean("specie")
                    .assign_coords(specie="Plants and vertebrates average")
                )
            else:
                ds = dataset.isel(specie=specie, time=time)

            ds = ds.sortby(ds.pressure_impact.sum("pressure")) #  ds = ds.sortby(1 - ds.pressure_impact.sum("pressure"))
            
            with plt.style.context("seaborn-talk"):
            
                fig, ax = plt.subplots()

                bottom = xr.zeros_like(ds.msa_region)

                plt.bar(ds.region, ds.msa_region, label="MSA region", bottom=bottom) #could have been ax.bar
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

                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = "MSA & Pressures")
                ax.xaxis.set_ticks(ds.region.data)
                ax.tick_params(axis='x', labelrotation=90)
                plt.tight_layout()
                fig.savefig(f"{data_path.output}pressure_and_msa_{title}.pdf")
    plt.close("all")

    # Diff Maps t_final - t_initial
    for specie in range(-1, dataset.specie.size):

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

            plt.savefig(f"{data_path.output}{title}.png", dpi=200)
            #plt.show()
            plt.close("all")


def get_paths(input=""):
    scen_name = os.path.basename(input)
    project_name = os.path.basename(os.path.dirname(input))
    return dict(
        scenario_input=os.path.join(input, "netcdf", ""),
        output=os.path.join(project_name, scen_name, ""),
        area_region_input=os.path.join("input", ""),
    )


def create_output_folder(data_path, parameters):
    os.makedirs(data_path.output, exist_ok=True)


def load_parameters(filename="parameters.ini"):
    config = configparser.ConfigParser()
    config.read(filename)
    return dict(
        input_path=config["MSA_TOOL"]["input_path"],
        list_of_scen=[
            scen.lstrip() for scen in config["MSA_TOOL"]["list_of_scen"][1:-1].split(",")
        ],
        years=[int(year) for year in config["MSA_TOOL"]["years"][1:-1].split(",")],
        make_figures=config.getboolean("MSA_TOOL", "make_figures"),
        force_compute_files=config.getboolean("MSA_TOOL", "force_compute_files"),
    )

def need_to_compute(filename, parameters):
    if os.path.isfile(filename) and not parameters.force_compute_files:
        return False
    else:
        return True

def get_clfbopt(data_path):

    filename = os.path.join(data_path.scenario_input, "..", "dat", "options.dat")

    if os.path.isfile(filename):

        with open(filename, "r") as f:
            file_content = "[section]\n" + f.read()

        config = configparser.ConfigParser()
        config.read_string(file_content)
    
        return config.getint("section", "clfbopt")
    else:
        return 1

if __name__ == "__main__":

    parameters = Parameters()
    selected_years = parameters.years

    for scen in tqdm.tqdm(parameters.list_of_scen):

        print(f"========== {scen} ==========")

        path_in = os.path.join(parameters.input_path, scen)
        data_path = DataPath(path_in)
        
        create_output_folder(data_path, parameters)
        compute_share(data_path, parameters)
        compute_msa_lu(data_path, parameters)
        compute_msa_cc(data_path, parameters)
        compute_msa_n(data_path, parameters)
        compute_overall_msa(data_path, parameters)
        compute_pressure_impact(data_path, parameters)

        if parameters.make_figures:
            print("    Making figures")
            make_figures(data_path, parameters)

    end_time = time.time()
    print("Finished script at", time.strftime("%H:%M:%S", time.localtime()))
    print(f"Elapsed time: {(end_time - start_time)/60.0:.2f} min")