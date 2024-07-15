import configparser
import os.path

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

'''
IMAGE-Land MSA Validation

Data compared to original GLOBIO MSA as presented in the paper "Projecting terrestrial biodiversity intactness with GLOBIO 4"
Data for validation: X:/IMAGE/scenario_wip/Scenario_lib/Biodiv_post2020/SSP2_CCI
'''

'''
Share of land use types - Figures
'''
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
            ref_for_coords = f"{data_path['scenario_input']}GLCT.NC"
            ref_dataset = xr.open_dataset(ref_for_coords, engine="netcdf4")
            dataset = dataset.assign_coords(
                latitude=ref_dataset.latitude.astype(np.float32).data,
                longitude=ref_dataset.longitude.astype(np.float32).data,
            )
            ref_dataset.close()
    return dataset.transpose(*sorted(dataset.dims))


def load_dataset(file_list: list, **kwargs_sel):
    """
    Function to load multiple datasets at once with chunks in time to optimize memory usage
    """

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        msa_dataset = xr.open_mfdataset(
            paths=file_list,
            chunks=dict(time="auto", latitude="auto", longitude="auto"),
            # chunks=dict(time="auto"),
            join="right",
            preprocess=pre_processing,
            engine="netcdf4",
        ).sel(**kwargs_sel)
    return msa_dataset


def get_paths(input=""):
    scen_name = os.path.basename(input)
    project_name = os.path.basename(os.path.dirname(input))
    return dict(
        scenario_input=os.path.join(input, "netcdf", ""),
        output=os.path.join(project_name, scen_name, ""),
        area_region_input=os.path.join("input", ""),
    )


def load_parameters(filename="parameters_validation.ini"):
    config = configparser.ConfigParser()
    config.read(filename)
    return dict(
        input_path=config["MSA_TOOL"]["input_path"],
        list_of_scen=[
            scen.lstrip()
            for scen in config["MSA_TOOL"]["list_of_scen"][1:-1].split(",")
        ],
        years=[int(year) for year in config["MSA_TOOL"]["years"][1:-1].split(",")],
        make_figures=config.getboolean("MSA_TOOL", "make_figures"),
        force_compute_files=config.getboolean("MSA_TOOL", "force_compute_files"),
    )


parameters = load_parameters()


path_in = os.path.join(parameters.get("input_path"))


data_path = get_paths(
    path_in
)


file_list = [
    f"{data_path['area_region_input']}image/share.nc",
    f"{data_path['area_region_input']}../../../input/GAreaCellNoWater.NC",
]


msa_dataset = load_dataset(file_list)

for land_use_type in msa_dataset.land_use_type.values:
    with plt.style.context("seaborn-talk"):

        msa_dataset.share.isel(time=0).sel(land_use_type=land_use_type).fillna(
            0.0
        ).where(msa_dataset.GAreaCellNoWater > 0.0).plot(
            rasterized=True,
            x="longitude",
            y="latitude",
            vmin=0.0,
            vmax=1.0,
            extend="neither",

        )
        title = (
            f"Share - {land_use_type} - {msa_dataset.isel(time=0).time.dt.year.data}"
        )
        plt.title(title)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(f'{title}.pdf', dpi=200)
        # plt.show()
        plt.close("all")

del msa_dataset


'''
Scatter Plot - Figures
'''
def load_from_globio(filename, sheet_name):
    dataframe = pd.read_excel(
        filename, sheet_name=sheet_name, skiprows=33, index_col=0, header=[0, 1]
    )
    dataframe = dataframe.drop(
        ["Unnamed: 1_level_0", "Unnamed: 2_level_0", "CHECK"], axis=1
    )

    return (
        dataframe.unstack()
        .to_xarray()
        .rename(
            {"level_0": "pressure_msa", "IMAGE region": "species", "level_2": "region"}
        )
        .sel(species=["Plants", "Wbvert"])
        .drop_isel(region=26)
    )


globio_da = xr.concat(
    (
        load_from_globio(f"{data_path['area_region_input']}globio/GLOBIO4_results_2015.xlsm", "2015"),
        load_from_globio(f"{data_path['area_region_input']}globio/GLOBIO4_results_SSP2_CCI.xlsm", "2050")
    ),
     dim = "time"
)


path = "X:/user/ambrosiog/tools/biodiversity_index/Biodiv_post2020/SSP2_CCI/"

image_da = (
    xr.concat(
        [
            xr.open_dataset(f"{data_path['area_region_input']}image/" + "pressure_impact.nc")["pressure_impact"],
            xr.open_dataset(f"{data_path['area_region_input']}image/" + "msa_region.nc")["msa_region"].expand_dims(
                pressure=["Remaining MSA"]
            ),
        ],
        dim="pressure",
    )
    .rename(pressure="pressure_msa", specie="species")
    .isel(
        time=[0, -1], region=[i for i in range(30) if i not in {26, 27, 28}], drop=True
    )
    .assign_coords(
        species=globio_da.species,
        region=globio_da.region.data,
        pressure_msa=[
            "Loss from climate",
            "Loss from land use",
            "Loss from nitrogen deposition",
            "Remaining MSA",
        ],
    )
)

globio_da = globio_da.assign_coords(time=image_da.time)

coord_to_sel = [
    "Loss from human encroachment",
    "Loss from infrastructure",
    "Loss from fragmentation",
]

globio_da[dict(pressure_msa=-1)] += globio_da.sel(pressure_msa=coord_to_sel).sum(
    "pressure_msa"
)


globio_da = globio_da.drop_sel(pressure_msa=coord_to_sel)


dataArray = xr.concat([globio_da, image_da], dim="model").assign_coords(
    model=["GLOBIO", "IMAGE"]
)

dataArray = dataArray.sortby("pressure_msa", ascending=False)


scatter_plot = xr.plot.scatter(
    dataArray.to_dataset("model"),
    x="GLOBIO",
    y="IMAGE",
    hue="pressure_msa",
    col="species",
    row="time",
)

for pressure in range(dataArray.pressure_msa.size):
    i = 0
    for time in range(dataArray.time.size):
        for species in range(dataArray.species.size):
            x = dataArray.isel(
                model=0, time=time, species=species, pressure_msa=pressure
            ).values.flatten()
            y = dataArray.isel(
                model=1, time=time, species=species, pressure_msa=pressure
            ).values.flatten()

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

            if slope == 0:
                continue

            x_test = np.linspace(x.min(), x.max(), num=20)
            y_test = x_test * slope + intercept

            scatter_plot.axes.flat[i].plot(x_test, y_test, color=f"C{pressure}", zorder=-1)
            scatter_plot.axes.flat[i].annotate(
                f"$R^2$ = {r_value ** 2.0:.2f}",
                (0.15, 0.85 - 0.1 * pressure),
                xycoords="axes fraction",
                color=f"C{pressure}",
            )

            i += 1

plt.suptitle("MSA and Pressures in fixed time", fontsize=14, y=1.05)
# plt.tight_layout()

plt.savefig(f'globio_image_2015_2050.pdf')

dataArray_diff = (dataArray.isel(time=-1) - dataArray.isel(time=0)).sortby(
    "pressure_msa", ascending=False
)


scatter_plot = xr.plot.scatter(
    dataArray_diff.to_dataset("model"),
    x="GLOBIO",
    y="IMAGE",
    hue="pressure_msa",
    col="species",
    # row="time",
)

for pressure in range(dataArray_diff.pressure_msa.size):
    i = 0

    for species in range(dataArray_diff.species.size):
        x = dataArray_diff.isel(
            model=0, species=species, pressure_msa=pressure
        ).values.flatten()
        y = dataArray_diff.isel(
            model=1, species=species, pressure_msa=pressure
        ).values.flatten()

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

        if slope == 0:
            continue

        x_test = np.linspace(x.min(), x.max(), num=20)
        y_test = x_test * slope + intercept

        scatter_plot.axes.flat[i].plot(x_test, y_test, color=f"C{pressure}", zorder=-1)
        scatter_plot.axes.flat[i].annotate(
            f"$R^2$ = {r_value ** 2.0:.2f}",
            (0.15, 0.9 - 0.1 * pressure),
            xycoords="axes fraction",
            color=f"C{pressure}",
        )

        # plt.ylabel("Change in MSA and Pressures (2050 - 2015)")
        i += 1

plt.suptitle("Change in MSA and Pressures (2050 - 2015)", fontsize=10, y=1.0)

# plt.rcParams["figure.figsize"] = (20,3)
plt.savefig(f'globio_image_change_2050_2015.pdf')