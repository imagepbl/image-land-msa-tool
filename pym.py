import numpy as np
import pandas as pd
import re as re
import os.path

from scipy.interpolate import interp1d  # for load_mym()

from csv import Sniffer as Sniffer
from string import whitespace as whitespace

LINESEP = "\n"

# ............................................................................ #
class Header:

    datatypes = {"real": float, "integer": int}

    def __init__(self, line):
        self.line = line
        self.dimensions = [1]
        self.has_domain = False
        self.has_time = False

        self.process()

    def process(self):
        """
        Extracts metadata from the header `line`.

        The header line contains "datatype name[d1,d2,...,dn](t) = [t1",
        with n [di] dimensions, datatype is equal to either "real" or "integer",
        where the time dimension ["(t)"] is optional.
        """
        domain_match = re.search(r"\((.*)\)", self.line)
        if domain_match:
            self.has_domain = True
            self.domain = domain_match.group(1)
            self.has_time = True if self.domain == "t" else False

        # > parse the [di] dimension values
        dim_match = re.search(r"\[(.*)\]", self.line)
        if dim_match:
            self.dimensions = [int(d) for d in dim_match.group(1).split(",")]

        # > a block of data values includes 1 time value if `has_time`
        self.data_block_size = np.prod(self.dimensions)
        if self.has_domain:
            self.data_block_size = self.data_block_size + 1

        # > establish Python data type
        mym_datatype = self.line.split()[0].lower()
        self.datatype = self.datatypes[mym_datatype]

        # > parse the data on the remainder of the line
        self.raw_data = self.line.partition("=")[-1].strip(whitespace + "[")


def read_mym(filename, path=""):
    """
    Read a MyM data file. Return numpy array.

    The MyM data file should contain a single variable, that can be
    time-dependent.

    Parameters
    ----------
    filename : string
        name of the file to be read in
    path : string
        path to the filename, either relative or absolute

    Returns
    -------
    header : string
        contains comment lines and variable specification, the last line should
        end with '[a1,...,an](t) = [' or '[a1,...,an] ='
    data : np.array
        contains the data as a [t,a1,...,an] shaped numpy matrix
    time : np.array
        vector with time values, with time.shape = [t,1]
    """
    # Open file, extract header, and read data
    filepath = os.path.join(path, filename)

    content = []
    metadata = []

    with open(filepath, "r") as mym_file:
        for line in mym_file:
            line, _, comment = line.partition("!")
            line = line.lstrip().replace("\n", " ")
            if not line and not content:
                metadata.append(comment)
            elif line:
                if not content:
                    header = Header(line)
                    content.append(header.raw_data)
                else:
                    content.append(line)

    content = "".join([s for s in content if s])

    # Process data
    # `first_chunk` results in a large enough string for the `Sniffer` to find the delimiter and
    # small enough to be fast.
    first_chunk = 64
    delimiters = "".join([",", whitespace])
    mym_format = Sniffer().sniff(content[:first_chunk], delimiters=delimiters)

    raw_data = np.fromstring(content, sep=mym_format.delimiter)

    # find the desired dimensions, where `domain_size == 1` for time-independent data
    domain_size, dimension_error = divmod(raw_data.size, header.data_block_size)
    if dimension_error:
        raise RuntimeError("File dimensions are parsed incorrectly.")

    # reshape `raw_data` in two steps to enable splitting off domain vector
    raw_dimensions = (domain_size, header.data_block_size)
    data = np.reshape(raw_data, raw_dimensions)

    if header.has_domain:
        # split off domain vector (often represents time), where `domain == data[:,0]` and reshape
        target_dimensions = (domain_size, *header.dimensions)
        domain, data = np.split(data, [1], axis=1)
        data = data.reshape(target_dimensions)
        domain = domain.reshape(domain_size)

        # domain entries should be strictly increasing, time should consist of integers
        if header.has_time:
            time_is_integer = all(t.is_integer() for t in domain)
            domain = domain.astype(dtype=int)
        else:
            time_is_integer = True
        domain_increases = all(domain[:-1] < domain[1:]) if domain_size > 1 else True
        domain_ok = time_is_integer and domain_increases
        if not domain_ok:
            raise RuntimeError("Domain dimension does not consist of strictly increasing numbers.")

        data = data.astype(header.datatype)

        return data, domain
    else:
        data = data.astype(header.datatype).reshape(header.dimensions)
        return data


def load_mym(filename, path="", time=None, extrapolate=""):
    """
    Load a MyM data file according to specifications. Return numpy array.

    The MyM data file should contain a single time-dependent variable.

    Parameters
    ----------
    filename : string
        name of the file to be read in, gives `data_in` and `time_in`
    path : string, optional
        path to the filename, either relative or absolute
    time : np.array
        time array to which the MyM data should be cast
    extrapolate : {"", "fill"}
        determines whether `data` will be extrapolated for `time` values
        outside the range `time_in` where `data_in` is defined.
        If `extrapolate` == "fill", `data` will be extended based on the
        closest defined values of `data_in`.

    Returns
    -------
    header : string
        contains comment lines and variable specification, the last line should
        end with '[a1,...,an](t) = ['
    data : np.array
        contains the data as a [t,a1,...,an] shaped numpy matrix
    """

    if time is None:
        time_start, time_stop, time_step = (1970, 2100, 5)
        time = np.arange(time_start, time_stop + time_step, step=time_step)

    try:
        data_in, time_in = read_mym(filename, path=path)
    except TypeError:
        print("Data read from [{}] is not time-dependent".format(filename))
        raise

    # TODO: check whether `time` has values outside of `time_in`,
    #       while `extrapolate` is off? {mvdb}
    # TODO: check whether we want to change interpolation method {mvdb}
    f = interp1d(time_in, data_in, axis=0, kind="linear")
    if extrapolate == "fill":
        time_bounds = min(time[time >= time_in[0]]), max(time[time <= time_in[-1]])
        bounds = f(time_bounds)
        f = interp1d(time_in, data_in, axis=0, kind="linear", fill_value=bounds)

    data = f(time)

    return data


def stringify(array, indent=4, sep=","):
    """
    Generate data string from array. Return string.
    """
    formatter = lambda number: "{:14s}".format("{:.4f}{}".format(number, sep))
    s = ""
    for row in array:
        line = [" " * indent]
        line += [formatter(x) for x in row]
        line += LINESEP
        s += "".join(line)

    return s


def get_yearly_data_table(data, dimensions, year, years, table):
    """
    Generate `data` for `year` in desired `table` format. Return numpy array.
    """
    if isinstance(data, pd.DataFrame):
        data_table = data.loc[:, [year]]
    elif isinstance(data, np.ndarray):
        data_table = data[year == years, ...]

    data_table = shape_data_table(data_table, dimensions, table)

    return data_table


def shape_data_table(data_table, dimensions, table):
    """
    Shape `data_table` to desired table format. Return numpy array.
    """
    if isinstance(data_table, (pd.DataFrame, pd.Series)):
        data_table = data_table.values

    to_shape = {"wide": [-1, dimensions[-1]], "long": [-1, 1]}
    data_table = data_table.reshape(to_shape[table])

    return data_table


def print_mym(data, years=None, sep=",", name="", table="long", comment=""):
    """
    Print `data` in MyM data format string. Return string.

    The time dimension in `years` is treated as a separate, by definition
    slowest changing dimension. Of the other `dimensions`, the leftmost
    dimension is the slowest changing dimension after `years`. The rightmost
    dimension is the fastest changing dimension.
    Long table format prints `data` in a single column, while wide table format
    prints `data` in matrix format with the fastest changing dimension on a
    single line.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series or numpy.array
        a DataFrame is used for a time-dependent variable, where each column
        contains the data for a single year, for a time-independent variable a
        Series is expected. A numpy array can contain both types of variables.
    years : array_like, optional
        contains time values associated with `data` when data is a numpy array
    sep : str, default=","
        character(s) used as separator of `data` elements
    name : string
        name of the variable contained in `data`
    table : {"long", "wide"}
        ways of formatting MyM `data` in string
    comment : string, optional
        description of `data`

    Returns
    -------
    data_string : string
        `data` in MyM string format

    See also
    --------
    write_mym : Write MyM `data_string` to file
    read_mym : Read MyM data from file
    """
    if table not in ["long", "wide"]:
        table = "long"
        print(
            "The [table] argument is not specified correctly, it should be"
            " either 'long' (default) or 'wide', using default."
        )

    # Generate `dimensions` and `datatype`
    # `dimensions` is cast into a list for the correct string representation
    if isinstance(data, (pd.DataFrame, pd.Series)):
        try:
            # Clean `data` for known MultiIndex issues
            data.index = data.index.remove_unused_levels()
            if all([level in data.index.names for level in ["image_region", "image_region_index"]]):
                data.index = data.index.droplevel(level="image_region_index")
            dimensions = list(data.index.levshape)

            # Check dimensionality
            # `data` should have a size that is equal to the product of its
            # dimensions, otherwise `data` has duplicates or is incomplete.
            if not np.prod(dimensions) == data.shape[0]:
                raise Exception(
                    "Data for [{}] variable has an inconsistent"
                    " format. Expected [{}] index entries, got"
                    " [{}].".format(name, np.prod(dimensions), data.shape[0])
                )
        except:
            dimensions = [len(data.index)]
        datatype = data.values.dtype
        years = [year for year in data] if isinstance(data, pd.DataFrame) else None
    elif isinstance(data, np.ndarray):
        dimensions = list(data.shape[1:])
        datatype = data.dtype
    else:
        raise TypeError(
            "Data for [{}] should be a numpy array or a pandas"
            " DataFrame or Series, instead data is of {}.".format(name, type(data))
        )

    if not np.issubdtype(datatype, np.number):
        raise ValueError("Data for [{}] contains non-numeric entries.".format(name))

    # Print header
    if comment:
        comment = "!" + comment if comment[0] != "!" else comment
        comment = comment + LINESEP if comment[-1] != LINESEP else comment
    mym_datatype = "real" if np.issubdtype(datatype, np.floating) else "integer"
    time = "(t) = [" if years is not None else " ="
    dimension_string = str(dimensions) if np.prod(dimensions) > 1 else ""
    header = comment + "{} {}{}{}{}".format(mym_datatype, name, dimension_string, time, LINESEP)

    # Print data
    data_string = [header]
    if years is None:
        data_string.append(stringify(shape_data_table(data, dimensions, table), sep=sep))
    else:
        for year in years:
            data_string.append("{}{}{}".format(year, sep, LINESEP))
            data_block = get_yearly_data_table(data, dimensions, year, years, table)
            data_string.append(stringify(data_block, sep=sep))

    # Close MyM data array
    # > last number in array should not be followed by a comma
    data_string[-1] = data_string[-1].rstrip("\n, ")
    if years is not None:
        data_string.append("]")
    data_string.append(";")

    return "".join(data_string)


def write_mym(
    data,
    years=None,
    filename=None,
    path=None,
    sep=",",
    variable_name="data",
    table="long",
    comment="",
):
    """
    Write mym `data` to file.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series or numpy.array
        a DataFrame is used for a time-dependent variable, where each column
        contains the data for a single year, for a time-independent variable a
        Series is expected. A numpy array can contain both types of variables.
    years : array_like, optional
        contains time values associated with `data` when data is a numpy array
    filename : str, optional
        name of the file to which `data` will be written
    path : str, optional
        path where `filename` will be created, `path` must exist
    sep: str, default=","
        character(s) used as separator of `data` elements
    variable_name : string, optional
        name of the variable contained in `data`
    table : {"long", "wide"}
        ways of formatting MyM `data` array in string
    comment : string, optional
        description of `data` printed at the top of the file

    Returns
    -------
    data_string : string
        `data` in MyM string format

    See also
    --------
    print_mym : Print `data` in MyM data format string.
    read_mym : Read MyM data from file
    """
    data_string = print_mym(
        data, years=years, sep=sep, name=variable_name, table=table, comment=comment
    )

    filename = variable_name + ".dat" if not filename else filename
    filepath = os.path.join(path, filename) if path else filename

    with open(filepath, "w+") as mym_file:
        mym_file.write(data_string)


if __name__ == "__main__":
    print("Performing pym integration tests.")
    dimensions = (4, 5, 6)
    data = np.random.rand(*dimensions)
    time = np.array([2015, 2020, 2025, 2030])
    variable_name = "test"
    filename = variable_name + ".dat"
    comment = (
        "This is a test data set of random numbers with dimensions {},"
        " where the first number indicates the number of time entries".format(dimensions)
    )

    print("  writing test data using [write_mym]:")
    try:
        write_mym(data, years=time, table="wide", variable_name=variable_name, comment=comment)
    except Exception as error:
        print("  x test data could not be written to mym-file, the following error occured:")
        print(error)
        raise error
    else:
        print("  > successfully written data to file")

    print("  reading test data using [read_mym]:")
    try:
        data_output, time_output = read_mym(filename)
    except Exception as error:
        print("Test data could not be read from [{}]".format(filename))
        print(error)
        raise error
    else:
        print("  > successfully read data into array")

    print("  checking equality of test input and output:")
    try:
        assert np.allclose(time_output, time, atol=1e-04)
        assert np.allclose(data_output, data, atol=1e-04)
    except AssertionError as error:
        print("  x time and data input and output do not match.")
        print(error)
        raise error
    else:
        print("  > input and output data match to required precision")
        print("Finished tests succesfully.")

    print()