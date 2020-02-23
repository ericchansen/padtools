import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import string
from datetime import datetime

def create_output_dir_if_needed(directory):
    output_dir = pathlib.Path(directory)
    if not output_dir.is_dir():
        os.mkdir(str(output_dir))
        print("Created: {}".format(str(output_dir)))
    else:
        print("Already exists: {}".format(str(output_dir)))

class ImageDirectory(object):
    """Data structure for a directory of images.

    Attributes:
        name: A string the user provides. Used in graphing.
        path_to_dir: A Pathlib object.
        image_filenames: A list of Pathlib objects.
        column_data: A dictionary that contains detailed information.
        summary_data: A dictionary that contains somewhat less detailed information.
        column_data_frame: A Pandas data frame of the aforementioned column data.
        summary_data_frame: A Pandas data frame of the aforementioned summary data.
        box_settings: Annoying and complicated manual placement of boxes. Gross.
        lanes_to_sample: List of strings.
        lane_map: Maps index (0, 1, 2, ...) to string ("A", "B", "C", ...).
    """
    supported_filetypes = ('.jpg')
    def __init__(
        self,
        name,
        path_to_dir=None,
        box_settings=None,
        lanes_to_sample=None
    ):
        self._path_to_dir = None
        self._image_filenames = None

        self.name = name
        self.path_to_dir = path_to_dir
        self.column_data = None
        self.summary_data = None
        self.column_data_frame = None
        self.summary_data_frame = None
        self.box_settings = box_settings

        if lanes_to_sample:
            self.lanes_to_sample = [x.upper() for x in lanes_to_sample]
        else:
            self.lanes_to_sample = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        self.lane_map = {number: character for number, character in enumerate(string.ascii_uppercase)}
    @property
    def path_to_dir(self):
        return self._path_to_dir
    @path_to_dir.setter
    def path_to_dir(self, value):
        if isinstance(value, pathlib.Path):
            self._path_to_dir = value
        elif isinstance(value, str):
            self._path_to_dir = pathlib.Path(value)
        elif value is None:
            pass
        else:
            raise Exception("How should I deal with this?")
    @property
    def image_filenames(self):
        if self._image_filenames is None:
            self._image_filenames = [pathlib.Path(os.path.join(self.path_to_dir, x)) for x in os.listdir(self._path_to_dir) if x.lower().endswith(self.supported_filetypes)]
        return self._image_filenames
    def __repr__(self):
        pretty_filenames = "    " + "\n    ".join([str(x) for x in self.image_filenames])
        oooo_so_pretty = \
            f"Name: {self.name}\n" + \
            f"Directory: {self._path_to_dir}\n" + \
            f"Images:\n" + \
            f"{pretty_filenames}\n" + \
            f"Box settings:\n" + \
            f"{self.box_settings}"
        return oooo_so_pretty
    def save_data_frames_to_csv(self, output_dir="."):
        print(f"Saving {self.name} data frames to {str(output_dir)}.")
        if not isinstance(output_dir, pathlib.Path):
            output_dir = pathlib.Path(output_dir)
        column_filename = self.name.lower().replace(" ", "_").replace("%", "") + "_columns.csv"
        column_path = output_dir / column_filename
        summary_filename = self.name.lower().replace(" ", "_").replace("%", "") + "_summary.csv"
        summary_path = output_dir / summary_filename
        self.column_data_frame.to_csv(column_path)
        print(f"Saved {column_path}.")
        self.summary_data_frame.to_csv(summary_path)
        print(f"Saved {summary_path}.")

def load_multiple_directories(notebook_user_inputs, verbose=False):
    directories = []
    for notebook_user_input in notebook_user_inputs:
        directory = ImageDirectory(
            name=notebook_user_input['title'],
            path_to_dir=notebook_user_input['directory'],
            box_settings=notebook_user_input['box_settings'],
            lanes_to_sample=notebook_user_input['lanes_to_sample']
        )
        directories.append(directory)

        if verbose:
            print(directory)
    return directories

def read_rgb_from_image(
    pathlib_filename,
    verbose=False
):
    if verbose:
        print(f"Reading: {str(pathlib_filename)}")
    matrix_of_pixels = cv2.imread(str(pathlib_filename))
    assert matrix_of_pixels is not None, "Read not working!"
    return matrix_of_pixels

def plot_image(
    image,
    title=None
):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(image)

def sample_boxes(
    pathlib_filename,
    box_width=50,
    box_height=500,
    num_boxes=12,
    offset_top=1250,
    offset_left=560,
    horizontal_spacer=109,
    output_dir="output",
    background_pixel_matrix=None,
    do_subtraction=False,
    draw=False,
    save_images=True,
    verbose=False,
    lanes_to_sample=None
):
    lane_map = {number: character for number, character in enumerate(string.ascii_uppercase)}

    column_data = []
    image = read_rgb_from_image(str(pathlib_filename), verbose=verbose)
    if background_pixel_matrix is not None and do_subtraction:
        image = image - background_pixel_matrix
    for i, x in enumerate(range(num_boxes)):
        # Define rectangle to sample.
        box_top_left_x = int(offset_left + horizontal_spacer * x + box_width * x)
        box_top_left_y = int(offset_top)
        box_bottom_right_x = int(box_top_left_x + box_width)
        box_bottom_right_y = int(offset_top + box_height)

        # Skip any columns we don't want.
        if lane_map[i] not in lanes_to_sample:
            continue

        # Sample colors.
        pixels_in_box = image[box_top_left_y:box_bottom_right_y, box_top_left_x:box_bottom_right_x]

        average_color = pixels_in_box.mean(axis=0).mean(axis=0)

        # Not used, but this will save the cropped pixels used for sampling.
        # Would be cool to embed this into an output Excel.
        # cv2.imwrite("test.jpg", pixels_in_box)

        # Work out the date and time.
        _, date_string, time_string = pathlib_filename.stem.split('_')
        file_creation_date = datetime.strptime(date_string, "%Y%m%d")
        file_creation_time = datetime.strptime(time_string, "%H%M%S")
        file_creation_datetime = datetime.strptime(date_string + time_string, "%Y%m%d%H%M%S")

        # Additional color options.
        average_inverted_blue = 255 - average_color[0]
        average_inverted_green = 255 - average_color[1]
        average_inverted_red = 255 - average_color[2]
        grayscale_intensity = sum(average_color) / 3
        grayscale_intensity = (average_color[0] + average_color[1] + average_color[2]) / 3
        inverted_grayscale_intensity = 255 - grayscale_intensity

        # Save data from box.
        datum = {
            "Filename": str(pathlib_filename),
            "Column": lane_map[i],
            "TopLeftX": box_top_left_x,
            "TopLeftY": box_top_left_y,
            "BottomRightX": box_bottom_right_x,
            "BottomRightY": box_bottom_right_y,
            "AverageBlue": average_color[0],
            "AverageGreen": average_color[1],
            "AverageRed": average_color[2],
            "AverageInvertedBlue": average_inverted_blue,
            "AverageInvertedGreen": average_inverted_green,
            "AverageInvertedRed": average_inverted_red,
            "GrayscaleIntensity": grayscale_intensity,
            "InvertedGrayscaleIntensity": inverted_grayscale_intensity,
            "Date": file_creation_date,
            "Time": file_creation_time,
            "Datetime": file_creation_datetime
        }
        column_data.append(datum)

        # If you want to draw.
        if save_images or draw:
            cv2.rectangle(
                image,
                (box_top_left_x, box_top_left_y),
                (box_bottom_right_x, box_bottom_right_y),
                (255, 0, 0),
                21
            )

    blue_described = pd.Series([x["AverageBlue"] for x in column_data]).describe()
    green_described = pd.Series([x["AverageGreen"] for x in column_data]).describe()
    red_described = pd.Series([x["AverageRed"] for x in column_data]).describe()

    inverted_blue_described = pd.Series([x["AverageInvertedBlue"] for x in column_data]).describe()
    inverted_green_described = pd.Series([x["AverageInvertedGreen"] for x in column_data]).describe()
    inverted_red_described = pd.Series([x["AverageInvertedRed"] for x in column_data]).describe()

    grayscale_intensity_described = pd.Series([x["GrayscaleIntensity"] for x in column_data]).describe()
    inverted_grayscale_intensity_described = pd.Series([x["InvertedGrayscaleIntensity"] for x in column_data]).describe()

    # Average of columns.
    summary_data = {
        "Filename": str(pathlib_filename),

        "BlueMean": blue_described["mean"],
        "BlueStd": blue_described["std"],
        "BlueMin": blue_described["min"],
        "Blue25%": blue_described["25%"],
        "Blue50%": blue_described["50%"],
        "Blue75%": blue_described["75%"],
        "BlueMax": blue_described["max"],

        "GreenMean": green_described["mean"],
        "GreenStd": green_described["std"],
        "GreenMin": green_described["min"],
        "Green25%": green_described["25%"],
        "Green50%": green_described["50%"],
        "Green75%": green_described["75%"],
        "GreenMax": green_described["max"],

        "RedMean": red_described["mean"],
        "RedStd": red_described["std"],
        "RedMin": red_described["min"],
        "Red25%": red_described["25%"],
        "Red50%": red_described["50%"],
        "Red75%": red_described["75%"],
        "RedMax": red_described["max"],

        "InvertedBlueMean": inverted_blue_described["mean"],
        "InvertedBlueStd": inverted_blue_described["std"],
        "InvertedBlueMin": inverted_blue_described["min"],
        "InvertedBlue25%": inverted_blue_described["25%"],
        "InvertedBlue50%": inverted_blue_described["50%"],
        "InvertedBlue75%": inverted_blue_described["75%"],
        "InvertedBlueMax": inverted_blue_described["max"],

        "InvertedGreenMean": inverted_green_described["mean"],
        "InvertedGreenStd": inverted_green_described["std"],
        "InvertedGreenMin": inverted_green_described["min"],
        "InvertedGreen25%": inverted_green_described["25%"],
        "InvertedGreen50%": inverted_green_described["50%"],
        "InvertedGreen75%": inverted_green_described["75%"],
        "InvertedGreenMax": inverted_green_described["max"],

        "InvertedRedMean": inverted_red_described["mean"],
        "InvertedRedStd": inverted_red_described["std"],
        "InvertedRedMin": inverted_red_described["min"],
        "InvertedRed25%": inverted_red_described["25%"],
        "InvertedRed50%": inverted_red_described["50%"],
        "InvertedRed75%": inverted_red_described["75%"],
        "InvertedRedMax": inverted_red_described["max"],

        "GrayscaleIntensityMean": grayscale_intensity_described["mean"],
        "GrayscaleIntensityStd": grayscale_intensity_described["std"],
        "GrayscaleIntensityMin": grayscale_intensity_described["min"],
        "GrayscaleIntensity25%": grayscale_intensity_described["25%"],
        "GrayscaleIntensity50%": grayscale_intensity_described["50%"],
        "GrayscaleIntensity75%": grayscale_intensity_described["75%"],
        "GrayscaleIntensityMax": grayscale_intensity_described["max"],

        "InvertedGrayscaleIntensityMean": inverted_grayscale_intensity_described["mean"],
        "InvertedGrayscaleIntensityStd": inverted_grayscale_intensity_described["std"],
        "InvertedGrayscaleIntensityMin": inverted_grayscale_intensity_described["min"],
        "InvertedGrayscaleIntensity25%": inverted_grayscale_intensity_described["25%"],
        "InvertedGrayscaleIntensity50%": inverted_grayscale_intensity_described["50%"],
        "InvertedGrayscaleIntensity75%": inverted_grayscale_intensity_described["75%"],
        "InvertedGrayscaleIntensityMax": inverted_grayscale_intensity_described["max"],

        "Date": file_creation_date,
        "Time": file_creation_time,
        "Datetime": file_creation_datetime
    }

    if draw:
        plt.figure()
        plt.title(pathlib_filename.stem)
        plt.imshow(image)
    if save_images:
        output_filename = os.path.join(
            os.getcwd(),
            output_dir,
            pathlib_filename.stem + "_OUTPUT_20190317" + pathlib_filename.suffix
        )
        cv2.imwrite(output_filename, image)
        if verbose:
            print("Wrote: {}".format(output_filename))
    return column_data, summary_data

def sample_directory_and_create_output_data(
    image_filenames,
    output_directory,
    box_settings,
    lanes_to_sample,
    verbose=False,
    draw=False,
    save_images=False
):
    column_data = []
    summary_data = []
    for image_filename in image_filenames:
        column_data_from_this_file, summary_data_from_this_file = \
            sample_boxes(
                image_filename,
                output_dir=output_directory,
                box_width=box_settings["box_width"],
                box_height=box_settings["box_height"],
                num_boxes=box_settings["num_boxes"],
                offset_left=box_settings["offset_left"],
                offset_top=box_settings["offset_top"],
                horizontal_spacer=box_settings["horizontal_spacer"],
                lanes_to_sample=lanes_to_sample,
                verbose=verbose,
                draw=draw
            )
        column_data.extend(column_data_from_this_file)
        summary_data.append(summary_data_from_this_file)
    return column_data, summary_data

def create_column_data_frame(
    column_data,
    every_other_lane=False
):
    column_df = pd.DataFrame(column_data)
    column_df["TimeDelta"] = column_df["Time"] - column_df["Time"].min()
    column_df["TimeDelta"] = column_df["TimeDelta"].astype('timedelta64[s]')
    column_df = column_df[[
        "Filename",
        "Column",
        "TopLeftX",
        "TopLeftY",
        "BottomRightX",
        "BottomRightY",
        "AverageBlue",
        "AverageGreen",
        "AverageRed",
        "AverageInvertedBlue",
        "AverageInvertedGreen",
        "AverageInvertedRed",
        "GrayscaleIntensity",
        "InvertedGrayscaleIntensity",
        "Date",
        "Time",
        "Datetime",
        "TimeDelta"
    ]]
    if every_other_lane:
        column_df = column_df.loc[column_df["Column"].isin([0, 2, 4, 6, 8, 10])]
    return column_df

def create_summary_data_frame(
    summary_data
):
    summary_df = pd.DataFrame(summary_data)
    summary_df["TimeDelta"] = summary_df["Time"] - summary_df["Time"].min()
    summary_df["TimeDelta"] = summary_df["TimeDelta"].astype('timedelta64[s]')
    summary_df = summary_df[[
        "Filename",

        "BlueMean",
        "BlueStd",
        "BlueMin",
        "Blue25%",
        "Blue50%",
        "Blue75%",
        "BlueMax",
        "GreenMean",
        "GreenStd",
        "GreenMin",
        "Green25%",
        "Green50%",
        "Green75%",
        "GreenMax",
        "RedMean",
        "RedStd",
        "RedMin",
        "Red25%",
        "Red50%",
        "Red75%",
        "RedMax",

        "InvertedBlueMean",
        "InvertedBlueStd",
        "InvertedBlueMin",
        "InvertedBlue25%",
        "InvertedBlue50%",
        "InvertedBlue75%",
        "InvertedBlueMax",

        "InvertedGreenMean",
        "InvertedGreenStd",
        "InvertedGreenMin",
        "InvertedGreen25%",
        "InvertedGreen50%",
        "InvertedGreen75%",
        "InvertedGreenMax",

        "InvertedRedMean",
        "InvertedRedStd",
        "InvertedRedMin",
        "InvertedRed25%",
        "InvertedRed50%",
        "InvertedRed75%",
        "InvertedRedMax",

        "GrayscaleIntensityMean",
        "GrayscaleIntensityStd",
        "GrayscaleIntensityMin",
        "GrayscaleIntensity25%",
        "GrayscaleIntensity50%",
        "GrayscaleIntensity75%",
        "GrayscaleIntensityMax",

        "InvertedGrayscaleIntensityMean",
        "InvertedGrayscaleIntensityStd",
        "InvertedGrayscaleIntensityMin",
        "InvertedGrayscaleIntensity25%",
        "InvertedGrayscaleIntensity50%",
        "InvertedGrayscaleIntensity75%",
        "InvertedGrayscaleIntensityMax",

        "Date",
        "Time",
        "Datetime",
        "TimeDelta"
    ]]
    return summary_df

def plot_colors_over_time(
    summary_df,
    title=None,
    save=False,
    output_dir=".",
    pad_sides=15
):
    if title:
        title += " Average Color Change Over Time"
    else:
        title = "Average Color Change Over Time"
    fig, ax = plt.subplots()

    summary_df.plot(ax=ax, x="TimeDelta", y="BlueMean", yerr="BlueStd", capsize=3, color="blue")
    summary_df.plot(ax=ax, x="TimeDelta", y="RedMean", yerr="RedStd", capsize=3, color="red")
    summary_df.plot(ax=ax, x="TimeDelta", y="GreenMean", yerr="GreenStd", capsize=3, color="green")
    ax.set_title(title)
    ax.set_ylabel("Intensity (0 - 255)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(
        summary_df.iloc[0]["TimeDelta"] - pad_sides,
        summary_df.iloc[-1]["TimeDelta"] + pad_sides
    )
    ax.legend(["Blue Mean", "Red Mean", "Green Mean"])
    if save:
        leaf = title.lower().replace(" ", "_").replace("%", "") + ".png"
        filename = output_dir / leaf
        fig.savefig(str(filename))

def plot_inverted_colors_over_time(
    summary_df,
    title=None,
    save=False,
    output_dir=".",
    pad_sides=15
):
    if title:
        title += " Average Inverted Color Change Over Time"
    else:
        title = "Average Inverted Color Change Over Time"
    fig, ax = plt.subplots()
    summary_df.plot(ax=ax, x="TimeDelta", y="InvertedBlueMean", yerr="InvertedBlueStd", capsize=3, color="blue")
    summary_df.plot(ax=ax, x="TimeDelta", y="InvertedRedMean", yerr="InvertedRedStd", capsize=3, color="red")
    summary_df.plot(ax=ax, x="TimeDelta", y="InvertedGreenMean", yerr="InvertedGreenStd", capsize=3, color="green")
    ax.set_title(title)
    ax.set_ylabel("Intensity (0 - 255)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(
        summary_df.iloc[0]["TimeDelta"] - pad_sides,
        summary_df.iloc[-1]["TimeDelta"] + pad_sides
    )
    ax.legend(["Inverted Blue Mean", "Inverted Red Mean", "Inverted Green Mean"])
    if save:
        leaf = title.lower().replace(" ", "_").replace("%", "") + ".png"
        filename = output_dir / leaf
        fig.savefig(str(filename))

def plot_grayscale_over_time(
    summary_df,
    title=None,
    save=False,
    output_dir=".",
    pad_sides=15
):
    if title:
        title += " Average Grayscale Intensity Over Time"
    else:
        title = "Average Grayscale Intensity Over Time"
    fig, ax = plt.subplots()
    summary_df.plot(
        ax=ax,
        x="TimeDelta",
        y="GrayscaleIntensityMean",
        yerr="GrayscaleIntensityStd",
        capsize=3,
        color="gray",
        legend=False
    )
    ax.set_title(title)
    ax.set_ylabel("Intensity (0 - 255)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(
        summary_df.iloc[0]["TimeDelta"] - pad_sides,
        summary_df.iloc[-1]["TimeDelta"] + pad_sides
    )
    if save:
        leaf = title.lower().replace(" ", "_").replace("%", "") + ".png"
        filename = output_dir / leaf
        fig.savefig(str(filename))

def plot_inverted_grayscale_over_time(
    summary_df,
    title=None,
    save=False,
    output_dir=".",
    pad_sides=15
):
    if title:
        title += " Average Inverted Grayscale Intensity Over Time"
    else:
        title = "Average Inverted Grayscale Intensity Over Time"
    fig, ax = plt.subplots()
    summary_df.plot(
        ax=ax,
        x="TimeDelta",
        y="InvertedGrayscaleIntensityMean",
        yerr="InvertedGrayscaleIntensityStd",
        capsize=3,
        color="gray",
        legend=False
    )
    ax.set_title(title)
    ax.set_ylabel("Intensity (0 - 255)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(
        summary_df.iloc[0]["TimeDelta"] - pad_sides,
        summary_df.iloc[-1]["TimeDelta"] + pad_sides
    )
    if save:
        leaf = title.lower().replace(" ", "_").replace("%", "") + ".png"
        filename = output_dir / leaf
        fig.savefig(str(filename))

def plot_many_inverted_grayscale_intensities_over_time(
    summary_dfs,
    names,
    save=False,
    output_dir=".",
    pad_sides=15,
    height_in_cm=8.3,
    width_in_cm=8.3,
    dpi=600
):
    plt.rcParams["font.size"] = 10
    plt.rcParams.update({"font.size": 10})
    width_in_in = width_in_cm * 0.3937
    height_in_in = height_in_cm * 0.3937

    overall_title = "Average Inverted Grayscale\nIntensity Over Time"
    fig, ax = plt.subplots()

    for summary_df, legend_name in zip(summary_dfs, names):
        summary_df.plot(
            ax=ax,
            x="TimeDelta",
            y="InvertedGrayscaleIntensityMean",
            yerr="InvertedGrayscaleIntensityStd",
            capsize=3,
            label=legend_name
        )
    # ax.set_title(overall_title)
    ax.set_ylabel("Intensity (0 - 255)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(
        summary_df.iloc[0]["TimeDelta"] - pad_sides,
        summary_df.iloc[-1]["TimeDelta"] + pad_sides
    )

    plt.legend(
        # bbox_to_anchor=(1.04, 1),
        bbox_to_anchor=(0.5, 0.14),
        ncol=2,
        frameon=False,
        loc="upper center",
        borderaxespad=0,
        bbox_transform=fig.transFigure
    )

    plt.tight_layout()

    if save:
        fig.set_size_inches(width_in_in, height_in_in)
        plt.tight_layout()

        leaf = overall_title.lower().replace(" ", "_").replace("%", "").replace("\n", " ") + ".tiff"
        filename = output_dir / leaf
        fig.savefig(
            str(filename),
            dpi=dpi,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight"
        )