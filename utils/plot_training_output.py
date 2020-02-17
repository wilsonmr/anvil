"""
plot_training_output.py
"""
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from os import mkdir
import argparse
from datetime import datetime

from load_training_output import check_all_inputs, TrainingOutput
from analyse_training_output import detect_plateau


figures_dir = "./figures/"  # directory in which to save figs
extension = ".png"  # file extension for figs (include dot)

# Argument choices and defaults
params_choices = (
    "lattice_length",
    "m_sq",
    "lam",
    "use_arxiv_version",
    "hidden_nodes",
    "n_affine",
    "n_batch",
    "optimizer",
    "lr",
    "factor",
    "patience",
)
default_data_dirs = [
    "./training_output",
]

xdata_choices = ("epochs", "time")
xdata_default = "epochs"
ydata_choices = ("loss", "acceptance", "tauint")
ydata_default = "acceptance"

tunits_choices = ("seconds", "minutes", "hours")
tunits_default = "hours"


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", help="save output.", action="store_true")
parser.add_argument(
    "-o",
    "--output",
    metavar="",
    help=f"output file name. Default: {figures_dir}<...>{extension} with <...> generated automatically based on other inputs.",
    default=None,  # gets changed later
)
parser.add_argument(
    "-d",
    "--dirs",
    metavar="",
    nargs="+",
    help=f"list of directories containing training data (specifically, 'runcard.yml' and 'training_data.out'). Default: {default_data_dirs}.",
    default=default_data_dirs,
)
parser.add_argument(
    "-p",
    "--params",
    metavar="",
    nargs="+",
    help=f"parameters to compare between different trained models. These will be listed on the plot legend. Options: {', '.join(params_choices)}. Default: None.",
    choices=params_choices,
    default=None,
)
parser.add_argument(
    "-x",
    "--xdata",
    metavar="",
    help=f"x-axis data. Options: {', '.join(xdata_choices)}. Default: {xdata_default}.",
    choices=xdata_choices,
    default=xdata_default,
)
parser.add_argument(
    "-y",
    "--ydata",
    metavar="",
    help=f"y-axis data. Options: {', '.join(ydata_choices)}. Default: {ydata_default}.",
    choices=ydata_choices,
    default=ydata_default,
)
parser.add_argument(
    "-e",
    "--errorbars",
    help="plot the standard deviation as error bars. Only possible for ydata = acceptance, tauint",
    action="store_true",
)
parser.add_argument(
    "--plateau",
    help="plot location of auto-detected plateau in training",
    action="store_true",
)
parser.add_argument(
    "--lr", help="plot learning rate on a second axis", action="store_true"
)
parser.add_argument(
    "--tunits",
    metavar="",
    help=f"units for time axis, if appropriate. Options: {', '.join(tunits_choices)}. Defalt: {tunits_default}.",
    choices=tunits_choices,
    default=tunits_default,
)
args = parser.parse_args()

params = args.params
data_dirs = args.dirs
for i in range(len(data_dirs)):
    if data_dirs[i][-1] != "/":
        data_dirs[i] += "/"
xdata_type = args.xdata
ydata_type = args.ydata
plot_errorbars = args.errorbars
plot_plateau = args.plateau
plot_lr = args.lr
tunits = args.tunits
save = args.save

# Output file
if args.output is None:
    if len(data_dirs) == 1:  # name after data directory
        output = figures_dir + data_dirs[0].strip("/") + extension
    elif params is None:  # generic name + current datetime
        time = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        output = figures_dir + "figure_" + time + extension
    else:  # name after parameters which we're comparing
        output = (
            figures_dir
            + ydata_type[:3]
            + "-"
            + xdata_type[:3]
            + "_"
            + "-".join(param[:3] for param in params)
            + extension
        )
else:
    output = args.output


# Perform checks
required_files = ["training_data.out", "runcard.yml"]
if plot_lr == True:
    required_files.append("learning_rate.out")
check_all_inputs(data_dirs, required_files)


# Create plot
plt.rcParams.update({"font.size": 13, "axes.linewidth": 2})
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("Training data")

if ydata_type == "tauint":
    yaxis_label = r"$\tau_{int}$"
else:
    yaxis_label = ydata_type
if xdata_type == "time":
    xaxis_label = f"time ({tunits})"
else:
    xaxis_label = xdata_type
ax.set_xlabel(xaxis_label)
ax.set_ylabel(yaxis_label)

if plot_lr == True:
    ax2 = ax.twinx()
    ax2.set_ylabel("learning rate")
    ax2.set_yscale("log")

if plot_errorbars == True and ydata_type in ("acceptance", "tauint"):
    yerr_type = "std_" + ydata_type
else:
    yerr_type = None

# Cycle through data directories and add to plot
for data_dir in data_dirs:
    training_output = TrainingOutput(data_dir)

    xdata = training_output.get_data(xdata_type, tunits)
    ydata = training_output.get_data(ydata_type)

    label = training_output.get_label(params)
    color = next(ax._get_lines.prop_cycler)["color"]

    if yerr_type is not None:
        yerr = training_output.get_data(yerr_type)
        ax.errorbar(xdata, ydata, yerr=yerr, label=label, color=color)
    else:
        ax.plot(xdata, ydata, label=label, color=color)

    if plot_plateau == True:
        _, i = detect_plateau(ydata)
        ax.plot([xdata[i], xdata[i]], [ydata.min(), ydata.max()], ":", color=color)

    if plot_lr == True:
        lr_xdata = training_output.get_lr_data(xdata_type, tunits)
        lr_ydata = training_output.get_lr_data("lr")
        ax2.plot(lr_xdata, lr_ydata, "--", color=color)

ax.legend()
fig.tight_layout()
if save == True:
    if not exists(figures_dir):
        mkdir(figures_dir)
    plt.savefig(output)
plt.show()
