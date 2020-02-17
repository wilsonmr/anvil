"""
analyse_training_output.py

Note: Currently, all Runcards must have the same set of parameters for this script to work!!!
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

from load_training_output import check_all_inputs, TrainingOutput, organise_repeats

tables_dir = "tables/"

def is_plateau(acceptance, thresh=0.01):
    """Returns True if standard deviation of an array of acceptances is below a given threshold."""
    if np.std(acceptance) < thresh:
        return True
    else:
        return False


def detect_plateau(acceptance, n_points=10):
    """Evaluate whether the acceptance has plateaued, and if so determine which iteration.

    Parameters
    ----------
    acceptance: array or list of acceptances,
    n_points: (int) minimum number of iterations before we can conclude that it was plateaued

    Returns
    -------
    bool for whether a plateau has been detected or not,
    index corresponding to the moment the plateau threshold was reached
    """
    detected = False
    if len(acceptance) > n_points:
        # Check if longer than n_points - if not, we're going to say we can't detect
        # a plateau
        detected = is_plateau(acceptance[-n_points:])

    if detected == False:
        return detected, -1
    else:
        i = 0
        while detected == True:
            i -= 1
            detected = is_plateau(acceptance[i - n_points : i])
        return True, i


def reach_target(acceptance, target):
    """Return the iteration for the acceptance to exceed a given target value.

    Parameters
    ----------
    acceptance: array or list of acceptances
    target: target value for the acceptance to reach

    Returns
    -------
    True/False depending on whether target reached
    index corresponding to the iteration which that target was reached
    """
    i_exceed = np.where(acceptance > target)[0]
    if len(i_exceed) > 0:
        return True, i_exceed[0]
    else:
        return False, -1


if __name__ == "__main__":
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

    output_default = "output_table.csv"

    targets_default = [0.5, 0.7]

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        help=f"output file name. Default: {output_default}.",
        default=output_default,
    )
    parser.add_argument(
        "-d",
        "--dirs",
        metavar="",
        nargs="+",
        help=f"list of directories containing training data (specifically, 'runcard.yml' and 'training_data.out'). Default: {default_data_dirs}",
        default=default_data_dirs,
    )
    parser.add_argument(
        "-t",
        "--targets",
        metavar="",
        nargs="+",
        help=f"list of target acceptances. Default: {targets_default}.",
        type=float,
        default=targets_default,
    )
    parser.add_argument(
        "--tunits",
        metavar="",
        help=f"units for time axis, if appropriate. Options: {', '.join(tunits_choices)}. Default: {tunits_default}.",
        choices=tunits_choices,
        default=tunits_default,
    )
    args = parser.parse_args()

    output = args.output
    data_dirs = args.dirs
    for i in range(len(data_dirs)):
        if data_dirs[i][-1] != "/":
            data_dirs[i] += "/"
    targets = args.targets
    tunits = args.tunits

    # Perform basic checks on inputs
    check_all_inputs(data_dirs, ["training_data.out", "runcard.yml"])

    # Check for repeats (currently not doing anything with this feature)
    training_outputs = []
    for data_dir in data_dirs:
        training_outputs.append(TrainingOutput(data_dir))
    groups = organise_repeats(training_outputs)

    # Cycle through training outputs
    outfile = open(output, "w")
    headers = ""
    for group in groups:
        for training_output in group:

            # Column headers
            if not headers:
                params_headers = ", ".join(
                    [str(k) for k in training_output.get_runcard_params().keys()]
                )
                targ_headers = ""
                for t in targets:
                    targ_headers += (
                        rf" Reached {t}%?, Epochs, Time ({tunits}), Loss, Tau_int,"
                    )
                targ_headers += f" Plateau?, Epochs, Time ({tunits}), Loss, Tau_int"
                headers = params_headers + "," + targ_headers + "\n"
                outfile.write(headers)

            # Load data
            acceptance = training_output.get_data("acceptance")
            epochs = training_output.get_data("epochs")
            time = training_output.get_data("time", tunits)
            loss = training_output.get_data("loss")
            tauint = training_output.get_data("tauint")

            # Acceptance targets and plateauing
            targ_vals = ""
            for t in targets:
                TF, i = reach_target(acceptance, t)
                targ_vals += f" {TF}, {int(epochs[i])}, {time[i]:.3g}, {loss[i]:.3g}, {tauint[i]:.3g},"
            TF, i = detect_plateau(acceptance)
            targ_vals += f" {TF}, {int(epochs[i])}, {time[i]:.3g}, {loss[i]:.3g}, {tauint[i]:.3g}"
            params_vals = ", ".join(
                [str(v) for v in training_output.get_runcard_params().values()]
            )

            outfile.write(params_vals + "," + targ_vals + "\n")

    outfile.close()
