import json
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from evaluation.algo_configs import get_display_name


def make_bar_plot(
    key,
    algorithms,
    stuff,
    reduce_fn=lambda lst: [max(lst)],
    title=None,
    ylabel=None,
    xlabel="Algorithms",
    save_file=None,
    show=False,
    log_y=False,
    alg_to_plot_params=None,
    ylim=None,
):
    """

    Parameters
    ----------
    key
    algorithms
    stuff
    reduce_fn: list -> list, reduces all iterations in a single trial
        i.e. for utility, the list of [u1,u2,...] should just be reduced to [max(ui)]
    title
    ylabel
    xlabel
    save_file
    show

    Returns
    -------

    """
    if alg_to_plot_params is None:
        alg_to_plot_params = dict()
    rc(
        "font",
        **{
            "family": "serif",
            "serif": ["Times"],
        },
    )
    rc("text", usetex=True)
    plt.tick_params(labelsize=15)

    def get_data(alg):
        data = []
        for d in stuff:
            if alg in d["output"]:
                data.extend(reduce_fn([t[key] for t in d["output"][alg]["data"]]))
        return data

    all_data = {a: get_data(a) for a in algorithms}
    labels = [alg_to_plot_params.get(a, dict()).get("plt_name", a) for a in algorithms]
    plt.bar(
        labels,
        [np.mean(all_data[k]) for k in algorithms],
    )
    plt.xticks(rotation=45, ha="right")
    std_errors = [np.std(all_data[k]) / np.sqrt(len(all_data[k])) for k in algorithms]
    plt.errorbar(
        labels,
        [np.mean(all_data[k]) for k in algorithms],
        std_errors,
        fmt="none",
        color="black",
        capsize=5,
    )
    plt.title(title, size=17)
    plt.ylabel(ylabel, size=17)
    plt.xlabel(xlabel, size=17)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True, axis="y")
    if log_y:
        plt.yscale("log")
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(DIR, "output", "sat_sched")
    plt_dir = os.path.join(DIR, "output", "sat_sched_plots")
    algorithm_dirs = [
        os.path.join(DIR, "satellite_scheduling", "baseline_algorithm_configs.json"),
        os.path.join(DIR, "satellite_scheduling", "rm_algorithm_configs.json"),
    ]
    alg_to_plot_params = dict()
    for fn in algorithm_dirs:
        with open(fn, "r") as f:
            t = json.load(f)
        for alg_config in t:
            alg_to_plot_params[get_display_name(alg_config)] = alg_config
    os.makedirs(plt_dir, exist_ok=True)
    constraint_generation_files = [fn for fn in os.listdir(output_dir) if "_cg_" in fn]
    iterative_pricing_files = [fn for fn in os.listdir(output_dir) if "_cg_" not in fn]
    for scenario, files in (
        ("Constraint generation", constraint_generation_files),
        ("Iterative pricing", iterative_pricing_files),
        ("Default", iterative_pricing_files),
    ):
        stuff = []
        for fn in files:
            with open(os.path.join(output_dir, fn)) as f:
                stuff.append(json.load(f))
        algorithms = set()
        for d in stuff:
            algorithms.update(d["output"].keys())
        algorithms = sorted(algorithms)
        # manually remove dsa, as it only has one instance of being successful
        if "dsa" in algorithms:
            algorithms.remove("dsa")
        for alg in stuff[0]["output"]:
            print(stuff[0]["output"][alg]["data"][0].keys())
            break
        if scenario == "Default":
            reduce_fns = [(lambda x: [x[-1]]) for _ in range(3)]
        else:
            reduce_fns = [
                lambda lst: [max(lst)],
                lambda x: x,
                lambda x: x,
            ]
        make_bar_plot(
            key="cost",
            algorithms=algorithms,
            stuff=stuff,
            # title=f"{scenario} Utility of best solution found",
            ylabel="Utility",
            save_file=os.path.join(plt_dir, scenario.lower().replace(" ", "_") + "_cost.png"),
            reduce_fn=reduce_fns[0],
            alg_to_plot_params=alg_to_plot_params,
            ylim=(0, 120),
        )
        make_bar_plot(
            key="msg_count",
            algorithms=algorithms,
            stuff=stuff,
            # title=f"{scenario} Average messages passed",
            ylabel="Messages",
            save_file=os.path.join(plt_dir, scenario.lower().replace(" ", "_") + "_msg_count.png"),
            reduce_fn=reduce_fns[1],
            log_y=True,
            alg_to_plot_params=alg_to_plot_params,
        )
        make_bar_plot(
            key="time",
            algorithms=algorithms,
            stuff=stuff,
            # title=f"{scenario} Average time per iteration",
            ylabel="Time (s)",
            save_file=os.path.join(plt_dir, scenario.lower().replace(" ", "_") + "_time.png"),
            reduce_fn=reduce_fns[2],
            log_y=False,
            alg_to_plot_params=alg_to_plot_params,
        )
