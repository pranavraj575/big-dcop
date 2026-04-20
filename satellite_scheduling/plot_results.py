import json
import os

import numpy as np
import matplotlib.pyplot as plt


def make_bar_plot(
    key,
    algorithms,
    stuff,
    reduce_fn=lambda lst: [max(lst)],
    title=None,
    ylabel=None,
    xlabel="algorithms",
    save_file=None,
    show=False,
    log_y=False,
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

    def get_data(alg):
        data = []
        for d in stuff:
            if alg in d["output"]:
                data.extend(reduce_fn([t[key] for t in d["output"][alg]["data"]]))
        return data

    all_data = {a: get_data(a) for a in algorithms}
    plt.bar(
        algorithms,
        [np.mean(all_data[k]) for k in algorithms],
    )
    plt.xticks(rotation=45, ha="right")
    plt.errorbar(
        algorithms,
        [np.mean(all_data[k]) for k in algorithms],
        [np.std(all_data[k]) for k in algorithms],
        fmt="none",
        color="black",
        capsize=5,
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if log_y:
        plt.yscale("log")
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(DIR, "output", "sat_sched")
    plt_dir = os.path.join(DIR, "output", "sat_sched_plots")
    os.makedirs(plt_dir, exist_ok=True)
    constraint_generation_files = [fn for fn in os.listdir(output_dir) if "_cg_" in fn]
    iterative_pricing_files = [fn for fn in os.listdir(output_dir) if "_cg_" not in fn]
    for scenario, files in (
        ("Constraint generation", constraint_generation_files),
        ("Iterative pricing", iterative_pricing_files),
        ("Default", constraint_generation_files),
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
            title=f"{scenario} Utility of best solution found",
            ylabel="Utility",
            save_file=os.path.join(plt_dir, scenario.lower().replace(" ", "_") + "cost.png"),
            reduce_fn=reduce_fns[0],
        )
        make_bar_plot(
            key="msg_count",
            algorithms=algorithms,
            stuff=stuff,
            title=f"{scenario} Average messages passed",
            ylabel="Messages",
            save_file=os.path.join(plt_dir, scenario.lower().replace(" ", "_") + "msg_count.png"),
            reduce_fn=reduce_fns[1],
            log_y=True,
        )
        make_bar_plot(
            key="time",
            algorithms=algorithms,
            stuff=stuff,
            title=f"{scenario} Average time per iteration",
            ylabel="Time (s)",
            save_file=os.path.join(plt_dir, scenario.lower().replace(" ", "_") + "time.png"),
            reduce_fn=reduce_fns[2],
            log_y=False,
        )
