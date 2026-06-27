import os
import json
import itertools
import matplotlib.pyplot as plt

from matplotlib import rc
from evaluation.algo_configs import get_display_name
import numpy as np
from collections import defaultdict
import argparse
import shutil

latex_exists = bool(shutil.which("latex"))

if latex_exists:
    rc(
        "font",
        **{
            "family": "serif",
            "serif": ["Times"],
        },
    )
rc("text", usetex=latex_exists)

p = argparse.ArgumentParser()
p.add_argument(
    "--output-dir",
    default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
    type=str,
    help="output dir sent to run_experiments.sh",
)
p.add_argument(
    "--plot-dir",
    default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "cosp_solver_plots"),
    type=str,
    help="dir to send plots into",
)
p.add_argument(
    "--not-same-scale",
    action="store_true",
    required=False,
    help="stops different frameworks from being forced to have same scale.",
)
p.add_argument(
    "--max-iteration",
    default=None,
    type=int,
    help="specify if performance results should only reflect the first n iterations",
)
args = p.parse_args()

frameworks = ("iterative_pricing", "constraint_generation")
output_dir = args.output_dir
plot_dir = args.plot_dir

os.makedirs(plot_dir, exist_ok=True)
data = defaultdict(lambda: list())
algorithms = None
for framework in frameworks:
    pth = os.path.join(output_dir, framework)
    for fn in os.listdir(pth):
        with open(os.path.join(pth, fn)) as f:
            t = json.load(f)
        temp = [get_display_name(cfg) for cfg in t["algorithm_configs"]]
        if algorithms is not None:
            assert algorithms == temp
        algorithms = temp

        for algo_name, run in t["output"].items():
            data[(framework, algo_name)].append(run["aux_info"])

for framework, algo_name in itertools.product(frameworks, algorithms):
    print(f"{len(data[(framework, algo_name)])} samples: {framework}, {algo_name}")
if args.max_iteration is None:
    all_get_stats = (
        lambda entry: entry["best_total_scheduled"],
        lambda entry: entry["runtime_s"],
        lambda entry: entry["runtime_s"],
    )
else:
    all_get_stats = (
        lambda entry: max(entry["utility_per_iter"][: args.max_iteration]),
        lambda entry: sum(entry["runtime_per_iter"][: args.max_iteration]),
        lambda entry: sum(entry["runtime_per_iter"][: args.max_iteration]),
    )

for title, get_stats in zip(
    ("fulfillment", "time", "log_time"),
    all_get_stats,
):
    min_stat = min(min(get_stats(entry) for entry in stuff) for _, stuff in data.items() if stuff)
    max_stat = max(max(get_stats(entry) for entry in stuff) for _, stuff in data.items() if stuff)
    print(title, "bounds", min_stat, max_stat)
    for framework in frameworks:
        stats = np.array([list(map(get_stats, data[(framework, alg)])) for alg in algorithms])

        plt.tick_params(labelsize=15)
        plt.bar(algorithms, stats.mean(axis=1) - min_stat, bottom=min_stat)
        plt.xticks(rotation=45, ha="right")
        # stats.std is sqrt(1/n * biased variance)
        # sample std is sqrt(1/(n-1) * biased variance) = stats.std *sqrt(n/(n-1))
        # std error is sample std/sqrt(n) = stats.std /sqrt(n-1)
        std_errors = stats.std(axis=1) / np.sqrt(stats.shape[1] - 1)
        plt.errorbar(
            algorithms,
            stats.mean(axis=1),
            std_errors,
            fmt="none",
            color="black",
            capsize=5,
        )
        plt.title(f"{framework} performance", size=17)
        ylabels = {"time": "time (s)", "log_time": "time (s)", "fulfillment": "proportion of requests fulfilled"}
        plt.ylabel(ylabels[title], size=17)
        plt.xlabel("algorithm", size=17)

        plt.grid(True, axis="y")
        if not args.not_same_scale:
            plt.ylim(min_stat, max_stat)
        if title in ["log_time"]:
            plt.yscale("log")
        plt.grid(True, axis="y")
        save_file = os.path.join(plot_dir, f"{framework}_{title}.png")
        plt.savefig(save_file, bbox_inches="tight", dpi=300)

        plt.close()

if args.max_iteration is None:
    all_get_stat_list = (
        lambda entry: entry["utility_per_iter"],
        lambda entry: entry["runtime_per_iter"],
    )
else:
    all_get_stat_list = (
        lambda entry: entry["utility_per_iter"][: args.max_iteration],
        lambda entry: entry["runtime_per_iter"][: args.max_iteration],
    )
for title, get_stats_list in zip(("utility", "runtime"), all_get_stat_list):
    min_stat = min(min([min(get_stats_list(entry)) for entry in stuff]) for _, stuff in data.items() if stuff)
    max_stat = max(max([max(get_stats_list(entry)) for entry in stuff]) for _, stuff in data.items() if stuff)
    max_iterations = max(max([len(get_stats_list(entry)) for entry in stuff]) for _, stuff in data.items() if stuff)

    for framework, include_error in itertools.product(frameworks, (True, False)):
        plt.tick_params(labelsize=15)
        temp_stats = [list(map(get_stats_list, data[(framework, alg)])) for alg in algorithms]
        stats = np.nan * np.ones((len(algorithms), len(data[(framework, algorithms[0])]), max_iterations))
        for i, alg_stats in enumerate(temp_stats):
            for j, sample_stats in enumerate(alg_stats):
                stats[i, j, : len(sample_stats)] = sample_stats

        for algo_name, sts in zip(algorithms, stats):
            n = sts.shape[1]
            (line,) = plt.plot(np.arange(n), np.nanmean(sts, axis=0), label=algo_name)
            if include_error:
                # for each iteration, this is the number of samples
                counts = np.sum(np.logical_not(np.isnan(sts)), axis=0)
                std_errors = np.nanstd(sts, axis=0) / np.sqrt(counts - 1)
                plt.fill_between(
                    np.arange(n),
                    np.nanmean(sts, axis=0) - std_errors,
                    np.nanmean(sts, axis=0) + std_errors,
                    color=line.get_color(),
                    alpha=0.25,
                )
        plt.legend()
        plt.title(f"{framework} performance", size=17)
        ylabels = {"utility": "proportion of requests fulfilled", "runtime": "time (s)"}
        plt.ylabel(ylabels[title], size=17)
        plt.xlabel("iteration", size=17)
        plt.grid(True, axis="both")
        if not args.not_same_scale:
            plt.ylim(min_stat, max_stat)

        save_file = os.path.join(plot_dir, f"{framework}_iter_plot_{title}{'_w_err' if include_error else ''}.png")
        plt.savefig(save_file, bbox_inches="tight", dpi=300)

        plt.close()
