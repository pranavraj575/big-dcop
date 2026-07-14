import os
import json
import itertools
import matplotlib.pyplot as plt

from matplotlib import rc

from graph_coloring.algo_configs import get_display_name
import numpy as np
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
    default=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")],
    type=str,
    nargs="+",
    help="output dir (or list of these) sent to run_experiments.sh",
)
p.add_argument(
    "--plot-dir",
    default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "cosp_solver_plots"),
    type=str,
    help="dir to send plots into",
)
p.add_argument(
    "--algorithms",
    default=None,
    type=str,
    required=False,
    help="json of algorithms to actually plot",
)
p.add_argument(
    "--not-same-scale",
    action="store_true",
    required=False,
    help="stops different frameworks from being forced to have same scale.",
)
p.add_argument(
    "--title",
    action="store_true",
    required=False,
    help="outputs plots with titles.",
)
p.add_argument(
    "--hyperparam-optim",
    action="store_true",
    required=False,
    help="for iterative pricing, choose the optimal step size for each algorithm (among those recieved).",
)
p.add_argument(
    "--max-iteration",
    default=None,
    type=int,
    help="specify if performance results should only reflect the first n iterations",
)
p.add_argument(
    "--dpi",
    default=300,
    type=int,
    help="dpi of plot",
)
args = p.parse_args()

plot_dir = args.plot_dir
possible_frameworks = ("iterative_pricing", "constraint_generation")

os.makedirs(plot_dir, exist_ok=True)
data = list()
algorithms = None
if args.algorithms is not None:
    with open(args.algorithms) as f:
        algorithms = [get_display_name(ac) for ac in json.load(f)]
# build dataset
for output_dir, framework in itertools.product(args.output_dir, possible_frameworks):
    pth = os.path.join(output_dir, framework)
    if not os.path.exists(pth):
        continue
    for fn in os.listdir(pth):
        with open(os.path.join(pth, fn)) as f:
            t = json.load(f)
        temp = [get_display_name(cfg) for cfg in t["algorithm_configs"]]
        if algorithms is None:
            algorithms = temp
        else:
            assert set(algorithms).issubset(temp)

        for algo_name, run in t["output"].items():
            trial_info = t["run_info"].copy()
            trial_info["algo_name"] = algo_name
            trial_data = run["aux_info"]
            data.append({"info": trial_info, "data": trial_data})
alg_name_map = dict()
temp_alg_name_map = dict()


def get_plotting_name(nm):
    global alg_name_map

    if nm in alg_name_map:
        nm = alg_name_map[nm]
    return (
        nm.replace("(context-based)", " (CB)")
        .replace("(context based)", " (CB)")
        .replace(") (", "; ")
        .replace("MaxSum", "Maxsum")
        .replace("  ", " ")
    )


# plot best utility over hyperparameter c for iterative pricing
iterative_pricing_data = list(filter(lambda d: d["info"]["framework"] == "iterative_pricing", data))
if args.max_iteration is None:
    all_get_stats = (lambda dic: dic["data"]["best_total_scheduled"],)
else:
    all_get_stats = (lambda dic: max(dic["data"]["utility_per_iter"][: args.max_iteration]),)
c_values = sorted(set(dic["info"]["step_size_c"] for dic in iterative_pricing_data))
c_values = list(map(lambda c: int(c) if int(c) == c else c, c_values))
print("c values:", c_values)
data_with_optimal_c = dict()
if len(c_values) > 1:
    for get_stats, include_error, include_legend, log_x in itertools.product(
        all_get_stats, (True,), (True, False), (False, True)
    ):
        alg_data = [
            list(filter(lambda d: d["info"]["algo_name"] == algo_name, iterative_pricing_data)) for algo_name in algorithms
        ]
        data_by_c = [[list(filter(lambda d: d["info"]["step_size_c"] == c_val, ad)) for c_val in c_values] for ad in alg_data]
        temp_stats = [[list(map(get_stats, tt)) for tt in t] for t in data_by_c]
        stats = np.nan * np.ones(
            (len(algorithms), len(c_values), max(max(len(list(map(get_stats, tt))) for tt in t) for t in data_by_c))
        )
        for alg_i, t in enumerate(temp_stats):
            for cval_j, tt in enumerate(t):
                stats[alg_i, cval_j, : len(tt)] = tt
        # shaped (num algs, num c values, num trials)
        for alg_idx, (algo_name, sts) in enumerate(zip(algorithms, stats)):
            (line,) = plt.plot(c_values, np.nanmean(sts, axis=1), label=get_plotting_name(algo_name), marker=".", alpha=0.75)
            opt_c_idx = np.argmax(np.nanmean(sts, axis=1))
            if algo_name not in data_with_optimal_c:
                print("c=", c_values[opt_c_idx], "for alg", algo_name, "val", np.nanmean(sts, axis=1)[opt_c_idx])
            else:
                assert data_with_optimal_c[algo_name] == (data_by_c[alg_idx][opt_c_idx])
            data_with_optimal_c[algo_name] = data_by_c[alg_idx][opt_c_idx]
            if args.hyperparam_optim:
                temp_alg_name_map[algo_name] = f"{algo_name} ($\\alpha={c_values[opt_c_idx]}$)"
            if include_error:
                # for each iteration, this is the number of samples
                counts = np.sum(np.logical_not(np.isnan(sts)), axis=1)
                std_errors = np.nanstd(sts, axis=1) / np.sqrt(counts - 1)
                plt.fill_between(
                    c_values,
                    np.nanmean(sts, axis=1) - std_errors,
                    np.nanmean(sts, axis=1) + std_errors,
                    color=line.get_color(),
                    alpha=0.25,
                )
        if include_legend:
            plt.legend(fontsize=14)
        plt.tick_params(labelsize=15)
        if args.title:
            plt.title("iterative pricing performance across step sizes", size=17)
        plt.ylabel("Proportion of requests fulfilled", size=17)
        plt.xlabel("$\\alpha$ (step size)", size=17)
        plt.grid(True, axis="both")
        if log_x:
            plt.xscale("log")

        save_file = os.path.join(
            plot_dir,
            f"iterative_pricing_c_{'log_' if log_x else ''}graph{'_w_err' if include_error else ''}{'_lg' if include_legend else ''}.png",
        )
        plt.savefig(save_file, bbox_inches="tight", dpi=args.dpi)
        plt.close()
data_with_optimal_c = sum((v for k, v in data_with_optimal_c.items()), [])
# add all other data
data_with_optimal_c += list(filter(lambda d: d["info"]["framework"] != "iterative_pricing", data))
if args.hyperparam_optim:
    data = data_with_optimal_c
alg_name_map.update(temp_alg_name_map)
# print stats
frameworks = sorted(set(dic["info"]["framework"] for dic in data))
for framework in frameworks:
    frm_data = list(filter(lambda d: d["info"]["framework"] == framework, data))
    alg_data = [list(filter(lambda d: d["info"]["algo_name"] == algo_name, frm_data)) for algo_name in algorithms]
    if all(len(t) == len(alg_data[0]) for t in alg_data):
        print(f"{len(alg_data[0])} samples: {framework}")
    else:
        for algo_name, t in zip(algorithms, alg_data):
            print(f"{len(t)} samples: {framework}, {algo_name}")

# plot single statistics as a bar graph: best utility and runtime
if args.max_iteration is None:
    all_get_stats = (
        lambda dic: dic["data"]["best_total_scheduled"],
        lambda dic: dic["data"]["runtime_s"],
        lambda dic: dic["data"]["runtime_s"],
        lambda dic: dic["data"]["total_messages"],
    )
else:
    all_get_stats = (
        lambda dic: max(dic["data"]["utility_per_iter"][: args.max_iteration]),
        lambda dic: sum(dic["data"]["runtime_per_iter"][: args.max_iteration]),
        lambda dic: sum(dic["data"]["runtime_per_iter"][: args.max_iteration]),
        lambda dic: dic["data"]["total_messages"],
    )


for title, get_stats in zip(
    ("fulfillment", "time", "log_time", "messages"),
    all_get_stats,
):
    min_stat = min(map(get_stats, data))
    if title == "time":
        min_stat = 0
    max_stat = max(map(get_stats, data))
    w, x = 0.4, np.arange(len(algorithms))
    x_vals = x - w / 2
    for framework in frameworks:
        frm_data = list(filter(lambda d: d["info"]["framework"] == framework, data))
        alg_data = [list(filter(lambda d: d["info"]["algo_name"] == algo_name, frm_data)) for algo_name in algorithms]
        stats = np.array([list(map(get_stats, ag)) for ag in alg_data])

        plt.tick_params(labelsize=15)

        bars = plt.bar(
            x_vals, stats.mean(axis=1) - min_stat, bottom=min_stat, width=w, label=framework.replace("_", " ").capitalize()
        )
        for bar in bars:
            txt = f"{bar.get_height() + min_stat:.3f}"
            # plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + min_stat, txt, ha="center", va="bottom", fontsize=12)
        # stats.std is sqrt(1/n * biased variance)
        # sample std is sqrt(1/(n-1) * biased variance) = stats.std *sqrt(n/(n-1))
        # std error is sample std/sqrt(n) = stats.std /sqrt(n-1)
        std_errors = stats.std(axis=1) / np.sqrt(stats.shape[1] - 1)
        plt.errorbar(
            x_vals,
            stats.mean(axis=1),
            std_errors,
            fmt="none",
            color="black",
            capsize=5,
        )
        # if title=='messages':
        #     print(list(zip(algorithms,stats.mean(axis=1))))
        x_vals = x_vals + w
        if title == "time" and False:
            print(f"{framework}\tavg time to run all algorithms once: {stats.mean(axis=1).sum()}")
            temp_scenarios = {scen: None for scen in set(dt["info"]["scenario"] for dt in frm_data)}

            for scen in temp_scenarios:
                temp_alg_dt = np.array(
                    [
                        list(filter(lambda d: d["info"]["algo_name"] == algo_name and d["info"]["scenario"] == scen, frm_data))
                        for algo_name in algorithms
                    ]
                )
                temp_stats = np.array([list(map(get_stats, ag)) for ag in temp_alg_dt])
                temp_scenarios[scen] = f"\t{temp_stats.mean(axis=1).sum()} s,\t{len(temp_stats[0])} samples"
            for scen in sorted(temp_scenarios.keys(), key=lambda sc: float(temp_scenarios[sc][: temp_scenarios[sc].index("s")])):
                print(f"\t scenario {scen}: {temp_scenarios[scen]}")

            # temp = np.array([list(map(get_stats, ag)) for ag in frm_data])

    plt.xticks(rotation=45, ha="right")
    if args.title:
        plt.title(f"{title}", size=17)
    ylabels = {
        "time": "Time (s)",
        "log_time": "Time (s)",
        "fulfillment": "Proportion of requests fulfilled",
        "messages": "Total messages passed",
    }
    plt.ylabel(ylabels[title], size=17)
    plt.xlabel("Algorithm", size=17)
    plt.xticks(x, labels=map(get_plotting_name, algorithms))

    plt.legend(fontsize=15, loc="lower center", bbox_to_anchor=(0.5, 1))
    # plt.legend(fontsize=14)
    plt.grid(True, axis="y")
    if not args.not_same_scale:
        plt.ylim(min_stat, max_stat)
    if title in ["log_time"]:
        plt.yscale("log")
    plt.grid(True, axis="y")
    save_file = os.path.join(plot_dir, f"{title}_bar.png")
    plt.savefig(save_file, bbox_inches="tight", dpi=args.dpi)

    plt.close()

# plot list statistics as a line: utility/runtime per iteration
if args.max_iteration is None:
    all_get_stat_list = (
        lambda dic: dic["data"]["utility_per_iter"],
        lambda dic: dic["data"]["runtime_per_iter"],
    )
else:
    all_get_stat_list = (
        lambda dic: dic["data"]["utility_per_iter"][: args.max_iteration],
        lambda dic: dic["data"]["runtime_per_iter"][: args.max_iteration],
    )

for title, get_stats_list in zip(("utility", "runtime"), all_get_stat_list):
    min_stat = min(map(min, map(get_stats_list, data)))
    max_stat = max(map(max, map(get_stats_list, data)))
    overall_max_iterations = max(map(len, map(get_stats_list, data)))
    for framework, include_error, include_legend in itertools.product(frameworks, (True,), (True, False)):
        plt.tick_params(labelsize=15)
        frm_data = list(filter(lambda d: d["info"]["framework"] == framework, data))
        alg_data = [list(filter(lambda d: d["info"]["algo_name"] == algo_name, frm_data)) for algo_name in algorithms]
        temp_stats = [list(map(get_stats_list, ag)) for ag in alg_data]
        max_iterations = max(map(len, map(get_stats_list, frm_data)))
        stats = np.nan * np.ones((len(algorithms), len(temp_stats[0]), max_iterations))
        if (include_error and title == "utility") and max_iterations < overall_max_iterations:
            print(f"framework {framework} only has samples up to {max_iterations} iterations")

        for i, alg_stats in enumerate(temp_stats):
            for j, sample_stats in enumerate(alg_stats):
                stats[i, j, : len(sample_stats)] = sample_stats

        for algo_name, sts in zip(algorithms, stats):
            n = np.sum(np.logical_not(np.all(np.isnan(sts), axis=0)))
            assert np.all(np.isnan(sts[:, n:]))
            sts = sts[:, :n]
            (line,) = plt.plot(np.arange(n), np.nanmean(sts, axis=0), label=get_plotting_name(algo_name))

            if include_error:
                # for each iteration, this is the number of samples
                counts = np.sum(np.logical_not(np.isnan(sts)), axis=0)
                std_errors = np.nanstd(sts, axis=0) / np.sqrt(np.maximum(counts - 1, 0.1))
                plt.fill_between(
                    np.arange(n),
                    np.nanmean(sts, axis=0) - std_errors,
                    np.nanmean(sts, axis=0) + std_errors,
                    color=line.get_color(),
                    alpha=0.25,
                )
        if include_legend:
            plt.legend(fontsize=14)
        if args.title:
            plt.title(f"{framework} performance", size=17)
        plt.tick_params(labelsize=15)
        ylabels = {"utility": "Proportion of requests fulfilled", "runtime": "Time (s)"}
        plt.ylabel(ylabels[title], size=17)
        plt.xlabel("Iteration", size=17)
        plt.grid(True, axis="both")
        if not args.not_same_scale:
            plt.ylim(min_stat, max_stat)

        save_file = os.path.join(
            plot_dir, f"{framework}_iter_plot_{title}{'_w_err' if include_error else ''}{'_lg' if include_legend else ''}.png"
        )
        plt.savefig(save_file, bbox_inches="tight", dpi=args.dpi)

        plt.close()
