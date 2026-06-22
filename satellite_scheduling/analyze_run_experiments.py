import os,json
import matplotlib.pyplot as plt

from matplotlib import rc
from evaluation.algo_configs import get_display_name
import numpy as np
from collections import defaultdict
import argparse


p = argparse.ArgumentParser()
p.add_argument("--output_dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)),'output'), type=str, help="output dir sent to run_experiments.sh")
p.add_argument("--plot_dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)),'output','cosp_solver_plots'), type=str, help="dir to send plots into")
args = p.parse_args()

frameworks=('iterative_pricing','constraint_generation')
output_dir=args.output_dir
plot_dir=args.plot_dir

os.makedirs(plot_dir,exist_ok=True)
data=defaultdict(lambda:list())
algorithms=None
for framework in frameworks:
    pth=os.path.join(output_dir,framework)
    for fn in os.listdir(pth):
        with open(os.path.join(pth,fn)) as f:
            t=json.load(f)
        temp=[get_display_name(cfg) for cfg in t['algorithm_configs']]
        if algorithms is not None:
            assert algorithms==temp
        algorithms=temp

        for algo_name,run in t['output'].items():
            data[(framework,algo_name)].append(run['aux_info'])
def get_stats(entry):
    return entry['best_total_scheduled']
min_stat=min(min(get_stats(entry)for entry in stuff) for _,stuff in data.items())
max_stat=max(max(get_stats(entry)for entry in stuff) for _,stuff in data.items())
print(min_stat)
print(max_stat)
for framework in frameworks:
    stats=np.array([list(map(get_stats ,data[(framework,alg)])) for alg in algorithms])

    rc(
        "font",
        **{
            "family": "serif",
            "serif": ["Times"],
        },
    )
    rc("text", usetex=True)
    plt.tick_params(labelsize=15)

    plt.bar(
        algorithms,
        stats.mean(axis=1)-min_stat,
        bottom=min_stat
    )
    plt.xticks(rotation=45, ha="right")
    # stats.std is sqrt(1/n * biased variance)
    # sample std is sqrt(1/(n-1) * biased variance) = stats.std *sqrt(n/(n-1))
    # std error is sample std/sqrt(n) = stats.std /sqrt(n-1)
    std_errors = stats.std(axis=1) / np.sqrt(stats.shape[1]-1)
    plt.errorbar(
        algorithms,
        stats.mean(axis=1),
        std_errors,
        fmt="none",
        color="black",
        capsize=5,
    )
    plt.title(f'performances with {framework} framework', size=17)
    plt.ylabel('proportion of requests fulfilled', size=17)
    plt.xlabel('algorithm', size=17)

    plt.grid(True, axis="y")
    plt.ylim(min_stat,max_stat)
    #plt.yscale("log")
    save_file=os.path.join(plot_dir,f'{framework}.png')
    plt.savefig(save_file, bbox_inches="tight", dpi=300)

    plt.close()
