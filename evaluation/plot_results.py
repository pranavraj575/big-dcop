import argparse, json, matplotlib.pyplot as plt
import os.path

import numpy as np


def make_plot(args, key, results, save_path):
    algs = []
    means = []
    stds = []
    ns = []
    for alg in results:
        if key in results[alg]['summary']:
            if args.algorithms is None or alg in args.algorithms:
                algs.append(alg)
                means.append(results[alg]['summary'][key]['mean'])
                stds.append(results[alg]['summary'][key]['std'])
                ns.append(results[alg]['summary'][key]['n'])
    stds = np.array(stds)
    ns = np.array(ns)
    sample_stds = stds*np.sqrt(ns/(ns - 1))  # sample stdv divides by n-1 instead of n
    standard_error = sample_stds/np.sqrt(ns)
    # 95% confidence int, assuming means are normally distributed
    plt.errorbar(np.arange(len(means)), means, yerr=standard_error*1.96,
                 fmt='o', elinewidth=1, capsize=5, ecolor='black')
    plt.xticks(np.arange(len(means)), algs)
    plt.xticks(rotation=45, ha="right")
    if args.log_scale:
        plt.yscale("log")
    plt.title(key)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        required=False,
        default="output/results.json",
        type=str,
        help="Directory of json result file.",
    )
    p.add_argument('--output',
                   required=False,
                   default="output/plots",
                   type=str,
                   help='output dir',
                   )
    p.add_argument('--algorithms',
                   required=False,
                   nargs="*",
                   default=None,
                   type=str,
                   help='algorithms to include (defaults to all)',
                   )
    p.add_argument('--log_scale',
                   required=False,
                   action='store_true',
                   help='plot on log scale',
                   )
    args = p.parse_args()
    f = open(args.path, 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    for problem, results in data.items():
        problem = os.path.basename(problem).split('.')[0]
        all_keys = set()
        for algorithm in results:
            all_keys = all_keys.union(set(results[algorithm]['summary'].keys()))
        for key in all_keys:
            plt_name = f'{problem}_{key}{"_log" if args.log_scale else ""}.png'
            make_plot(args, key, results, os.path.join(args.output, plt_name))
