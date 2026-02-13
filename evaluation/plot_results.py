import argparse, json, matplotlib.pyplot as plt
import os.path
import pandas as pd

import numpy as np


def plot_wrt_n(df, key, save_path, args,algs=None):
    if algs is None:
        algs = sorted(set(df['algorithm']), key=lambda s: s.lower())
    ns = sorted(set(df['n']))
    for alg in algs:
        temp_df = df.where(df['algorithm'] == alg)
        x = ns
        means = []
        std_errors = []
        for n in ns:
            temp_temp_df = temp_df.where(temp_df['n'] == n)
            y = temp_temp_df[key]
            means.append(y.mean())
            sample_size = y.notna().sum()
            sample_variance = y.var()*sample_size/(sample_size - 1)
            std_error = np.sqrt(sample_variance)/np.sqrt(n)
            std_errors.append(std_error)
        means, std_errors = np.array(means), np.array(std_errors)
        t = plt.plot(x, means, label=alg)
        color = t[0].get_c()
        z = 1.96
        plt.fill_between(x, means - std_errors*z, means + std_errors*z, color=color, alpha=.2)
    plt.legend()
    plt.ylabel(key)
    plt.xlabel('n')
    if args.log_scale:
        plt.yscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        required=False,
        default=os.path.join(DIR, "output", "results.csv"),
        type=str,
        help="Directory of csv result file.",
    )
    p.add_argument('--output',
                   required=False,
                   default=os.path.join(DIR, "output", "plots"),
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

    plt_dir = args.output
    os.makedirs(plt_dir, exist_ok=True)

    df = pd.read_csv(args.path)

    # get n parameter from problem file name
    df['n'] = df['problem'].map(lambda s: int(s.split('_')[1][1:]))

    df.plot(x='n', y='cost', kind='scatter')
    plt.savefig(os.path.join(plt_dir, 'test.png'))
    plt.close()
    if args.algorithms is None:
        algs = sorted(set(df['algorithm']), key=lambda s: s.lower())
    else:
        algs = args.algorithms


    plot_wrt_n(df=df,
               key='cost',
               save_path=os.path.join(plt_dir, 'cost_by_algorithm_over_n.png'),
               algs=algs,
               args=args
               )