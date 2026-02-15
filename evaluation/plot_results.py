import argparse, itertools, matplotlib.pyplot as plt
import os.path
import pandas as pd

import numpy as np


def kernel_smoothed_plot_wrt_value(df,
                                   key,
                                   x_param,
                                   save_path,
                                   kernel_fn,
                                   z_score=1.96,
                                   grid=None,
                                   x_log=False,
                                   y_log=False,
                                   algs=None,
                                   ):
    """
    Parameters
    ----------
    z_score
    df
    key
    x_param
    save_path
    kernel_fn: function of K(x0,x), x's weight when estimating x0
        for std_error to make sense, should have K(x0,x0)=1
    x_log
    y_log
    algs

    Returns
    -------

    """
    if algs is None:
        algs = sorted(set(df['algorithm']), key=lambda s: s.lower())
    if grid is None:
        grid = sorted(set(df[x_param]))
    for alg in algs:
        temp_df = df[df['algorithm'] == alg]
        means = []
        std_errors = []
        for x0 in grid:
            weights = temp_df[x_param].map(lambda x: kernel_fn(x0, x))
            y = temp_df[key]
            cum_weight = weights.sum()
            mu = (weights*y).sum()/cum_weight
            means.append(mu)
            sample_variance = (((y - mu)**2)*weights).sum()/(cum_weight - 1)
            std_error = np.sqrt(sample_variance)/np.sqrt(cum_weight)
            std_errors.append(std_error)
        means, std_errors = np.array(means), np.array(std_errors)
        t = plt.plot(grid, means, label=alg)
        color = t[0].get_c()
        plt.fill_between(grid, means - std_errors*z_score, means + std_errors*z_score, color=color, alpha=.2)
    plt.legend()
    plt.ylabel(key)
    plt.xlabel(x_param)
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_wrt_param(df, key, x_param, save_path, x_log=False, y_log=False, algs=None):
    kernel_smoothed_plot_wrt_value(df=df,
                                   key=key,
                                   x_param=x_param,
                                   save_path=save_path,
                                   kernel_fn=lambda x0, x: int(x0 == x),
                                   grid=None,
                                   x_log=x_log,
                                   y_log=y_log,
                                   algs=algs)


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
    p.add_argument('--y_keys',
                   required=False,
                   nargs="+",
                   default=['cost'],
                   type=str,
                   help='things to plot on y value',
                   )
    args = p.parse_args()

    plt_dir = args.output
    os.makedirs(plt_dir, exist_ok=True)

    df = pd.read_csv(args.path)

    # get n parameter from problem file name
    df['n'] = df['problem'].map(lambda s: int(s.split('_')[1][1:]))
    if args.algorithms is None:
        algs = sorted(set(df['algorithm']), key=lambda s: s.lower())
    else:
        algs = args.algorithms
    timeout_params = sorted(set(df['timeout_param']))
    for timeout_param, key in itertools.product(timeout_params,
                                                args.y_keys
                                                ):
        this_plot_dir = os.path.join(plt_dir, f'{key}_over_n')
        save_dir = os.path.join(this_plot_dir, f'timeout_{timeout_param}.png')
        relevant_df = df[df['timeout_param'] == timeout_param]
        relevant_df = relevant_df[relevant_df[key].notnull()]
        plot_wrt_param(
            df=relevant_df,
            key=key,
            x_param='n',
            save_path=save_dir,
            algs=algs,
        )
        print(f'saved to {save_dir}')

    n_params = sorted(set(df['n']))
    for n_param, key in itertools.product(n_params,
                                          args.y_keys
                                          ):
        this_plot_dir = os.path.join(plt_dir, f'{key}_over_timeout')
        save_dir = os.path.join(this_plot_dir, f'n_prm_{n_param}.png')
        relevant_df = df[df['n'] == n_param]
        relevant_df = relevant_df[relevant_df[key].notnull()]
        plot_wrt_param(
            df=relevant_df,
            key=key,
            x_param='timeout_param',
            save_path=save_dir,
            algs=algs,
            x_log=True,
        )
        print(f'saved to {save_dir}')

        this_plot_dir = os.path.join(plt_dir, f'{key}_over_time')
        save_dir = os.path.join(this_plot_dir, f'n_prm_{n_param}.png')
        # points from 10^-3 to highest time value, spaced evenly on a logarithmic plot
        grid = np.power(10,
                        np.linspace(-3,
                                    np.log10(max(relevant_df['time'])),
                                    num=20
                                    ),
                        )
        choose_gaussian_kernel_b = lambda x0: x0
        kernel_smoothed_plot_wrt_value(
            df=relevant_df,
            key=key,
            kernel_fn=lambda x0, x: np.exp(-(x0 - x)**2/(2*choose_gaussian_kernel_b(x0)**2)),
            grid=grid,
            x_param='time',
            save_path=save_dir,
            algs=algs,
            x_log=True,
        )
        print(f'saved to {save_dir}')
