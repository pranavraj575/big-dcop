import argparse, itertools, matplotlib.pyplot as plt
import os.path
import pandas as pd

import numpy as np
import json
from algo_configs import get_display_name


def kernel_smoothed_plot_wrt_value(df,
                                   key,
                                   x_param,
                                   save_path,
                                   kernel_fn,
                                   args,
                                   z_score=1.96,
                                   grid=None,
                                   x_log=False,
                                   y_log=False,
                                   algs=None,
                                   title=None,
                                   ):
    """
    Parameters
    ----------
    df
    key
    x_param
    save_path
    kernel_fn: function of K(x0,x), x's weight when estimating x0
        for std_error to make sense, should have K(x0,x0)=1
    args
    z_score
    grid: grid of x values to use when plotting
        can also be dict of algorithm to grid
    x_log
    y_log
    algs
    title

    Returns
    -------

    """
    if algs is None:
        algs = sorted(set(df['algorithm']), key=lambda s: s.lower())
    if type(grid) != dict:
        grid = {alg: grid for alg in algs}
    styles = set()
    for alg in algs:
        temp_df = df[df['algorithm'] == alg]
        if args.subsample is not None:
            if args.subsample < 1.:
                temp_df = temp_df[np.random.random(len(temp_df)) < args.subsample]
            else:
                ss_n = int(args.subsample)
                if ss_n < len(temp_df):
                    idxs = np.random.default_rng().choice(len(temp_df), ss_n, replace=False)
                    temp_df = temp_df.iloc[idxs]
        g = grid[alg]

        if g is None:
            g = sorted(set(temp_df[x_param]))
        means = []
        std_errors = []
        for x0 in g:
            weights = temp_df[x_param].map(lambda x: kernel_fn(x0, x))
            y = temp_df[key]
            cum_weight = weights.sum()
            mu = (weights*y).sum()/cum_weight
            means.append(mu)
            sample_variance = (((y - mu)**2)*weights).sum()/(cum_weight - 1)
            std_error = np.sqrt(sample_variance)/np.sqrt(cum_weight)
            std_errors.append(std_error)
        means, std_errors = np.array(means), np.array(std_errors)
        t, = plt.plot(g, means, label=alg)
        style = (t.get_c(), t.get_linestyle())
        if style in styles:
            # todo: can make this more fancy, but this works up until 2*(# of colors) = 20 lines
            t.set_linestyle('--')
        style = (t.get_c(), t.get_linestyle())
        styles.add(style)

        plt.fill_between(g, means - std_errors*z_score, means + std_errors*z_score, color=t.get_c(), alpha=.2)
    plt.legend()
    plt.ylabel(key)
    plt.xlabel(x_param)
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
    plt.title(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_wrt_param(df, key, x_param, save_path, args, x_log=False, y_log=False, algs=None, title=None):
    kernel_smoothed_plot_wrt_value(df=df,
                                   key=key,
                                   x_param=x_param,
                                   save_path=save_path,
                                   kernel_fn=lambda x0, x: int(x0 == x),
                                   args=args,
                                   grid=None,
                                   x_log=x_log,
                                   y_log=y_log,
                                   algs=algs,
                                   title=title,
                                   )


def print_stats_by_alg(df, algs, prefix=''):
    for alg in algs:
        print(prefix + f"algorithm {alg}:\t{len(df[df['algorithm'] == alg])} entries")


if __name__ == '__main__':
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        required=False,
        default=os.path.join(DIR, "output", "results_rm_scalefree.csv"),
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
                   default=None,
                   type=str,
                   help='json file with algorithms to include (defaults to using all algorithms found in data)',
                   )

    p.add_argument("--grid_n", type=int, default=10, help="number of points to put on plotting grid")
    p.add_argument('--y_keys',
                   required=False,
                   nargs="+",
                   default=['cost'],
                   type=str,
                   help='things to plot on y value, to specifiy log scale, use <key>:log',
                   )
    p.add_argument("--subsample", type=float, default=None,
                   help="subsample datapoints. if 0< subsample < 1, samples probabilistically, if subsample>=1, samples n values at random without replacement")
    args = p.parse_args()
    if args.subsample is not None:
        assert args.subsample > 0, f"--subsample value must be positive value, got {args.subsample}"
    plt_dir = args.output
    os.makedirs(plt_dir, exist_ok=True)

    df = pd.read_csv(args.path)
    print(f'loaded df, length {len(df)}')
    keys = []
    for key in args.y_keys:
        configs = []
        if ':' in key:
            key, m = key.split(':')
            configs = m.split(',')
        keys.append((key, configs))
        assert key in set(
            df.keys()), f"key '{key}' with configs {configs} is not in data. Valid keys: {set(df.keys())}"

    # ADD ROWS TO DF
    # get n parameter from problem file name
    df['n'] = df['problem'].map(lambda s: int(s.split('_')[1][1:]))
    n_params = sorted(set(df['n']))
    # get graph type parameter from problem file name
    df['graph_type'] = df['problem'].map(lambda s: s.split('_')[3])
    graph_types = sorted(set(df['graph_type']))
    # rescale y values by dividing by average value across each n param
    for (key, configs) in keys:
        df['rescaled_' + key] = df[key]
        relevant_df = df[df[key].notnull()]
        for n_param in n_params:
            rows = (relevant_df['n'] == n_param)
            relevant_df.loc[rows, 'rescaled_' + key] = relevant_df.loc[rows, key]/np.mean(relevant_df.loc[rows, key])
    if args.algorithms is None:
        algs = sorted(set(df['algorithm']), key=lambda s: s.lower())
    else:
        with open(args.algorithms) as f:
            algs = [get_display_name(alg_config) for alg_config in json.load(f)]

    print_stats_by_alg(df, algs)

    choose_gaussian_kernel_b = lambda x0: x0
    gaussian_kernel = lambda x0, x: np.exp(-(x0 - x)**2/(2*choose_gaussian_kernel_b(x0)**2))

    timeout_params = sorted(set(df['timeout_param']))

    # plot the plots wrt n, splitting by values of timeout_param
    for (key, configs) in keys:
        this_plot_dir = os.path.join(plt_dir, f'{"_".join(configs)}{key}_over_n')
        save_dir = os.path.join(this_plot_dir, f'plot.png')
        # dont consider mid-run data for this plot
        relevant_df = df[df['status'] != "RUNNING"]
        relevant_df = relevant_df[relevant_df[key].notnull()]
        plot_wrt_param(
            df=relevant_df,
            key=key,
            x_param='n',
            save_path=save_dir,
            algs=algs,
            y_log='log' in configs,
            title="performance on graph coloring problems",
            args=args,
        )
        print(f'saved to {save_dir}')
    for graph_config in ({'split_by': [('n', n_params), ]},
                         {'split_by': []},
                         {'split_by': [('graph_type', graph_types), ]},
                         {'key_modifier': lambda key: f'rescaled_{key}',
                          'prefix': 'rescaled_'
                          }
                         ):
        if graph_config.get('split_by'):
            split_by_keys, split_by_values = list(zip(*graph_config['split_by']))
        else:
            split_by_keys, split_by_values = (), ()
        split_by_values = split_by_values + (keys,)
        for splits in itertools.product(*split_by_values):
            key, configs = splits[-1]
            if 'key_mod' in graph_config:
                real_key = graph_config['key_mod'](key)
            else:
                real_key = key

            relevant_df = df[df[real_key].notnull()]
            for k, v in zip(split_by_keys, splits):
                relevant_df = relevant_df[relevant_df[k] == v]
            split_str = {k: v for k, v in zip(split_by_keys, splits)}
            print(f'plotting {key} for {split_str}, {len(relevant_df)} total values')
            print_stats_by_alg(relevant_df, algs, prefix='\t')

            this_plot_dir = os.path.join(plt_dir, f'{"_".join(configs)}{key}_over_time')
            if not split_by_keys:
                plt_name = 'combined_plot'
            else:
                plt_name = '_'.join([f'{k}_{v}' for k, v in zip(split_by_keys, splits)])
            plt_name = graph_config.get('prefix', '') + plt_name
            save_dir = os.path.join(this_plot_dir, f'{plt_name}.png')
            # points from lowest to highest time value, spaced evenly on a logarithmic plot
            grid = {
                alg: np.exp(np.linspace(np.log(min(relevant_df[relevant_df['algorithm'] == alg]['time'])),
                                        np.log(max(relevant_df[relevant_df['algorithm'] == alg]['time'])),
                                        num=args.grid_n))
                for alg in algs
            }

            kernel_smoothed_plot_wrt_value(
                df=relevant_df,
                key=real_key,
                kernel_fn=gaussian_kernel,
                grid=grid,
                x_param='time',
                save_path=save_dir,
                algs=algs,
                x_log=True,
                y_log='log' in configs,
                title="performance on graph coloring problems",
                args=args,
            )
            print(f'saved to {save_dir}')
