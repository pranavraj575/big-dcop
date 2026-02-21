import argparse, itertools, matplotlib.pyplot as plt
import os.path
import pandas as pd

import numpy as np


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
                ss_n=int(args.subsample)
                if ss_n<len(temp_df):
                    idxs=np.random.default_rng().choice(len(temp_df),ss_n,replace=False)
                    temp_df=temp_df.iloc[idxs]
        g=grid[alg]
        
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
def print_stats_by_alg(df,algs,prefix=''):
    for alg in algs:
        print(prefix+f"algorithm {alg}:\t{len(df[df['algorithm'] == alg])} entries")

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
                   nargs="*",
                   default=None,
                   type=str,
                   help='algorithms to include (defaults to all)',
                   )

    p.add_argument("--grid_n", type=int, default=10, help="number of points to put on plotting grid")
    p.add_argument('--y_keys',
                   required=False,
                   nargs="+",
                   default=['cost'],
                   type=str,
                   help='things to plot on y value',
                   )
    p.add_argument("--subsample", type=float, default=None, help="subsample datapoints. if 0< subsample < 1, samples probabilistically, if subsample>=1, samples n values at random without replacement")
    args = p.parse_args()
    if args.subsample is not None:
        assert args.subsample>0, "--subsample value must be positive value, got {args.subsample}"
    plt_dir = args.output
    os.makedirs(plt_dir, exist_ok=True)

    df = pd.read_csv(args.path)
    print(f'loaded df, length {len(df)}')
    for key in args.y_keys:
        assert key in set(df.keys()), f"key '{key}' is not in data. Valid keys: {set(df.keys())}"

    # get n parameter from problem file name
    df['n'] = df['problem'].map(lambda s: int(s.split('_')[1][1:]))
    if args.algorithms is None:
        algs = sorted(set(df['algorithm']), key=lambda s: s.lower())
    else:
        algs = args.algorithms
    print_stats_by_alg(df,algs)
    
    timeout_params = sorted(set(df['timeout_param']))
    for timeout_param, key in itertools.product(timeout_params,
                                                args.y_keys
                                                ):
        this_plot_dir = os.path.join(plt_dir, f'{key}_over_n')
        save_dir = os.path.join(this_plot_dir, f'timeout_{timeout_param}.png')
        relevant_df = df[df['timeout_param'] == timeout_param]
        # dont consider mid-run data for this plot
        relevant_df = relevant_df[relevant_df['status'] != "RUNNING"]
        relevant_df = relevant_df[relevant_df[key].notnull()]
        plot_wrt_param(
            df=relevant_df,
            key=key,
            x_param='n',
            save_path=save_dir,
            algs=algs,
            title="performance on graph coloring problems",
            args=args,
        )
        print(f'saved to {save_dir}')

    n_params = sorted(set(df['n']))
    for n_param, key in itertools.product(n_params,
                                          args.y_keys
                                          ):
        
        relevant_df = df[df['n'] == n_param]
        relevant_df = relevant_df[relevant_df[key].notnull()]
        print(f'plotting {key} for n={n_param}, {len(relevant_df)} total values')
        print_stats_by_alg(relevant_df,algs,prefix='\t')
        # commented out since we can do better by plotting mid run data
        """
        this_plot_dir = os.path.join(plt_dir, f'{key}_over_timeout')
        save_dir = os.path.join(this_plot_dir, f'n_prm_{n_param}.png')
        plot_wrt_param(
            df=relevant_df[relevant_df['status']!="RUNNING"],
            key=key,
            x_param='timeout_param',
            save_path=save_dir,
            algs=algs,
            x_log=True,
        )
        print(f'saved to {save_dir}')
        """
        this_plot_dir = os.path.join(plt_dir, f'{key}_over_time')
        save_dir = os.path.join(this_plot_dir, f'n_prm_{n_param}.png')
        # points from lowest to highest time value, spaced evenly on a logarithmic plot
        grid = {
            alg: np.exp(np.linspace(np.log(min(relevant_df[relevant_df['algorithm'] == alg]['time'])),
                                    np.log(max(relevant_df[relevant_df['algorithm'] == alg]['time'])),
                                    num=args.grid_n))
            for alg in algs
        }
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
            title="performance on graph coloring problems",
            args=args,
        )
        print(f'saved to {save_dir}')
