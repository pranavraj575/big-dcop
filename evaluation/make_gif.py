import json
import os
import subprocess
from PIL import Image
import yaml
import pandas as pd
from evaluation.algo_configs import get_display_name
import networkx as nx
import matplotlib.pyplot as plt
import argparse


def create_gif(image_paths, output_gif_path, duration=200):
    images = [Image.open(image_path) for image_path in image_paths]
    if all(im.size == images[0].size for im in images):
        # no size shenanigans needed, just save as gif
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,  # 0 means infinite loop
        )

    W, H = max([im.width for im in images]), max([im.height for im in images])

    # get background color
    # im.getcolors() returns unsorted list of (count, color)
    colors = images[0].getcolors(maxcolors=images[0].width * images[0].height)
    assert colors is not None
    _, background_color = max(colors, key=lambda x: x[0])

    # resize images to the maximum image size
    def resized(im):
        resized = Image.new(mode=im.mode, size=(W, H), color=background_color)
        resized.paste(im)
        return resized

    resized(images[0]).save(
        output_gif_path,
        save_all=True,
        append_images=(resized(im) for im in images[1:]),
        duration=duration,
        loop=0,
    )


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()

    p.add_argument(
        "scenario",
        type=str,
        help="yaml file with graph coloring scenario (run graph_coloring_generator.py for this)",
    )
    p.add_argument(
        "--algorithms",
        type=str,
        default=os.path.join(DIR, "evaluation", "configs", "algorithm_configs.json"),
        help="json file with algorithm configs",
    )
    p.add_argument(
        "--plot_temp_dir",
        type=str,
        default=os.path.join(DIR, "output", "temp_color_plots"),
        help="directory to save temporary plots",
    )
    p.add_argument(
        "--gif_dir",
        type=str,
        default=os.path.join(DIR, "output", "graph_color_gifs"),
        help="directory to save gifs",
    )

    p.add_argument(
        "--display_time",
        action="store_true",
        help="whether to display the clock time at each frame",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=300,
        help="duration of gif frames",
    )
    p.add_argument(
        "--node_size",
        type=int,
        default=800,
        help="size of nodes",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for node positions",
    )
    p.add_argument(
        "--start_decimals",
        type=int,
        default=1,
        help="number of decimals to start the positional encoding of colors",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="dpi for plotting",
    )
    args = p.parse_args()

    collect_csv = os.path.join(args.plot_temp_dir, "temp_collect_metrics.csv")
    temp_yaml = os.path.join(args.plot_temp_dir, "temp_yaml.yaml")
    with open(args.algorithms, "r") as f:
        algorithms = json.load(f)
    os.makedirs(args.plot_temp_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)
    max_decimals = 0
    alg_to_info = dict()

    # make pydcop instance, edit colors so we can recover assignment from just costs
    with open(args.scenario, "r") as f:
        pydcop_dict = yaml.safe_load(f)
    colors = pydcop_dict["domains"]["colors"]["values"]
    var_to_dec = dict()
    for i, variable in enumerate(pydcop_dict["variables"]):
        var_to_dec[variable] = i + args.start_decimals
        max_decimals = max(max_decimals, i + args.start_decimals)
        prefix = "0." + "0" * (i + args.start_decimals - 1)
        cost_fn = ""
        for j, color in enumerate(colors[:-1]):
            cost_fn += f"{prefix}{j} if {variable} == '{color}' else ("
        cost_fn += f"{prefix}{len(colors) - 1}"
        cost_fn += ")" * cost_fn.count("(")
        pydcop_dict["variables"][variable]["cost_function"] = cost_fn
    with open(temp_yaml, "w") as f:
        yaml.safe_dump(pydcop_dict, f)
    # make graph, precompute positions
    G = nx.Graph()
    G.add_edges_from([constraint["variables"] for _, constraint in pydcop_dict["constraints"].items()])
    pos = nx.spring_layout(G, seed=args.seed)

    # do computation for each algorithm
    for algorithm_config in algorithms:
        algo_name = get_display_name(algorithm_config)
        gif_path = os.path.join(args.gif_dir, algo_name.replace("+", "plus").replace(" ", "_") + ".gif")

        cmd = [
            "pydcop",
            "solve",
            "--algo",
            algorithm_config["name"],
            "--run_metrics",
            collect_csv,
            "--collect_on",
            "value_change",
        ]

        if "algo_params" in algorithm_config:
            for param in algorithm_config["algo_params"]:
                cmd.extend(["--algo_param", param])
        if algorithm_config["name"].startswith("regret_matching"):
            cmd.extend(["--algo_param", "deterministic_start:1"])
        cmd += [temp_yaml]
        subprocess.run(cmd, check=True)  # capture_output=True, text=True)

        df = pd.read_csv(collect_csv)
        df = df[pd.notna(df["cost"])]
        variables = list(var_to_dec.keys())
        options = {"edgecolors": "tab:gray", "node_size": args.node_size, "alpha": 1}
        i = 0
        fns = []
        costs = list(df["cost"])
        times = list(df["time"])
        # add to the start the initial deterministic start
        if algorithm_config["name"].startswith("regret_matching"):
            costs = [int(costs[0])] + costs
            times = [0] + times
        # linger on initial frame for longer
        costs = [costs[0]] + costs
        times = [times[0]] + times

        # recover colorings from just the cost, generate gif from frames
        for time, cost in zip(times, costs):
            color_to_var = {c: [] for c in colors}
            for var in var_to_dec:
                t_cost = round(cost, max_decimals)
                # prevent floating point errors by shifting one at a time
                # idx=int(cost*(10**var_to_dec[var]))%10
                for _ in range(var_to_dec[var]):
                    t_cost = round((10 * t_cost) % 10, max_decimals)
                c = colors[int(t_cost)]
                color_to_var[c].append(var)
            for c, var_list_c in color_to_var.items():
                node_color = "tab:" + {"B": "blue", "R": "red", "G": "green", "O": "orange"}[c]
                nx.draw_networkx_nodes(G, pos, nodelist=var_list_c, node_color=node_color, **options)
            nx.draw_networkx_edges(
                G,
                pos,
                width=1.0,
            )
            fn = os.path.join(args.plot_temp_dir, str(i) + ".png")
            if args.display_time:
                plt.xlabel(f"Time (s): {time}", loc="left", size=13)
            plt.savefig(fn, dpi=args.dpi)
            plt.close()
            fns.append(fn)
            i += 1
        create_gif(image_paths=fns, output_gif_path=gif_path, duration=args.duration)
        # clean temp directory
        for fn in fns:
            os.remove(fn)
