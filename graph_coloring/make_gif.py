import json
import os
import subprocess
from PIL import Image
import yaml
import pandas as pd
from graph_coloring.algo_configs import get_display_name
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
        default=os.path.join(DIR, "graph_coloring", "configs", "algorithm_configs.json"),
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
        "--uniform_start",
        action="store_true",
        help="whether to initialize all nodes to the same color (only implemented for RM algorithm)",
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
        "--dpi",
        type=int,
        default=100,
        help="dpi for plotting",
    )
    args = p.parse_args()

    collect_csv = os.path.join(args.plot_temp_dir, "temp_collect_metrics.csv")
    with open(args.algorithms, "r") as f:
        algorithms = json.load(f)
    os.makedirs(args.plot_temp_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)
    # "edgecolors": "tab:gray"
    options = {"edgecolors": "black", "node_size": args.node_size, "alpha": 1}
    options_unassigned = {"node_color": "gray", "node_size": max(args.node_size // 2, 1), "alpha": 1}
    alg_to_info = dict()

    # make pydcop instance, edit colors so we can recover assignment from just costs
    with open(args.scenario, "r") as f:
        pydcop_dict = yaml.safe_load(f)
    colors = pydcop_dict["domains"]["colors"]["values"]

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
        if algorithm_config["name"].startswith("regret_matching") and args.uniform_start:
            cmd.extend(["--algo_param", "deterministic_start:1"])
        cmd += [args.scenario]
        subprocess.run(cmd, check=True)  # capture_output=True, text=True)

        df = pd.read_csv(collect_csv)

        variables = list(pydcop_dict["variables"])
        var_to_color_record = []
        var_to_color = dict()
        for idx, row in df.iterrows():
            var_to_color[row["variable"]] = row["value"]
            var_to_color_record.append((row["time"], var_to_color.copy()))
        max_time = max(t for t, _ in var_to_color_record)

        var_to_color_record = [(0, dict())] * 2 + var_to_color_record + [var_to_color_record[-1]]
        fns = []
        for i, (time, var_to_color) in enumerate(var_to_color_record):
            fig, ax = plt.subplots()
            if time is None:
                continue
            color_to_var = {c: [] for c in colors}
            unused_vars = set(variables)

            for var, c in var_to_color.items():
                color_to_var[c].append(var)
                unused_vars.remove(var)

            for c, var_list_c in color_to_var.items():
                node_color = "tab:" + {"B": "blue", "R": "red", "G": "green", "O": "orange"}[c]
                nx.draw_networkx_nodes(G, pos, nodelist=var_list_c, node_color=node_color, **options)
            # unused nodes here
            nx.draw_networkx_nodes(G, pos, nodelist=list(unused_vars), **options_unassigned)
            edge_colors = []
            edge_style = []
            for u, v in G.edges():
                if (u not in var_to_color) or (v not in var_to_color):
                    edge_colors.append("gray")
                    edge_style.append("--")
                elif var_to_color[u] == var_to_color[v]:
                    edge_colors.append("red")
                    edge_style.append("solid")
                else:
                    edge_colors.append("black")
                    edge_style.append("solid")
            nx.draw_networkx_edges(G, pos, width=2.0, edge_color=edge_colors, style=edge_style)
            fn = os.path.join(args.plot_temp_dir, str(i) + ".png")
            if args.display_time:
                plt.xlabel(f"Time (s): {time}", loc="left", size=13)

                plt.plot((0, 1), (-0.1, -0.1), color="gray", alpha=0.420, lw=5, transform=ax.transAxes, clip_on=False)
                plt.plot(
                    (0, time / max_time), (-0.1, -0.1), color="#420dab", alpha=1, lw=3, transform=ax.transAxes, clip_on=False
                )
            plt.savefig(fn, dpi=args.dpi, bbox_inches="tight")
            plt.close()
            fns.append(fn)
            i += 1
        print(f"saving gif to {gif_path} with {len(fns)} images")
        create_gif(image_paths=fns, output_gif_path=gif_path, duration=args.duration)
        print("saved")
        # clean temp directory
        for fn in fns:
            os.remove(fn)
