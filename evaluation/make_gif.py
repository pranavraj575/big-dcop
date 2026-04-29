import json
import os
import subprocess
from PIL import Image
import yaml
import pandas as pd
from evaluation.algo_configs import get_display_name
import networkx as nx
import matplotlib.pyplot as plt


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
    scenario = os.path.join(DIR, "output", "graph_coloring_instances_hard", "gc_n10_k3_random_1.yaml")
    # scenario=os.path.join(DIR,'tests','instances','graph_coloring1.yaml')
    temp_yaml = os.path.join(DIR, "output", "temp_yaml.yaml")
    collect_csv = os.path.join(DIR, "output", "temp_collect_metrics.csv")
    algorithms_dir = os.path.join(DIR, "evaluation", "configs", "algorithm_configs.json")
    plot_dir = os.path.join(DIR, "output", "temp_color_plots")
    with open(algorithms_dir, "r") as f:
        algorithms = json.load(f)
    os.makedirs(plot_dir, exist_ok=True)
    decimals = 1
    max_decimals = 0
    alg_to_info = dict()
    with open(scenario, "r") as f:
        pydcop_dict = yaml.safe_load(f)
    colors = pydcop_dict["domains"]["colors"]["values"]
    var_to_dec = dict()
    for i, variable in enumerate(pydcop_dict["variables"]):
        var_to_dec[variable] = i + decimals
        max_decimals = max(max_decimals, i + decimals)
        prefix = "0." + "0" * (i + decimals - 1)
        cost_fn = ""
        for j, color in enumerate(colors[:-1]):
            cost_fn += f"{prefix}{j} if {variable} == '{color}' else ("
        cost_fn += f"{prefix}{len(colors) - 1}"
        cost_fn += ")" * cost_fn.count("(")
        pydcop_dict["variables"][variable]["cost_function"] = cost_fn
    with open(temp_yaml, "w") as f:
        yaml.safe_dump(pydcop_dict, f)

    G = nx.Graph()
    G.add_edges_from([constraint["variables"] for _, constraint in pydcop_dict["constraints"].items()])
    pos = nx.spring_layout(G, seed=0)

    for algorithm_config in algorithms:
        algo_name = get_display_name(algorithm_config)
        gif_path = os.path.join(plot_dir, algo_name.replace("+", "plus").replace(" ", "_") + ".gif")

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
        cmd += [temp_yaml]
        subprocess.run(cmd, check=True)  # capture_output=True, text=True)

        df = pd.read_csv(collect_csv)
        costs = df["cost"]
        variables = list(var_to_dec.keys())
        plt.axis("off")
        options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 1}
        i = 0
        fns = []
        for cost in costs:
            if pd.isnull(cost):
                continue

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
                node_color = "tab:" + {"B": "blue", "R": "red", "G": "green"}[c]
                nx.draw_networkx_nodes(G, pos, nodelist=var_list_c, node_color=node_color, **options)
            nx.draw_networkx_edges(
                G,
                pos,
                width=1.0,
            )
            fn = os.path.join(plot_dir, str(i) + ".png")
            plt.savefig(fn)
            fns.append(fn)
            plt.close()
            i += 1
        create_gif(image_paths=fns, output_gif_path=gif_path, duration=200)
        for fn in fns:
            os.remove(fn)
