import os
import subprocess
import sys

def generate_graph_coloring_problems(
    n_problems: int,
    output_dir: str,
    node_count: int = 20,
    color_count: int = 3,
    graph_type: str = "random",  # random, grid, scalefree
    p_edge: float = 0.4,         # probability of edge (for random graphs)
    m_edge: int = 2,             # edges per node (for scalefree graphs)
    use_seed= False,             # whether to use fixed random seed
):

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            pass

    print(f"Generating {n_problems} problems in '{output_dir}'...")

    for i in range(1, n_problems + 1):
        filename = f"gc_n{node_count}_k{color_count}_{graph_type}_{i}.yaml"
        filepath = os.path.join(output_dir, filename)

        base_cmd = ["pydcop", "generate", "graph_coloring"]
        cmd_args = [
            "--variables_count", str(node_count),   # -v
            "--colors_count", str(color_count),     # -c
            "--graph", graph_type,                  # -g
        ]

        # Add specific parameters based on graph type
        if graph_type == "random":
            cmd_args.extend(["--p_edge", str(p_edge)]) # -p
        elif graph_type == "scalefree":
            cmd_args.extend(["--m_edge", str(m_edge)]) # -m

        if use_seed:
            cmd_args.extend(["--seed", str(i)]) # -m

        cmd = base_cmd + cmd_args

        try:

            with open(os.path.join(output_dir, filename), "w") as outfile:
                subprocess.run(cmd, stdout=outfile, check=True)
            print(f"[{i}/{n_problems}] Generated: {filename}")

        except subprocess.CalledProcessError as e:
            print(e.stderr)


if __name__ == "__main__":

    INPUT_DIR = "output/graph_coloring_instances"
    NUM_PROBLEMS = 5

    # generate random 3 coloring problems
    for n in range(10, 51, 10):
        generate_graph_coloring_problems(
            n_problems=NUM_PROBLEMS,
            output_dir=INPUT_DIR,
            node_count=n,       # -v
            color_count=3,       # -c
            graph_type="random", # -g (random, grid, scalefree)
            p_edge=0.5,           # -p (only used if type is random)
            use_seed=True,
        )

    print("\nDone.")