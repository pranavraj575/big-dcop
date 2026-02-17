import os, shutil
import subprocess
import sys
import argparse
import itertools
import time

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

    DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Generate graph coloring instances")
    parser.add_argument("--output_dir", type=str, default=os.path.join(DIR, "output", "graph_coloring_instances_hard"),
                        help="directory to save problem .yaml files")
    parser.add_argument("--num_problems", type=int, default=1, help="Number of graph coloring problems to sample for each case")
    parser.add_argument("--graph_n",type=int,nargs='+',default=list(range(10,51,10)),
                        help='number of nodes in the graph to use (can list multiple)')
    parser.add_argument("--color_count",type=int,nargs='+',default=[3],
                        help='number of colors available (can list multiple)')
    parser.add_argument("--dont_use_seed",action='store_true',
                        help='dont use a random seed (will generate different graphs on each run)')
    parser.add_argument("--dont_clear_dir",action='store_true',
                        help='dont clear the directory listed in --output_dir')
    args = parser.parse_args()
    if not args.dont_clear_dir and os.path.exists(args.output_dir):
        for t in range(5,0,-1):
            print(f"ABOUT TO OVERWRITE {args.output_dir}, if that was not intended, CtrL+C in the next {t} seconds",end='\r')
            time.sleep(1)
        shutil.rmtree(args.output_dir)
    # generate random 3 coloring problems
    for n,c in itertools.product(args.graph_n,args.color_count):
        generate_graph_coloring_problems(
            n_problems=args.num_problems,
            output_dir=args.output_dir,
            node_count=n,       # -v
            color_count=c,       # -c
            graph_type="random", # -g (random, grid, scalefree)
            p_edge=(4.6*n) / (n*(n-1)),           # -p (only used if type is random)
            use_seed=not args.dont_use_seed,
        )

    print("\nDone.")