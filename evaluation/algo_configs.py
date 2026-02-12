# algos to evaluate
# algorithm display name, dict of config
"""
dict keys:
    alg_name: if specified, overrides first name (first name will be used for display)
    algo_params: list of algorithm params in format <param>:<value>
"""
ALGORITHMS = [
    ("dsa", {"algo_params": ["stop_cycle:100", "variant:B"]}),
    ("mgm", {"algo_params": ["stop_cycle:100"]}),
    ("maxsum", {"dist": "adhoc"}),
    ("ftrl", {"algo_params": ["stop_cycle:100"]}),
    ("RM", {"alg_name": 'regret_matching', "algo_params": ["stop_cycle:100"]}),
    ("dpop", {}),
]


def get_algo_info(algorithm, config):
    """
    Helper to convert the dict config into a list of CLI flags
    returns display name, algorithm base name, and list of commands
    """
    # handle variations like "dsa(variant:A)"
    display_name = algorithm
    if 'alg_name' in config:
        algorithm = config['alg_name']
    base_name = algorithm.split('(')[0]
    cmd_args = []

    if "dist" in config:
        cmd_args.extend(["--dist", config["dist"]])
    alg_parameters = []
    if '(' in algorithm:
        alg_parameters += algorithm[algorithm.index('(') + 1:-1].split(',')
    if "algo_params" in config:
        alg_parameters += config["algo_params"]
    for param in alg_parameters:
        cmd_args.extend(["--algo_params", param])

    return display_name, base_name, cmd_args


if __name__ == '__main__':
    for alg, config in ALGORITHMS:
        print(get_algo_info(alg, config))
