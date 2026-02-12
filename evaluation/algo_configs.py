def get_display_name(algo_config):
    return algo_config.get('display_name', algo_config['name'])


def get_algo_info(algo_config):
    """
    Helper to convert the dict config into a list of CLI flags
    returns display name, algorithm base name, and list of commands
    """
    assert "name" in algo_config
    # handle variations like "dsa(variant:A)"
    alg_name = algo_config["name"]
    base_name = alg_name.split('(')[0]
    cmd_args = []

    if "dist" in algo_config:
        cmd_args.extend(["--dist", algo_config["dist"]])
    alg_parameters = []
    if '(' in alg_name:
        alg_parameters += alg_name[alg_name.index('(') + 1:-1].split(',')
    if "algo_params" in algo_config:
        alg_parameters += algo_config["algo_params"]
    for param in alg_parameters:
        cmd_args.extend(["--algo_params", param])

    return base_name, cmd_args


if __name__ == '__main__':
    import json

    with open('algorithm_configs.json') as f:
        ALGORITHMS = json.load(f)
        for alg_config in ALGORITHMS:
            print(get_algo_info(alg_config))
