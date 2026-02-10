# algos to evaluate
ALGORITHMS = [
    "dsa",
    "mgm",
    "maxsum",
    "ftrl",
    "regret_matching",
    "dpop"
]

# map algo name to arguments
ALGO_CONFIG = {
    "dpop": {},
    "maxsum": {
        "dist": "adhoc" 
    },
    "dsa": {
        "algo_params": ["stop_cycle:100", "variant:B"] 
    },
    "mgm": {
        "algo_params": ["stop_cycle:100"]
    },
    "ftrl": {
        "algo_params": ["stop_cycle:100"]
    },
    "regret_matching": {
        "algo_params": ["stop_cycle:100"]
    },
}

def get_algo_args(algo_name):
    """
    Helper to convert the dict config into a list of CLI flags 
    """
    # handle variations like "dsa(variant:A)"
    base_name = algo_name.split('(')[0]     
    config = ALGO_CONFIG.get(base_name, {})
    
    cmd_args = []
    
    if "dist" in config:
        cmd_args.extend(["--dist", config["dist"]])
        
    if "algo_params" in config:
        for param in config["algo_params"]:
            cmd_args.extend(["--algo_params", param])
            
    return cmd_args