import json
import yaml


def reformat_file_for_maxsum(problem_file):
    """
    Ensures all agents have a 'capacity' attribute, which is required for the adhoc distribution method.
    """
    try:
        with open(problem_file, "r") as f:
            data = yaml.safe_load(f)
        
        agents = data.get('agents', {})
        needs_update = False
        
        # if agents is a list ['a1', 'a2'], must convert it to a dict
        if isinstance(agents, list):
            new_agents = {}
            for agent_name in agents:
                new_agents[agent_name] = {'capacity': 1000} # give capacity
            data['agents'] = new_agents
            needs_update = True
            
        # if agents dict already, must ensure every agent has a 'capacity' key
        elif isinstance(agents, dict):
            for agent_name, agent_data in agents.items():
                if agent_data is None:
                    agent_data = {}
                    agents[agent_name] = agent_data
                
                # add capacity if missing
                if 'capacity' not in agent_data:
                    agent_data['capacity'] = 1000
                    needs_update = True

        if needs_update:
            with open(problem_file, "w") as f:
                yaml.dump(data, f, sort_keys=False)
                
    except Exception as e:
        pass

def extract_json_from_output(output_str):
    """
    Robustly attempts to find and parse JSON from mixed console output.
    """
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        pass

    try:
        end_idx = output_str.rfind('}')
        if end_idx == -1: return None
        
        # Simple heuristic: scan backwards for start bracket
        # (This is simplistic; a regex or stack-based parser is better for complex nesting,
        # but pydcop output is usually flat enough at the top level).
        start_idx = output_str.find('{')
        
        if start_idx != -1 and start_idx < end_idx:
             json_str = output_str[start_idx : end_idx + 1]
             return json.loads(json_str)
    except:
        pass
    
    return None

def extract_json_from_output(output_str):
    """
    Robustly attempts to find and parse JSON from mixed console output.
    """
    try:
        # 1. Try parsing the whole thing (cleanest case)
        return json.loads(output_str)
    except json.JSONDecodeError:
        pass

    try:
        # 2. Try to find the last occurrence of '{' and matching '}'
        # This assumes the JSON result is the last major block printed.
        # We search from the end.
        end_idx = output_str.rfind('}')
        if end_idx == -1: return None
        
        # Simple heuristic: scan backwards for start bracket
        # (This is simplistic; a regex or stack-based parser is better for complex nesting,
        # but pydcop output is usually flat enough at the top level).
        start_idx = output_str.find('{')
        
        if start_idx != -1 and start_idx < end_idx:
             json_str = output_str[start_idx : end_idx + 1]
             return json.loads(json_str)
    except:
        pass
    
    return None