import math
from typing import Literal, Any

def compare_objs_recursive(obj1, obj2, compare_mode: Literal['equal', 'left_in_right', 'right_in_left'] = 'equal'):
    if compare_mode == 'right_in_left':
        return compare_objs_recursive(obj2, obj1, compare_mode='left_in_right')

    if isinstance(obj1, float) and math.isnan(obj1):
        obj1 = None
    if isinstance(obj2, float) and math.isnan(obj2):
        obj2 = None
    if obj1 is None and obj2 is None:
        return True
    
    if obj1 is None and obj2 is not None:
        return compare_mode == 'left_in_right'
    if obj1 is not None and obj2 is None:
        return False
    
    if isinstance(obj1, (int, float, bool)) and isinstance(obj2, (int, float, bool)):
        if abs(float(obj1) - float(obj2)) < 1e-10:
            return True
        else:
            return False
    elif type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(compare_objs_recursive(item1, item2) for item1, item2 in zip(obj1, obj2))
    
    if isinstance(obj1, dict):
        keys1, keys2 = set(obj1.keys()), set(obj2.keys())
        if compare_mode == 'left_in_right':
            if not set(obj1.keys()).issubset(set(obj2.keys())):
                return False
        else:
            if keys1 != keys2:
                return False
        return all(compare_objs_recursive(obj1[key], obj2[key]) for key in keys1)

    return obj1 == obj2


def check_wandb_run(api, params, wandb_project, wandb_key='params', true_states=tuple(['finished', 'running']), ignore_tags=tuple()):
    if get_wandb_run(api, params, wandb_project, wandb_key, true_states, ignore_tags) is not None:
        return True
    return False


def get_wandb_run(
    api, 
    params: dict[str, Any], 
    wandb_project: str, 
    wandb_key: str = 'params', 
    true_states: tuple[str] = ('finished', 'running'), 
    ignore_tags: tuple[str] = ()
):
    api.flush()
    
    mongo_filters = {}
    
    if true_states:
        mongo_filters["state"] = {"$in": list(true_states)}
        
    if ignore_tags:
        mongo_filters["tags"] = {"$nin": list(ignore_tags)}
        
    if params and isinstance(params, dict):
        for key, value in params.items():
            # Strict check: Must be str, int, or bool. Must NOT be float.
            if isinstance(value, (str, int, bool)) and not isinstance(value, float):
                config_path = f"config.{wandb_key}.{key}" if wandb_key else f"config.{key}"
                mongo_filters[config_path] = value

    try:
        runs = api.runs(wandb_project, filters=mongo_filters)
    except Exception as e:
        print(f"Failed to fetch runs: {e}")
        return None
        
    for run in runs:
        if run.tags and any(tag in ignore_tags for tag in run.tags):
            continue
        if run.state not in true_states:
            continue
            
        run_config = run.config.get(wandb_key, {}) if wandb_key else run.config
        
        if compare_objs_recursive(params, run_config, compare_mode='left_in_right'):
            return run
            
    return None


def get_runs_by_model_tags(api, reference_projects: list, model_to_tag_map: dict) -> dict:
    """
    Fetches W&B runs across multiple projects based on a dictionary of Model -> Tag mappings.
    
    Args:
        api (wandb.Api): The W&B API object.
        reference_projects (list): List of project paths e.g., ["my-entity/proj_x", "my-entity/proj_y"]
        model_to_tag_map (dict): Dictionary mapping model names to their corresponding W&B tags.
        
    Returns:
        dict: A dictionary mapping the model names to a list of matching wandb.Run objects.
    """
    # Initialize the results dictionary
    results = {model: [] for model in model_to_tag_map.keys()}
    
    # Create a reverse lookup dictionary (Tag -> List of Models)
    # This is useful just in case multiple models share the same tag
    tag_to_models = {}
    for model, tag in model_to_tag_map.items():
        if tag not in tag_to_models:
            tag_to_models[tag] = []
        tag_to_models[tag].append(model)
        
    # Extract the unique tags we actually care about
    target_tags = list(tag_to_models.keys())
    
    # Iterate through projects
    for project in reference_projects:
        # Performant Server-Side Filter: Only fetch runs that contain at least one of our target tags
        runs = api.runs(
            path=project, 
            filters={"tags": {"$in": target_tags}}
        )
        
        # Iterate over the matching runs (should only be 1-3 per model as you mentioned)
        for run in runs:
            # Check which of our target tags are present in this specific run
            for tag in run.tags:
                if tag in tag_to_models:
                    # Append the run to the corresponding model(s)
                    for model in tag_to_models[tag]:
                        results[model].append(run)
                        
    return results
