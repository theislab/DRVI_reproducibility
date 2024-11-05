from typing import Literal

def compare_objs_recursive(obj1, obj2, compare_mode: Literal['equal', 'left_in_right', 'right_in_left'] = 'equal'):
    if compare_mode == 'right_in_left':
        return compare_objs_recursive(obj2, obj1, compare_mode='left_in_right')
    
    if obj1 is None and obj2 is not None:
        return compare_mode == 'left_in_right'
    if obj1 is not None and obj2 is None:
        return False
    
    if isinstance(obj1, (int, float, bool)):
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
    api.flush()
    try:
        runs = api.runs(wandb_project)
    except:
        return False
    for run in runs:
        if run.tags and any([tag in ignore_tags for tag in run.tags]):
            continue
        if compare_objs_recursive(params, run.config[wandb_key], compare_mode='left_in_right'):
            if run.state in true_states:
                return True
    return False


def get_wandb_run(api, params, wandb_project, wandb_key='params', true_states=tuple(['finished', 'running']), ignore_tags=tuple()):
    api.flush()
    try:
        runs = api.runs(wandb_project)
    except:
        return None
    for run in runs:
        if run.tags and any([tag in ignore_tags for tag in run.tags]):
            continue
        if compare_objs_recursive(params, run.config[wandb_key], compare_mode='left_in_right'):
            if run.state in true_states:
                return run
    return None
