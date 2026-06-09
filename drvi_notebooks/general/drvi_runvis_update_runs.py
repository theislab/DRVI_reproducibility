# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: python_apptainer
#     language: python
#     name: python_apptainer
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import argparse
import sys
import traceback
from pathlib import Path
from collections import defaultdict

import anndata as ad
import numpy as np
import wandb

from drvi_notebooks.utils.misc import compare_objs_recursive

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)

if hasattr(sys, 'ps1'):
    # args = parser.parse_args("--wandb_project DRVI_runs__DRVI_5.0 --seed 0".split(" "))
    args = parser.parse_args("--wandb_project DRVI_runs__DRVI_baselines_2.0 --seed 0".split(" "))
else:
    args = parser.parse_args()
print(args)

wandb_project = args.wandb_project
seed = args.seed


# %%
api = wandb.Api()
api.flush()

runs = list(api.runs("moinfar_proj/" + wandb_project))

np.random.seed(seed)
np.random.shuffle(runs)


# %%
def tag_duplicate_runs(wandb_project, entity='moinfar_proj'):
    api = wandb.Api()
    
    print("Fetching runs...")
    # Fetch all runs. Using the API without filters pulls them in
    runs = list(api.runs(f"{entity}/{wandb_project}"))
    
    # Sort runs by creation time so the oldest comes first.
    # We want to keep the oldest and tag the newer ones as duplicates.
    runs.sort(key=lambda r: r.created_at)
    
    # Dictionary to group runs by a fast, hashable signature
    seen_runs_by_sig = defaultdict(list)
    
    duplicate_count = 0

    for run in runs:
        # Skip if it's already tagged to save time
        if run.tags and 'duplicate' in run.tags:
            continue
            
        params = run.config.get('params', {})
        if not params:
            continue
            
        lms_smi = run.summary.get('LMS-SMI')
        
        # 1. Create a fast grouping signature
        # We extract only safe, exact-match types to create a tuple. 
        # Tuples can be hashed and used as dictionary keys.
        safe_items = tuple(
            sorted((k, v) for k, v in params.items() 
            if isinstance(v, (str, int, bool)) and not isinstance(v, float))
        )
        
        is_duplicate = False
        
        # 2. Only compare against older runs that share the same exact safe parameters
        for seen_run in seen_runs_by_sig[safe_items]:
            seen_params = seen_run.config.get('params', {})
            seen_lms_smi = seen_run.summary.get('LMS-SMI')
            
            # Check 1: Exact recursive parameter match (handles floats/NaNs)
            if compare_objs_recursive(params, seen_params, compare_mode='equal'):
                
                # Check 2: LMS-SMI Sanity Check
                smi_matches = False
                if lms_smi is None and seen_lms_smi is None:
                    smi_matches = True # Both missing
                elif lms_smi is not None and seen_lms_smi is not None:
                    # Float comparison with tolerance
                    if abs(float(lms_smi) - float(seen_lms_smi)) < 1e-5:
                        smi_matches = True
                
                if smi_matches:
                    print(f"Tagging {run.name}({run.id}) as duplicate of older run {seen_run.name}({seen_run.id})")
                    is_duplicate = True
                    
                    # --- THE FAST UPDATE TRICK ---
                    # Convert tags to a set to avoid 'duplicate', 'duplicate' stacking
                    current_tags = set(run.tags) if run.tags else set()
                    current_tags.add('duplicate')
                    
                    run.tags = list(current_tags)
                    run.update() # Instantly updates the W&B backend
                    duplicate_count += 1
                    break
                else:
                    print(f"⚠️ WARNING: {run.name} has identical params to {seen_run.name}, but LMS-SMI differs! ({lms_smi} vs {seen_lms_smi}). Not tagging.")
                    
        # 3. If it wasn't a duplicate, add it to the "seen" list so future runs check against it
        if not is_duplicate:
            seen_runs_by_sig[safe_items].append(run)

    print(f"\nFinished! Tagged {duplicate_count} duplicate runs.")

# Run the function
tag_duplicate_runs(wandb_project=wandb_project)

# %%
# for run_info in runs:
#     try:
#         print(f"\nChecking run {run_info.name}({run_info.id})")

#         # Safely get params, defaulting to empty dict if missing
#         params = run_info.config.get('params', {})
#         input_adata = params.get('input_adata', "")
        
#         # Ensure it exists and is a string
#         if not input_adata or not isinstance(input_adata, str):
#             continue

#         print(f"Checking input path for run {run_info.name}({run_info.id})...")
#         if not input_adata.endswith("_pca.h5ad"):
#             print("Skipping...")
#             continue

#         print(f"Fixing {run_info.name}({run_info.id})...")
#         print(input_adata)

#         # --- THE FAST WAY ---
#         # 1. Update the local dictionary
#         params['input_adata'] = input_adata.replace("_pca.h5ad", ".h5ad")
        
#         # 2. Reassign the params back to the config
#         run_info.config['params'] = params
        
#         # 3. Push only the config metadata to the server instantly
#         run_info.update()
        
#         print(f"Updated successfully in a fraction of a second!")

#     except Exception as e:
#         traceback.print_exc()

# %%
# for run_info in runs:
#     try:
#         print(f"\nChecking run {run_info.name}({run_info.id})")

#         input_adata = run_info.config['params'].get('input_adata', None)
#         print(f"Checking input path for run {run_info.name}({run_info.id})...")
#         if not input_adata.endswith("_pca.h5ad"):
#             print("Skipping...")
#             continue

#         print(f"Fixing {run_info.name}({run_info.id})...")
#         print(input_adata)

#         run = wandb.init(
#             project=wandb_project,
#             id=run_info.id,
#             resume=True,
#             entity='moinfar_proj',
#         )

#         params = run_info.config['params']
#         params['input_adata'] = params['input_adata'].replace("_pca.h5ad", ".h5ad")
#         run.config.update({'params': params})
        
#         wandb.finish()
#     except Exception as e:
#         traceback.print_exc()
#         wandb.finish()


# %%
# for run_info in runs:
#     try:
#         print(f"\nChecking run {run_info.name}({run_info.id})")

#         dispersion_param_value = run_info.config['params'].get('dispersion', None)
#         print(dispersion_param_value)
#         if dispersion_param_value is not None:
#             continue
        
#         run = wandb.init(
#             project=wandb_project,
#             id=run_info.id,
#             resume=True,
#             entity='moinfar_proj',
#         )

#         params = run_info.config['params']
#         params['dispersion'] = 'gene'
#         run.config.update({'params': params})
        
#         wandb.finish()
#     except Exception as e:
#         traceback.print_exc()
#         wandb.finish()

# %%
wandb.finish()
# %%
