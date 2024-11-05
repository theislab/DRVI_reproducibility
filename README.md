# DRVI reproducibility

# Requirements
Install dependencies in requirements.txt

Then run the following commands to be able to run py files as notebooks:
```commandline
jupyter nbextension install jupytext --user --py
jupyter nbextension enable jupytext --user --py
```

Install the reproducibility package
```commandline
git clone https://gitlab.com/moinfar/drvi_reproducibility.git
cd drvi_reproducibility
pip install -e .
```

Install Rapids and rapids-singlecell package for faster scanpy GPU accelerated functions.
Read more about Rapids installation (here)[https://docs.rapids.ai/install].
```commandline
pip install rapids-singlecell  # Already in requirements

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==23.12.*" "dask-cudf-cu12==23.12.*" "cuml-cu12==23.12.*" \
    "cugraph-cu12==23.12.*" "cuspatial-cu12==23.12.*" "cuproj-cu12==23.12.*" \
    "cuxfilter-cu12==23.12.*" "cucim-cu12==23.12.*" "pylibraft-cu12==23.12.*" \
    "raft-dask-cu12==23.12.*"
```

## Authors and acknowledgment

## License
