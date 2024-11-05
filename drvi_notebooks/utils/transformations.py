import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def make_pseudobulk(adata: ad.AnnData, col: str, layer: str=None, keep_cols=[], mod='sum') -> ad.AnnData:
    print(f"We do not check {keep_cols} to be unique per '{col}'. Check it yourself.")
    adata.strings_to_categoricals()
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    indicator = pd.get_dummies(adata.obs[col]).T
    indicator_sparse = sparse.csr_matrix(indicator)

    if layer is None:
        data = adata.X
    else:
        adata.layers[layer]
    data = indicator_sparse @ data
    if mod == 'sum':
        pass
    elif mod == 'mean':
        data = data / np.sum(indicator_sparse, axis=1)
    else:
        raise NotImplementedError()
    psuedo_bulk = ad.AnnData(
        data,
        var=adata.var,
        obs=pd.DataFrame({col: indicator.index}, index=indicator.index)
    )
    
    psuedo_bulk.obs = psuedo_bulk.obs.join(adata.obs.drop_duplicates(subset=[col], keep='first').set_index(col)[keep_cols])
    return psuedo_bulk
