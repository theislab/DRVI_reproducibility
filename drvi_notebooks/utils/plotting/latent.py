import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from drvi_notebooks.utils.latent import set_optimal_ordering


cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_10_pallete_without_grey = [c for c in cat_10_pallete if c != '#7f7f7f']
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
cat_100_pallete = sc.plotting.palettes.godsnot_102


def plot_per_latent_scatter(embed, plot_columns, col_name_mapping=None, xy_limit=5, max_cells_to_plot=None, use_var_index=False, 
                            dimensions=None, s=5, markerscale=5, alpha=0.5, save_fn=None, pp_fn=None, zero_lines=False,
                            presence_matrix=None, max_label_to_show=9, predefined_pallete=None, plot_height=5., **kwargs):
    if predefined_pallete is None and presence_matrix is not None:
       assert max_label_to_show <= 9
    if max_cells_to_plot is not None and embed.n_obs > max_cells_to_plot:
        embed = sc.pp.subsample(embed, n_obs=max_cells_to_plot, copy=True)
    data = pd.DataFrame(embed.X)
    data = data.clip(-xy_limit * 0.98, xy_limit * 0.98)
    if use_var_index:
        dim_names = embed.var.index.astype(str)
        data.columns = dim_names
    else:
        dim_names = [f'Dim {i+1}' for i in range(embed.n_vars)]
        data.columns = dim_names
    data.index = embed.obs.index
    for col in plot_columns:
        data[col] = embed.obs[col]

    if dimensions is None:
        if 'optimal_var_order' not in embed.uns:
            set_optimal_ordering(embed)
        optimal_ordering = embed.uns['optimal_var_order']
        dimensions = [x.tolist() for x in np.array_split(optimal_ordering, len(optimal_ordering) // 2)]

    numerical_columns = [col for col in plot_columns if data[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
    
    n_latent = embed.shape[1]
    for i, j in dimensions:
        dim_i = dim_names[i]
        dim_j = dim_names[j]
        print(f"plotting {dim_i}, {dim_j}")
    
        for col in plot_columns:
            original_col = col
            palette = None
            if col in numerical_columns:
                continue
            elif predefined_pallete is not None:
                palette = predefined_pallete
            elif len(data[col].unique()) <= 10:
                palette = dict(zip(list(sorted(data[col].unique())), cat_10_pallete))
            elif len(data[col].unique()) <= 20:
                palette = dict(zip(list(sorted(data[col].unique())), cat_20_pallete))
            elif len(data[col].unique()) <= 102:
                palette = dict(zip(list(sorted(data[col].unique())), cat_100_pallete))
            else:
                palette = None

            present_cats = None
            if presence_matrix is not None and col in presence_matrix:
                current_presence_matrix = presence_matrix[col]
                assert current_presence_matrix.shape[0] == n_latent
                assert set(current_presence_matrix.columns) == set(data[col].astype(str).unique())
                # max_presence = list(set((
                #     list(current_presence_matrix.iloc[i, :].sort_values(ascending=False)[:max_label_to_show//2].index) +
                #     list(current_presence_matrix.iloc[j, :].sort_values(ascending=False)[:max_label_to_show//2].index)
                # )))
                max_presence = list(current_presence_matrix.iloc[[i,j], :]
                    .max(axis=0).sort_values(ascending=False)[:max_label_to_show].index)
                data[f'relevant_{col}'] = data[col].apply(lambda x: x if x in max_presence else 'Other')
                col_name_mapping[f'relevant_{col}'] = f'Relevant {col_name_mapping[col]}s'
                col = f'relevant_{col}'
                present_cats = max_presence + ['Other']
                if predefined_pallete is None:
                    palette = {category: color for category, color in zip(max_presence, cat_10_pallete_without_grey)}
                    palette['Other'] = '#7f7f7f'
                else:
                    palette = {category: predefined_pallete[category] for category in max_presence}
                    palette['Other'] = '#7f7f7f'
                
                
            g = sns.jointplot(
                data=data, x=dim_i, y=dim_j, 
                hue=col, linewidth=0,
                palette=palette,
                joint_kws=dict(s=s, alpha=alpha),
                marginal_kws=dict(common_norm=False),
                height=plot_height,
                **{**dict(
                    hue_order=present_cats,
                ), **kwargs
                }
            )
            g.ax_marg_x.set_xlim(-xy_limit, xy_limit)
            g.ax_marg_y.set_ylim(-xy_limit, xy_limit)
            g.ax_joint.legend(title=col_name_mapping[col] if col_name_mapping is not None else col, 
                              loc='upper left', ncol=(len(data[col].unique()) // 25 + 1),
                              bbox_to_anchor=(1.3, 1), markerscale=markerscale, frameon=False) 
            g.ax_marg_x.grid(False)
            g.ax_marg_y.grid(False)
            g.ax_joint.grid(False)
            g.ax_joint.collections[0].set_rasterized(True)
            if zero_lines:
                g.ax_joint.axhline(y=0, linewidth=1, color='grey', ls=':', zorder=-10)
                g.ax_joint.axvline(x=0, linewidth=1, color='grey', ls=':', zorder=-10)
            if pp_fn is not None:
                pp_fn(g)
            if save_fn is not None:
                fig = g.fig
                save_fn(fig, dim_i, dim_j, original_col)
            plt.show()
        if len(numerical_columns) > 0:
            pca_adata = ad.AnnData(data[[dim_i, dim_j]].to_numpy(), obs=data[numerical_columns])
            pca_adata.obsm['X_pca']= np.concatenate([pca_adata.X, pca_adata.X, pca_adata.X], axis=1)
            sc.pl.pca(pca_adata, color=numerical_columns)
            plt.show()
        # break
    print("End of latent plots")


def scatter_plot_per_latent(embed, plot_key, plot_columns, xy_limit=5, max_cells_to_plot=None, dim_names='Dim', 
                            dimensions=None, predefined_pallete=None, **kwargs):
    if max_cells_to_plot is not None and embed.n_obs > max_cells_to_plot:
        embed = sc.pp.subsample(embed, n_obs=max_cells_to_plot, copy=True)
    if dimensions is None:
        if 'optimal_var_order' not in embed.uns:
            set_optimal_ordering(embed)
        optimal_ordering = embed.uns['optimal_var_order']
        dimensions = [x.tolist() for x in np.array_split(optimal_ordering, len(optimal_ordering) // 2)]
    if xy_limit is not None:
        embed.obsm[f'{dim_names} '] = embed.obsm[plot_key].clip(-xy_limit * 0.98, xy_limit * 0.98)
    else:
        embed.obsm[f'{dim_names} '] = embed.obsm[plot_key]
    for col in plot_columns:
        print(col)
        unique_values = list(sorted(list(embed.obs[col].astype(str).unique())))
        if predefined_pallete is not None:
            palette = predefined_pallete
        elif len(unique_values) <= 10:
            palette = dict(zip(unique_values, cat_10_pallete))
        elif len(unique_values) <= 20:
            palette = dict(zip(unique_values, cat_20_pallete))
        elif len(unique_values) <= 102:
            palette = dict(zip(unique_values, cat_100_pallete))
        else:
            palette = None
        fig = sc.pl.embedding(embed, f'{dim_names} ', color=col, dimensions=dimensions, palette=palette, legend_loc=None,
                              return_fig=True, **kwargs)
        for ax in fig.axes:
            if hasattr(ax, '_colorbar'):
                continue
            if xy_limit is not None:
                ax.set_xlim(-xy_limit, xy_limit)
                ax.set_ylim(-xy_limit, xy_limit)
            ax.set_title("")
        yield plt