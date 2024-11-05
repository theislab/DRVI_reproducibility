import os
from pathlib import Path

import pandas as pd
import numpy as np
import scanpy as sc
from drvi.utils.hvg import hvg_batch


class ImmuneDataCleaner:
    def __init__(
            self,
            input_file,
            output_directory,
            condition_key="batch",
            cell_type_key='final_annotation',
            drop_conditions=tuple(['Villani']),
            n_hvg=2000,
            sample_frac=1,
    ):
        self.input_file = Path(input_file)
        self.output_directory = Path(output_directory)
        self.condition_key = condition_key
        self.cell_type_key = cell_type_key
        self.drop_conditions = drop_conditions
        self.n_hvg = n_hvg
        self.sample_frac = sample_frac

    @property
    def outputs(self):
        return {
            'rna': self.output_directory / "adata.h5ad",
            'rna_hvg': self.output_directory / "adata_hvg.h5ad",
        }

    def read(self):
        adata = sc.read(self.input_file)

        if self.sample_frac < 1:
            adata.obs['_keep'] = np.random.uniform(0, 1, size=adata.n_obs)
            adata = adata[adata.obs['_keep'] < self.sample_frac].copy()
            del adata.obs['_keep']
        return adata

    def clean(self, adata):
        adata = adata[~adata.obs[self.condition_key].isin(self.drop_conditions)]
        adata.layers['counts'] = adata.layers['counts'].round()  # No need if we drop Villani
        return adata

    def enrich_layers(self, adata):
        adata.layers['lognorm'] = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata, layer='lognorm')
        sc.pp.log1p(adata, layer='lognorm')

    def hvg(self, adata):
        hvg_genes = hvg_batch(adata, batch_key=self.condition_key, target_genes=self.n_hvg, adataOut=False)
        adata_hvg = adata[:, hvg_genes].copy()
        return adata_hvg, hvg_genes

    def calculate_gene_activities(self, adata):
        # TODO: Improve this
        def rank_genes_groups_df(adata, key=None):
            key = key or "rank_genes_groups"
            groups = adata.uns[key]['scores'].dtype.names
            d = pd.concat([
                pd.DataFrame({k: adata.uns[key][k][group]
                              for k in ['scores', 'names', 'logfoldchanges', 'pvals', 'pvals_adj']}).assign(group=group)
                for group in groups])
            d = d.reset_index(drop=True)
            return d

        def get_marker_genes(adata, layer, groupby, method, key):
            final_df = None
            for batch in adata.obs[self.condition_key].unique():
                for ref_group in adata.obs[self.cell_type_key].unique():
                    for test_group in adata.obs[self.cell_type_key].unique():
                        if test_group == ref_group:
                            continue
                        sub_adata = adata[adata.obs[self.condition_key] == batch]
                        sub_adata = sub_adata[sub_adata.obs[groupby].isin([test_group, ref_group])].copy()
                        if len(sub_adata.obs[groupby].unique()) < 2:
                            continue
                        sc.tl.rank_genes_groups(sub_adata, layer=layer, groupby=groupby, method=method, key_added=key,
                                                ref=ref_group)
                        de_res = rank_genes_groups_df(sub_adata, key=key)
                        de_res['test'] = de_res['group']
                        de_res['ref'] = np.where(de_res['group'] == test_group, ref_group, test_group)
                        de_res['batch'] = batch
                        if final_df is None:
                            final_df = de_res
                        else:
                            final_df = pd.concat([final_df, de_res])
            return final_df.drop_duplicates(subset=['names', 'test', 'ref', 'batch'])

        min_expressed_minus_rest_in_each_batch = 0
        min_expressed_batch_minus_rest = 1

        # adata.uns['log1p']['base'] = None
        method = 't-test'
        res = get_marker_genes(adata, layer='lognorm', groupby=self.cell_type_key, method=method, key=method)
        marker_genes = res[res['pvals_adj'] < 1e-3].copy()
        marker_genes = marker_genes.assign(
            qlogfc=lambda df: df['logfoldchanges'].apply(lambda x: 10 * x if x <= 0 else (x if x >= 0 else 0))).groupby(
            ['batch', 'names', 'test']).sum('qlogfc').reset_index()
        marker_genes = marker_genes.assign(
            de_acc=lambda df: df['qlogfc'].apply(lambda x: -10 if x < 1 else (1 if x > 1 else 0))).groupby(
            ['names', 'test']).sum('de_acc').reset_index()
        marker_genes = marker_genes[marker_genes['de_acc'] >= min_expressed_batch_minus_rest].copy().reset_index(
            drop=True)

        gene_activities = marker_genes.groupby('names').agg(lambda x: " / ".join(sorted(x))).reset_index().rename(
            columns={"test": "activity"})
        value_counts = gene_activities['activity'].value_counts()
        interesting_activities = value_counts[(value_counts > 10) | ~value_counts.index.str.contains("/")]
        gene_activities = gene_activities[gene_activities['activity'].isin(interesting_activities.index)]

        adata.var['gene_de_sig'] = gene_activities.set_index('names')['activity']

    def calculate_cell_embeddings(self, adata):
        adata.X = adata.layers['lognorm'].copy()
        sc.pp.scale(adata, max_value=10)

        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=10)
        sc.tl.umap(adata, spread=1.0, min_dist=0.5, random_state=123)

    def save_outputs(self, adata, adata_hvg):
        self.output_directory.mkdir(parents=True, exist_ok=True)
        adata.write(self.outputs['rna'])
        adata_hvg.write(self.outputs['rna_hvg'])

    def prepare_data(self):
        adata = self.read()
        adata = self.clean(adata)
        adata_hvg, hvg_genes = self.hvg(adata)
        adata.var['hvg'] = adata.var.index.isin(hvg_genes)
        for data in [adata, adata_hvg]:
            self.enrich_layers(data)
            self.calculate_gene_activities(data)
            self.calculate_cell_embeddings(data)
        self.save_outputs(adata, adata_hvg)


if __name__ == '__main__':
    data_cleaner = ImmuneDataCleaner(
        os.path.expanduser('~/Downloads/Immune_ALL_human.h5ad'),
        os.path.expanduser('~/Downloads/immune_prepared'),
    )
    data_cleaner.prepare_data()
