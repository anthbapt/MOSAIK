#!/usr/bin/env python

import numpy as np
import pandas as pd
import anndata as ad
import argparse
import os
import sys
import numpy as np
import pandas as pd
import os.path

def main():
    parser = argparse.ArgumentParser(description='Create AnnData object from counts matrix and metadata')
    parser.add_argument("-X", "--counts",
        help="cell x gene counts (raw) matrix in MatrixMarket format")
    parser.add_argument("--obs",
        help="cell metadata file in csv format, with first column as index")
    parser.add_argument("--var",
        help="feature/gene metadata in csv format, with first column as index")
    parser.add_argument("--coords",
        help="spatial coords file in csv format")
    parser.add_argument("--umap",
        help="umap dims file in csv format")
    parser.add_argument("-o", "--outputdir",
        help="Where to write h5ad file",
        default=".")
    parser.add_argument("--filename",
        help="Name for h5ad file",
        default="adata.h5ad")
    parser.add_argument("-n", "--name",
        help="Name of anndata object",
        default="CosMx study")
    parser.add_argument("--colors",
        help="csv files with colors to import",
        nargs='*')
    args = parser.parse_args()

    if not any([args.counts, args.obs]):
        sys.exit("Need counts or obs to create AnnData object")
    X = obs = var = None
    if args.counts is not None:
        X = ad.read_mtx(args.counts, dtype=np.int32).X
    if args.obs is not None:
        obs = pd.read_csv(args.obs, index_col=0)
    if args.var is not None:
        var = pd.read_csv(args.var, index_col=0)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if args.coords is not None:
        adata.obsm['spatial'] = pd.read_csv(args.coords).to_numpy()
    if args.umap is not None:
        adata.obsm['umap'] = pd.read_csv(args.umap).to_numpy()
    adata.uns['name'] = args.name
    adata.strings_to_categoricals()
    if args.colors is not None:
        for i in args.colors:
            file = os.path.basename(i)
            cat = file.rpartition("_colors.csv")[0]
            if cat in adata.obs:
                if adata.obs[cat].dtype.name == 'category':
                    cols = pd.read_csv(i, header=None)
                    colors_dict = dict(zip(cols[0], cols[1]))
                    adata.uns[cat + "_colors"] = [colors_dict[k] for k in adata.obs[cat].cat.categories]
                else:
                    print(f"{cat} is not a categorical column in AnnData object")
            else:
                print(f"{cat} not found in AnnData obs")

    adata.write(os.path.join(args.outputdir, args.filename), compression="gzip")

if __name__ == '__main__':
    sys.exit(main())