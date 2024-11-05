from pygam import LinearGAM, s
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.sparse

def gam_fit_predict(x, y, weights=None, pred_x=None, n_splines=4, spline_order=2):
    # Weights
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Construct dataframe
    use_inds = np.where(weights > 0)[0]

    # GAM fit
    gam = LinearGAM(s(0, n_splines=n_splines, 
                      spline_order=spline_order)).fit(x[use_inds], y[use_inds], 
                                                      weights=weights[use_inds])

    # Predict
    if pred_x is None:
        pred_x = x
    y_pred = gam.predict(pred_x)

    # Standard deviations
    p = gam.predict(x[use_inds])
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = (
        np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) **
                2 / ((x - np.mean(x)) ** 2).sum())
        * sigma
        / 2
    )

    return y_pred, stds

def plot_cells_pseudotime(
    adata, 
    meta_lineage_cells, 
    obs_key, 
    layer=None, 
    title=None, 
    ax = None,
    color = None,
    ct_color=None,
    scale = True,
    plot_cells=False,
    splines = False,
    **kwargs
):
    for lineage in meta_lineage_cells.index:
        x = adata.obs.loc[meta_lineage_cells[lineage],'dpt_pseudotime'].dropna()
        if obs_key in adata.obs.columns:
            y = adata[x.index].obs[obs_key].values
        else:
            if layer:
                if scipy.sparse.issparse(adata.layers[layer]):
                    # y = np.array(np.log2(adata[x.index, obs_key].layers[layer].A+1))
                    y = np.array(adata[x.index, obs_key].layers[layer].A)
                else:
                    # y = np.array(np.log2(adata[x.index, obs_key].layers[layer]+1))       
                    y = np.array(adata[x.index, obs_key].layers[layer]+1)    
            else:
                if scipy.sparse.issparse(adata.X):
                    # y = np.array(np.log2(adata[x.index, obs_key].X.A+1))
                    y = np.array(adata[x.index, obs_key].X.A)
                else:
                    # y = np.array(np.log2(adata[x.index, obs_key].X+1))    
                    y = np.array(adata[x.index, obs_key].X) 

        sns_df = pd.DataFrame({"x": x.squeeze(), "y": y.squeeze() })
        sns_df = sns_df.sort_values('x')
        if not splines:
            sns_df['pred_y'] = sns_df['y'].rolling(window=200, center=True, min_periods=1).mean()
        else:
            pred_x = np.linspace(0, sns_df['x'].max(), 200)
            pred_y, std = gam_fit_predict(sns_df['x'].values, sns_df['y'], pred_x=pred_x, n_splines=10, spline_order=2) 
            sns_df = pd.DataFrame({"x": pred_x, 'pred_y': pred_y})
        if scale:
            max_pred_y = sns_df['pred_y'].abs().median() #sns_df['pred_y'].abs().max()
            sns_df['pred_y'] = sns_df['pred_y']/max_pred_y
            
        sns.lineplot(data=sns_df, x="x", y="pred_y", zorder=1, label= title, color = (ct_color[lineage] if ct_color is not None else color), ax=ax, **kwargs)
    if plot_cells:
        sns.scatterplot(data=adata.obs.loc[meta_lineage_cells[lineage]], x='dpt_pseudotime',
                                y=-0.01, hue='l2_cell_type',
                                legend=None, alpha=0.5, ax = ax, palette=list(adata.uns['l2_cell_type_colors']))

    return ax, sns_df
