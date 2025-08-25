# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:56:27 2025

@author: acoxson
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch

def kfold_float_rescaling(dataset, extra_feat_cols=[]):

    # Stack per‐molecule scalars into (N_mols,1) arrays and then fit
    ezindo = np.vstack([data.e_zindo.numpy() for data in dataset]).reshape(-1, 1)        # (N,1)
    etddft = np.vstack([data.y.numpy()     for data in dataset]).reshape(-1, 1)        # (N,1)
    
    ezindo_scaler = StandardScaler().fit(ezindo)
    etddft_scaler = StandardScaler().fit(etddft)
    
    # Stack all atom‐level floats for coords + extra_features
    float_cols = [41,42,43] + extra_feat_cols
    
    atom_feats = np.vstack([
        data.x[:, float_cols].numpy()    # (n_atoms_i, len(float_cols))
        for data in dataset
    ])  # → (total_train_atoms, len(float_cols))
    
    coords     = atom_feats[:, :3]        # (TotalAtoms, 3)
    coord_mean  = coords.mean(axis=0)                    # (3,)
    coord_scale = np.sqrt(((coords - coord_mean)**2).sum(axis=1).mean())
    
    if extra_feat_cols != []:
        extras     = atom_feats[:, 3:]        # (TotalAtoms, len(extra_feat_cols))
        extras_scaler = StandardScaler().fit(extras)
        extra_feat_cols=float_cols[3:]
    else:
        extras,extras_scaler,extra_feat_cols=None, None,None

    
    scalers={"E_zindo":ezindo_scaler,
             "E_tddft":etddft_scaler,
             "coord_mean": coord_mean,
             "coord_scale": coord_scale,
             "extras_scaler": extras_scaler,
             "coord_cols": float_cols[:3],         # [41,42,43]
             "extras_cols": extra_feat_cols,        # whatever your extra_feat_cols are
             "atom_feats_cols_indices":float_cols}
    return scalers

def scale_feats(data,scalers):
    x_orig = data.x
    x      = x_orig.clone()    # avoid in‑place weirdness
    
    # --- coords ---
    if scalers["coord_cols"] is not None:
        c_idx = scalers["coord_cols"]         # [41,42,43]
        coords = x_orig[:, c_idx].numpy()     # (natoms,3)
        coords = (coords - scalers["coord_mean"]) / scalers["coord_scale"]
        x[:, c_idx] = torch.tensor(coords, dtype=x.dtype)
    
    # --- extras ---
    if scalers["extras_cols"] is not None:
        e_idx = scalers["extras_cols"]
        extras = x_orig[:, e_idx].numpy()     # (natoms, nextras)
        extras = scalers["extras_scaler"].transform(extras)
        x[:, e_idx] = torch.tensor(extras, dtype=x.dtype)

    
    e_zindo_S = scalers["E_zindo"].transform(data.e_zindo.numpy().reshape(-1, 1))
    e_zindo_S = torch.tensor(e_zindo_S, dtype=data.e_zindo.dtype)
    
    y_S = scalers["E_tddft"].transform(data.y.numpy().reshape(-1, 1))
    y_S = torch.tensor(y_S, dtype=data.y.dtype)

    return x, e_zindo_S, y_S


def true_vs_pred_plot_v2(
    y_real, y_pred, err_pred=None, fmt='ro', ms=3, extra_str='F', title='',
    save=False, savepath=None, savename='figure', show=False, xylims=None,
    units='eV', table_bbox=(0.02, 0.82, 0.36, 0.14), table_fontsize=12
    ):
    # Flatten + NaN-safe
    y_real = np.asarray(y_real).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_real) & np.isfinite(y_pred)
    yr = y_real[mask]
    yp = y_pred[mask]

    # Metrics
    if len(yr) > 1:
        r = float(np.corrcoef(yr, yp)[0, 1])
    else:
        r = np.nan
    mae  = float(np.mean(np.abs(yp - yr))) if len(yr) else np.nan
    rmse = float(np.sqrt(np.mean((yp - yr) ** 2))) if len(yr) else np.nan

    fig, ax = plt.subplots()

    # Points/errorbars
    if err_pred is None:
        ax.plot(yr, yp, fmt, ms=ms, zorder=1)
    else:
        err = np.asarray(err_pred).ravel()[mask]
        ax.errorbar(yr, yp, yerr=err, fmt=fmt, ms=ms, zorder=1)

    # Optional custom limits before y=x
    if xylims is not None:
        ax.set_xlim([xylims[0][0], xylims[0][1]])
        ax.set_ylim([xylims[1][0], xylims[1][1]])

    # y = x line spanning both axes ranges
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    ax.plot([lo, hi], [lo, hi], linestyle='-', color='k', lw=1,
            scalex=False, scaley=False, label='y=x', zorder=2)

    # Titles/labels
    ax.set_title(title)
    ax.set_xlabel('True ' + extra_str, fontsize=13)
    ax.set_ylabel('Predicted ' + extra_str, fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.grid(visible=True, which='major', axis='both')
    ax.minorticks_on()

    # Legend moved away from table (table is top-left)
    ax.legend(loc='lower right',fontsize='x-large')

    #  Metrics table
    cell_text = [[f"{r:.3f}", f"{mae:.3f}", f"{rmse:.3f}"]]
    table = ax.table(
        colLabels=[r"$r$", "MAE", "RMSE"],
        cellText=cell_text,
        cellLoc="center", colLoc="center",
        bbox=table_bbox  # (x0, y0, w, h) in axes coords
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    table.set_zorder(10)
    for _k, cell in table.get_celld().items():
        cell.set_zorder(11)

    # Style header + body
    ncols = 3
    for j in range(ncols):
        # header cell
        hdr = table[(0, j)]
        hdr.set_facecolor((1, 1, 1, 0.96))
        hdr.set_edgecolor("0.3")
        hdr.set_linewidth(0.8)
        # make header bold + black
        hdr.get_text().set_weight('bold')
        hdr.get_text().set_color('k')

        if (1, j) in table.get_celld():
            body = table[(1, j)]
            body.set_facecolor((1, 1, 1, 0.92))
            body.set_edgecolor("0.5")
            body.set_linewidth(0.6)
            body.get_text().set_weight('normal')  # ensure not bold
            body.get_text().set_color('k')

    # Save/show
    if save:
        if savepath and not os.path.exists(savepath):
            os.makedirs(savepath)
        out = (savepath + '/' if savepath else '') + savename + '.png'
        plt.savefig(out, bbox_inches='tight', dpi=600)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return None

def calc_rhoc(x,y):
    ''' 
    Concordance Correlation Coefficient
    https://nirpyresearch.com/concordance-correlation-coefficient/
    '''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

