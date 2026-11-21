import csv
import datetime
import math
import os
from pathlib import Path
import signal
import sys
import traceback

from affine import Affine
import geopandas as gpd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features
from rasterio.transform import rowcol, array_bounds
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from shapely.geometry import shape
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, skeletonize

from .aggregate import mosaic_cells
from ..handler import logger, timed
from .check_sample import get_cells_with_pts
from .project import ProjectPaths
from .utils import get_cell_list_from_grid_param

_FORCE_SERIAL = False

POLY_MODS={'firebreaks':
           {'firebreak':{'cat':93,'shape':2},
            'burnt other':{'cat':95,'shape':1}, 
            'not burnt': {'cat':255,'shape':3}, 
            'no_data':{'cat':0,'shape':0}}
            }

SHAPES = {'firebreaks':{0: "nodata", 1: "blob", 2: "linear", 3: "small fragment"}}

# thresholds live here so classify_day and the report always use the same ones
STAGE1 = {"min_len_m": 50, "a_ext": 10.0, "a_fill": 0.85, "a_pb": 0.80,
          "b_ext": 5.5, "b_w": 200, "b_medw": 170, "b_fill": 0.92, "b_pb": 0.90,
          "b_largest": 0.70, "b_nb": 0.05}

RULES = {
    "A": [("feret_m", ">=", "min_len_m"), ("ext_aspect", ">=", "a_ext"),
          ("fill", ">=", "a_fill"), ("path_burn", ">=", "a_pb")],
    "B": [("feret_m", ">=", "min_len_m"), ("ext_aspect", ">=", "b_ext"), ("W_m", "<=", "b_w"),
          ("med_w", "<=", "b_medw"), ("fill", ">=", "b_fill"), ("path_burn", ">=", "b_pb"),
          ("largest_frac", ">=", "b_largest"), ("nb_frac", "<=", "b_nb")],
}

CLASS_COLORS = {"linear": "#d62728", "blob": "#1f77b4", "small fragment": "#2ca02c"}
CLASS_ORDER = ["linear", "blob", "small fragment"]
KEY_FEATURES = ["ext_aspect", "fill", "path_burn", "W_m", "med_w",
                "largest_frac", "nb_frac", "path_m", "area_ha"]
LOG_FEATURES = {"ext_aspect", "W_m", "med_w", "max_w", "path_m", "area_ha", "aspect"}
PAIRS = [("ext_aspect", "fill"), ("ext_aspect", "W_m"),
         ("fill", "path_burn"), ("largest_frac", "nb_frac")]


### optimization helpers ----------------------------------------------------------------------
######  runtime optimization -------------------------------------------------------------------
class _Lazy(dict):
    """dict whose missing keys are computed on first access, so unused features cost nothing."""
    def __init__(self, base, lazy):
        super().__init__(base)
        self._lazy = lazy

    def __missing__(self, key):
        self[key] = self._lazy[key]()
        return self[key]

_CELL_CTX = {}
_G = {}        # arrays shared with forked workers (copy-on-write; nothing is pickled)

def _note(path, text):
    """Append to a plain file: survives lost stdout/stderr and ray's log capture."""
    with open(path, 'a') as f:
        f.write(text + '\n')
        
def _init_worker():
    global _FORCE_SERIAL
    _FORCE_SERIAL = True
    
def _worker(ids):
    mask, lab, objs, cfg = _G['mask'], _G['lab'], _G['objs'], _G['cfg']
    parts = [_classify_component_firebreak(mask, lab, cid, objs[cid - 1], cfg) for cid in ids]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))

def _n_jobs(params):
    if _FORCE_SERIAL:
        return 1
    v = 0
    if params['polygonize']['n_workers']:
        v = int(params['polygonize']['n_workers'])
    if v <= 0:
        v = int(os.environ.get('SLURM_CPUS_PER_TASK') or len(os.sched_getaffinity(0)))
    return max(v, 1)
    
def _make_tasks(ids, n_jobs, batch=256):
    """Largest components as single tasks first, the many small ones in batches."""
    ids = [int(i) for i in ids]
    head = 4 * n_jobs
    return [[i] for i in ids[:head]] + [ids[k:k + batch] for k in range(head, len(ids), batch)]


#######  parameter optimization -------------------------------------------------------------------- 
def _stage1_overrides(v):
    """None / '' / dict / 'b_ext=5.0,b_fill=0.9' -> dict of floats."""
    if not v or str(v) in ('None', ''):
        return {}
    if isinstance(v, str):
        v = dict(kv.split('=') for kv in v.split(','))
    return {k.strip(): float(x) for k, x in v.items()}

def _ok(v, op, thr):
    return v >= thr if op == ">=" else v <= thr

def _opt(pp, key, default, cast=float):
    v = pp[key] if key in pp else None
    return default if v is None or str(v) in ('', 'None') else cast(v)
    
def stage1_pass(vals, s=STAGE1):
    """vals: dict or Series with the feature names. Returns the name of the first rule that passes, or None."""
    for name, conds in RULES.items():
        if all(_ok(vals[f], op, s[t]) for f, op, t in conds):
            return name
    return None

def _firebreak_cfg(params):
    pp, px = params['polygonize'], float(params['res'])
    cfg = dict(px=px, r_close=2, hole_area_px=2000, r_open_min=3, open_factor=1.6,
               max_width_m=300, max_thick_factor=1.25, blob_ratio=3.0, blob_grow=2,
               len_ratio=1.0, arm_aspect=3, min_len_m=STAGE1['min_len_m'], min_sat_aspect=3.0,
               nb_m=float(pp['nb_m']), nb_far_m=float(pp['nb_far_m']))
    cfg['min_len_px'] = cfg['min_len_m'] / px
    cfg['r_open_max'] = int(cfg['max_thick_factor'] * (cfg['max_width_m'] / px) / 2) + 1
    cfg['nb_px'] = round(cfg['nb_m'] / px)
    cfg['pad'] = max(cfg['r_open_max'], cfg['nb_px']) + cfg['r_close'] + 2
    return cfg
    
def log_tiny_linear(result, result_ext, max_px=5):
    st = np.ones((3, 3), int)
    for name, arr in (("before extension", result), ("after extension ", result_ext)):
        lab, n = ndi.label(arr == 2, structure=st)
        sizes = np.bincount(lab.ravel())[1:]
        tiny = int((sizes <= max_px).sum())
        logger.info(f"linear pieces {name}: {n}, of which <= {max_px}px: {tiny} ({tiny / max(n, 1):.0%})")
        
def flip_note(vals, s=STAGE1):
    """Plain-language summary of how close each rule is to flipping for this component."""
    out = []
    for name, conds in RULES.items():
        res = [(_ok(vals[f], op, s[t]), f, op, s[t], vals[f]) for f, op, t in conds]
        bad = [r for r in res if not r[0]]
        if not bad:
            _, f, op, thr, v = min(res, key=lambda r: abs(r[4] - r[3]) / max(abs(r[3]), 1e-9))
            out.append(f"rule {name} passes (tightest: {f} {v:.2f} vs {op} {thr})")
        elif len(bad) == 1:
            _, f, op, thr, v = bad[0]
            out.append(f"rule {name}: only {f} fails ({v:.2f} vs {op} {thr})")
        else:
            out.append(f"rule {name}: {len(bad)} conditions fail")
    return " | ".join(out)

def calib_table(coord_checker, sample_log, cls):
    rows = []
    for p in coord_checker:
        lg = sample_log.get(p["name"], {})
        f = lg.get("feat")
        if f:
            rows.append(dict(name=p["name"], expect=cls.get(p.get("expect")),
                             got=cls.get(lg.get("got")), **f))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("name")

def _auc(pos, neg):
    """P(a random 'pos' value exceeds a random 'neg' value); 0.5 = no separation."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def calibration_report(df, out_dir, s=STAGE1, log=print, mod_name=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not mod_name:
        mod_name = 'poly_calibration_table'
    df.to_csv(out_dir / f'{mod_name}.csv')          # re-tune offline from this

    feat_cols = [c for c in df.columns if c not in ("expect", "got")]
    d = df.reset_index()
    d["pred_linear"] = [stage1_pass(r, s) is not None for _, r in d.iterrows()]

    # components that hold several classes (e.g. break fused to a blob) can't be judged by Stage 1
    d["comp"] = d[feat_cols].round(4).astype(str).agg("|".join, axis=1)
    mixed = d.groupby("comp")["expect"].transform("nunique") > 1
    amb = d[mixed].drop_duplicates("comp")
    d1 = d[~mixed].drop_duplicates("comp")                 # one row per consistent component
    log(f"{len(d)} debug points -> {len(d1)} unique components with a single expected class, "
        f"{len(amb)} mixed components excluded")
    if len(amb):
        log("mixed components (decided by Stage 2, not Stage 1): " +
            ", ".join(d[mixed].groupby("comp")["name"].apply(lambda x: "/".join(x)).tolist()))

    # --- Stage 1 result: linear vs not
    d1 = d1.assign(truth_linear=(d1.expect == "linear"))
    log("\nStage 1 (linear vs not), rows = expected:\n" +
        pd.crosstab(d1.expect, d1.pred_linear).to_string())
    wrong = d1[d1.pred_linear != d1.truth_linear]
    for _, r in wrong.iterrows():
        kind = "MISSED linear" if r.truth_linear else f"FALSE linear (expected {r.expect})"
        log(f"  {kind}: {r['name']}  {flip_note(r, s)}")

    # --- final result at the debug points (includes Stage 2 and the extension step)
    log("\nFinal class at each point, rows = expected, columns = got:\n" +
        pd.crosstab(d.expect, d.got).to_string())

    # --- separability ranking
    rows = []
    for f in feat_cols:
        lin = d1.loc[d1.truth_linear, f].to_numpy(float)
        res = {"feature": f}
        for lbl, mask in (("vs_blob", d1.expect == "blob"),
                          ("vs_frag", d1.expect == "small fragment"),
                          ("vs_all", ~d1.truth_linear)):
            res[lbl] = _auc(lin, d1.loc[mask, f].to_numpy(float))
        rows.append(res)
    sep = pd.DataFrame(rows).set_index("feature")
    sep["linear_is"] = np.where(sep.vs_all >= 0.5, "higher", "lower")
    sep["best"] = sep.vs_all.apply(lambda a: max(a, 1 - a))
    log("\nSeparability (AUC; 0.5 = none, 1.0 = perfect), best first:\n" +
        sep.sort_values("best", ascending=False).round(2).to_string())

    # --- optional depth-3 tree
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        X = d1[feat_cols]
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(X, d1.truth_linear)
        log("\nDepth-3 tree, linear vs not:\n" + export_text(tree, feature_names=list(X.columns)))
    except ImportError:
        log("\n(sklearn not installed; skipping decision tree)")

    # --- figures
    thr_lines = {}
    for rn, conds in RULES.items():
        for f, op, t in conds:
            thr_lines.setdefault(f, []).append((s[t], rn))

    # 1. strip plots, one panel per feature
    feats = [f for f in KEY_FEATURES if f in d1.columns]
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.7 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes.ravel(), feats):
        for yi, cl in enumerate(CLASS_ORDER):
            sub = d1[d1.expect == cl]
            y = yi + rng.uniform(-0.28, 0.28, len(sub))
            w = (sub.pred_linear != (cl == "linear")).to_numpy()
            ax.scatter(sub[f], y, s=28, c=CLASS_COLORS[cl],
                       edgecolors=["k" if x else "none" for x in w], linewidths=1.4)
            for xv, yv, nm in zip(sub[f][w], y[w], sub["name"][w]):
                ax.annotate(nm.split("_")[0], (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for thr, rn in thr_lines.get(f, []):
            ax.axvline(thr, ls="--", lw=1, c="gray")
            ax.text(thr, 2.5, rn, fontsize=7, color="gray")
        if f in LOG_FEATURES:
            ax.set_xscale("log")
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_yticklabels([f"{c} ({(d1.expect == c).sum()})" for c in CLASS_ORDER], fontsize=8)
        ax.set_ylim(-0.6, 2.7)
        ax.set_title(f, fontsize=10)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("Dashed = rule thresholds (A/B); black ring = Stage 1 wrong", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "strip_by_feature.png", dpi=130)
    plt.close(fig)

    # 2. scatter of feature pairs
    pairs = [p for p in PAIRS if p[0] in d1.columns and p[1] in d1.columns]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.7), squeeze=False)
    for ax, (fx, fy) in zip(axes[0], pairs):
        for cl in CLASS_ORDER:
            sub = d1[d1.expect == cl]
            w = (sub.pred_linear != (cl == "linear")).to_numpy()
            ax.scatter(sub[fx], sub[fy], s=34, c=CLASS_COLORS[cl], label=f"{cl} ({len(sub)})",
                       edgecolors=["k" if x else "none" for x in w], linewidths=1.4)
            for xv, yv, nm in zip(sub[fx][w], sub[fy][w], sub["name"][w]):
                ax.annotate(nm.split("_")[0], (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if len(amb):
            ax.scatter(amb[fx], amb[fy], marker="D", s=30, c="lightgray", edgecolors="gray",
                       label=f"mixed component ({len(amb)})")
        for thr, rn in thr_lines.get(fx, []):
            ax.axvline(thr, ls="--", lw=1, c="gray")
        for thr, rn in thr_lines.get(fy, []):
            ax.axhline(thr, ls="--", lw=1, c="gray")
        if fx in LOG_FEATURES:
            ax.set_xscale("log")
        if fy in LOG_FEATURES:
            ax.set_yscale("log")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_pairs.png", dpi=130)
    plt.close(fig)
    log(f"\nfigures and poly_calibration_table.csv written to {out_dir}")
### --------------------------------------------------------------------------------------------
def _area_filled(rp):
    return rp.area_filled if hasattr(rp, "area_filled") else rp.filled_area

def fill_small_holes(comp, max_px):
    """Fill enclosed background smaller than max_px; background touching the crop border is never filled."""
    lab_bg = label(~comp, connectivity=1)
    sizes = np.bincount(lab_bg.ravel())
    border = np.unique(np.concatenate([lab_bg[0], lab_bg[-1], lab_bg[:, 0], lab_bg[:, -1]]))
    fill = sizes <= max_px
    fill[0] = False
    fill[border] = False
    return comp | fill[lab_bg]

def feret_max(m):
    """Longest straight-line distance across a binary shape (px)."""
    pts = np.argwhere(m)
    if len(pts) < 2:
        return 1.0
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts[:: max(1, len(pts) // 2000)]
    return float(pdist(hull).max()) + 1.0

def longest_path(sk):
    """Longest geodesic path through a skeleton. Returns (length_px, ordered path [(row, col)])."""
    pts = np.argwhere(sk)
    n = len(pts)
    if n < 2:
        return float(n), pts
    idx = np.full(sk.shape, -1)
    idx[pts[:, 0], pts[:, 1]] = np.arange(n)
    ri, ci, wt = [], [], []
    for dr, dc, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
        r2, c2 = pts[:, 0] + dr, pts[:, 1] + dc
        ok = (r2 >= 0) & (r2 < sk.shape[0]) & (c2 >= 0) & (c2 < sk.shape[1])
        j = np.full(n, -1)
        j[ok] = idx[r2[ok], c2[ok]]
        m = j >= 0
        ri.append(np.where(m)[0]); ci.append(j[m]); wt.append(np.full(m.sum(), w))
    g = coo_matrix((np.concatenate(wt), (np.concatenate(ri), np.concatenate(ci))),
                   shape=(n, n)).tocsr()
    ncomp, labs = connected_components(g, directed=False)
    start = np.where(labs == np.bincount(labs).argmax())[0][0]
    d0 = dijkstra(g, directed=False, indices=start)
    d0[np.isinf(d0)] = -1
    a = d0.argmax()
    d1, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
    d1[np.isinf(d1)] = -1
    b = d1.argmax()
    path = [b]
    while pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    return float(d1[b]), pts[path]

def skel_len(m):
    return longest_path(skeletonize(m))[0]

def _log(dbg, sample_log, **kw):
    for p in dbg:
        sample_log.setdefault(p["name"], {}).update(kw)

def _feat(dbg, sample_log, **kw):
    for p in dbg:
        sample_log.setdefault(p["name"], {}).setdefault("feat", {}).update(kw)

def setup_calibration_pts(pts, transform, doy, win, yr, mod_type, quiet=False):
    """
    input is path to .csv file with cols 'XCoord' and 'YCoord' and col 'Obs<year>' with observations for the year modeled. 
    Observations in 'Obs<year>' need to match classes in POLY_MODS dict for the <mod_type>. 
    <mod_type> needs to match exactly with the dictionary key in POLY_MODS dict
    Also needs name column in input table. Looks first for 'Name' or 'name'. TODO: fix hardcoded solution if these dont exist
    """
    
    if pts == None:
        cal_pts = []
    elif isinstance(pts, list):
        ## this would need to be a list of dictionaries with 'name','expect'and'xy' for each point
        cal_pts = pts
    else:
        cal_df = pd.read_csv(pts)
        
        obscol = f'Obs{yr}'
        if obscol not in cal_df.columns:
            logger.warning(f"MISSING column {obscol} in calibration point file -- dropping calibration for this year")
            return []
        
        name_col = next((col for col in cal_df.columns if col in ['Name', 'name']), 'name')
        if name_col in cal_df.columns and name_col !='name':
            cal_df['name'] = cal_df[name_col]
        elif name_col not in cal_df.columns:
            cal_df['name'] = cal_df['obs'].astype(str) + '_' + cal_df['Descrip']
        cat2shape = {v['cat']: v['shape'] for v in POLY_MODS[mod_type].values()}
        cal_df['expect'] = pd.to_numeric(cal_df[obscol], errors='coerce').map(cat2shape)
        unmapped = cal_df['expect'].isna()
        if unmapped.any():
            logger.debug(f"dropping {int(unmapped.sum())} calibration points with a missing/unknown {obscol} category")
        cal_df = cal_df[~unmapped & (cal_df['expect'] > 0)].copy()     # shape 0 (no_data, not burnt) dropped
        cal_df['expect'] = cal_df['expect'].astype(int)
        cal_df['xy'] = list(zip(cal_df['XCoord'], cal_df['YCoord']))
        cal_pts=cal_df[['name','expect','xy']]
        cal_pts = cal_pts.to_dict("records")

    out = []
    for p in cal_pts:
        r, c = rowcol(transform, *p["xy"])
        r, c = int(r), int(c)
        if 0 <= r < doy.shape[0] and 0 <= c < doy.shape[1]:
            out.append(dict(p, rc=(r, c), doy=int(doy[r, c]), win=int(win[r, c])))
        elif not quiet:
            logger.info(f"sample point {p['name']} {p['xy']} is outside the raster")
            
    return out

def _extra_features(orig, path, sk, L, feret, other, d_comp, not_end, params):
    """Diagnostic-only features (for the calibration table); not used by the classifier."""

    nb_m = params['polygonize']['nb_m']
    nb_far_m = params['polygonize']['nb_far_m']
    nb_min_px = params['polygonize']['nb_minpx']
    px = params['res']
    nb_px = nb_m / px
    nb_far_px = nb_far_m / px

    on = orig[path[:, 0], path[:, 1]] if len(path) else np.zeros(0, bool)
    if on.any():
        dd = np.diff(np.concatenate([[0], on.astype(int), [0]]))
        longest_run = (np.where(dd == -1)[0] - np.where(dd == 1)[0]).max()
    else:
        longest_run = 0
    plab = label(orig, connectivity=2)
    psz = np.bincount(plab.ravel())[1:]
    span_frac = feret_max(plab == (psz.argmax() + 1)) / max(feret, 1e-9) if len(psz) else 0.0
    near = (d_comp <= nb_px) & (d_comp > 0)
    far = (d_comp <= nb_far_px) & (d_comp > 0)
    nb_lab = label(other, connectivity=2)
    ids, cnt = np.unique(nb_lab[near & not_end & other], return_counts=True)
    return {"run_frac": longest_run / max(len(path), 1),
                "path_frac": len(path) / max(int(sk.sum()), 1),
                "tort": L / max(feret, 1e-9),
                "pieces": len(psz), "span_frac": span_frac,
                "n_nb": int(((ids > 0) & (cnt >= nb_min_px)).sum()),
                "nb_far": (other & far & not_end).sum() / max((far & not_end).sum(), 1),
                "nb_end": (other & near & ~not_end).sum() / max((near & ~not_end).sum(), 1)}

def separation(df, good="linear"):
    g, b = df[df.expect == good], df[df.expect != good]
    out = []
    for c in df.columns.drop("expect"):
        gs, bs = g[c].dropna(), b[c].dropna()
        if gs.empty or bs.empty:
            continue
        if gs.min() > bs.max():   v = f"good > bad: threshold ~{(gs.min()+bs.max())/2:.2f}"
        elif gs.max() < bs.min(): v = f"good < bad: threshold ~{(gs.max()+bs.min())/2:.2f}"
        else:                     v = "overlap"
        out.append((c, f"{gs.min():.2f} to {gs.max():.2f}", f"{bs.min():.2f} to {bs.max():.2f}", v))
    return pd.DataFrame(out, columns=["feature", "good range", "bad range", "verdict"])

def _endpoints(sk):
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    return np.argwhere(sk & (nb == 1))              # (row, col)

def _unit(v):
    n = np.hypot(*v)
    return v / n if n > 0 else None

def extend_linear(params, result, win):
    """
    Grow linear features to capture nearby fragments in same line (with some turn flexibility)
    """
    px = float(params['res'])
    ext_m = float(params['polygonize']['grow_ext_m']) # max gap from the current tip to the next fragment
    tol_m = params['polygonize']['grow_tol_m']  # half-width of the search strip
    dir_m = params['polygonize']['grow_dir_m']  # length of skeleton used to estimate tip direction
    min_inside = float(params['polygonize']['grow_min_overlap']) # fraction of a fragment's pixels that must lie in the strip
    max_hops = params['polygonize']['grow_max_hops'] # max fragments chained per endpoint
    alpha = params['polygonize']['grow_alpha'] # direction smoothing (1 = never turn, 0 = point at last fragment)
    cand_classes = params['polygonize']['conquer_class'] # classes eligible to be captured; use (3,) if blobs get captured
    max_frag_area  = params['polygonize']['conquer_areamax']    # px; larger components are never captured (protects real blobs)
    width_factor = params['polygonize']['grow_width_factor']          # crumb may be this x the line's width (+1 px)
    cos_align = np.cos(np.radians(float(params['polygonize']['grow_align_deg'])))

    logger.info('extending linear paths to incorporate fragments...') 
    ext_px = ext_m / px
    tol_px = tol_m / px
    dir_px = dir_m / px

    out = result.copy()

    # candidate fragments, labelled separately per window
    cand_mask = np.isin(result, cand_classes)
    cand_lab = label(np.where(cand_mask, win, 0), background=0, connectivity=2)
    frags = []
    for rp in regionprops(cand_lab):
        if rp.area > max_frag_area:
            continue
        rr, cc = rp.coords.T
        pts = rp.coords.astype(float)
        ev, evec = np.linalg.eigh(np.cov((pts - pts.mean(0)).T, bias=True))
        frags.append({"rows": rr, "cols": cc, "win": win[rr[0], cc[0]],
                      "cen": rp.centroid, "taken": False,
                      "elong": np.sqrt((ev[1] + 1 / 12) / (ev[0] + 1 / 12)),    # ~ length / width
                      "axis": evec[:, 1],                                        # (row, col) along the crumb
                      "thick": 2 * ndi.distance_transform_edt(np.pad(rp.image, 1)).max()})
    if not frags:
        return out

    #logger.info(f'chacking {len(frags)} preliminary candidate fragments')
    cens = np.array([f["cen"] for f in frags])          # (row, col)

    # anchors: linear components (per window)
    lin_lab = label(np.where(result == 2, win, 0), background=0, connectivity=2)
    for rp in regionprops(lin_lab):
        r0, c0, r1, c1 = rp.bbox
        sk = skeletonize(lin_lab[r0:r1, c0:c1] == rp.label)
        sk_pts = np.argwhere(sk) + [r0, c0]
        comp_a = lin_lab[r0:r1, c0:c1] == rp.label
        w_anchor = 2 * np.median(ndi.distance_transform_edt(np.pad(comp_a, 1))[1:-1, 1:-1][sk])
        w_id = win[sk_pts[0][0], sk_pts[0][1]]
        for tip in _endpoints(sk) + [r0, c0]:
            near = sk_pts[np.hypot(*(sk_pts - tip).T) <= dir_px]
            if len(near) < 4:
                continue
            d = _unit(tip - near.mean(0))               # points away from the body
            if d is None:
                continue
            tip = tip.astype(float)

            for _ in range(max_hops):
                # strip: 0 <= along <= ext_px, |across| <= tol_px   (all in (row, col) space)
                end = tip + d * ext_px
                lo = np.minimum(tip, end) - tol_px
                hi = np.maximum(tip, end) + tol_px
                idx = np.where((cens[:, 0] >= lo[0]) & (cens[:, 0] <= hi[0]) &
                               (cens[:, 1] >= lo[1]) & (cens[:, 1] <= hi[1]))[0]

                hits, far_pt, far_along = [], None, 0
                for i in idx:
                    f = frags[i]
                    if f["taken"] or f["win"] != w_id:
                        continue
                    if f["thick"] > width_factor * w_anchor + 1:               # much fatter than this line
                        continue
                    if f["elong"] > 2.0 and abs(f["axis"] @ d) < cos_align:    # elongated but across the line
                        continue
                    pts = np.column_stack([f["rows"], f["cols"]]).astype(float)
                    rel = pts - tip
                    along = rel @ d
                    if along.max() <= 0:                # entirely behind/at the tip
                        continue
                    across = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
                    inside = (along >= 0) & (along <= ext_px) & (across <= tol_px)
                    if inside.mean() < min_inside:
                        continue
                    hits.append(i)
                    j = along.argmax()
                    if along[j] > far_along:
                        far_along, far_pt = along[j], pts[j]

                if not hits or far_pt is None:
                    break
                for i in hits:
                    frags[i]["taken"] = True
                    out[frags[i]["rows"], frags[i]["cols"]] = 2

                step = far_pt - tip
                if np.hypot(*step) >= 3:                # ignore noisy tiny steps
                    d = _unit(alpha * d + (1 - alpha) * _unit(step))
                tip = far_pt
    return out

def _classify_component_firebreak(mask, lab, cid, slc, cfg, dbg=(), sample_log=None, wid=None, params=None):
    """Classify one closed component. Returns (rows, cols, classes) in full-raster coordinates.
    dbg / sample_log / params are only used for calibration points (main process only)."""
    px, pad = cfg['px'], cfg['pad']
    min_len_px, r_open_min = cfg['min_len_px'], cfg['r_open_min']
    sl = (slice(max(slc[0].start - pad, 0), min(slc[0].stop + pad, mask.shape[0])),
          slice(max(slc[1].start - pad, 0), min(slc[1].stop + pad, mask.shape[1])))
    r_off, c_off = sl[0].start, sl[1].start
    comp = lab[sl] == cid                      # closed shape of this component
    orig = comp & mask[sl]                     # actually burned pixels
    o = np.zeros(comp.shape, np.uint8)
    loc = lambda p: (p["rc"][0] - r_off, p["rc"][1] - c_off)

    def emit():
        rr, cc = np.nonzero(o)
        return rr.astype(np.int32) + r_off, cc.astype(np.int32) + c_off, o[rr, cc]

    # ---- cheap measurements
    solid = fill_small_holes(comp, cfg['hole_area_px'])
    A = int(solid.sum())
    edt = ndi.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1]
    max_hw = edt.max()
    feret = feret_max(solid)
    ext_aspect = feret ** 2 / max(A, 1)
    fill = orig.sum() / max(A, 1)

    # ---- path-based measurements, skipped when nothing could use them
    if bool(dbg) or feret >= min_len_px or max_hw >= r_open_min + 1:
        sk = skeletonize(solid)
        L, path = longest_path(sk)
        med_hw = np.percentile(edt[path[:, 0], path[:, 1]], 40) if len(path) else 0.0
        on = orig[path[:, 0], path[:, 1]] if len(path) else np.zeros(0, bool)
        path_burn = on.mean() if len(on) else 0.0
        plab = label(orig, connectivity=2)
        psz = np.bincount(plab.ravel())[1:]
        largest_frac = psz.max() / psz.sum() if len(psz) else 0.0
    else:
        sk, plab, L, path, med_hw, path_burn, largest_frac = None, None, 0.0, np.zeros((0, 2), int), 0.0, 0.0, 0.0
    W = A / max(L, 1)
    W_m, med_w_m = W * px, 2 * med_hw * px
    fused = (max_hw > cfg['blob_ratio'] * med_hw) and (max_hw >= r_open_min + 1)

    # ---- neighbour ring (two distance transforms): computed only when needed
    nb_cache = {}

    def nb():
        if not nb_cache:
            other = mask[sl] & ~comp
            d_comp = ndi.distance_transform_edt(~comp)
            if len(path) >= 2:
                e = np.ones(comp.shape, bool)
                e[tuple(path[0])] = False
                e[tuple(path[-1])] = False
                not_end = ndi.distance_transform_edt(e) > cfg['nb_px']
            else:
                not_end = np.ones(comp.shape, bool)
            ring = (d_comp <= cfg['nb_px']) & (d_comp > 0) & not_end
            nb_cache.update(other=other, d_comp=d_comp, not_end=not_end,
                            nb_frac=(other & ring).sum() / max(ring.sum(), 1))
        return nb_cache

    vals = _Lazy({"feret_m": feret * px, "ext_aspect": ext_aspect, "fill": fill,
                  "path_burn": path_burn, "W_m": W_m, "med_w": med_w_m,
                  "largest_frac": largest_frac},
                 {"nb_frac": lambda: nb()["nb_frac"]})

    if dbg:
        n = nb()
        feats = {"feret_m": feret * px, "path_m": L * px, "area_ha": A * px * px / 1e4, "W_m": W_m,
                 "aspect": L / max(W, 1e-9), "max_w": 2 * max_hw * px, "med_w": med_w_m,
                 "path_burn": path_burn, "fill": fill, "ext_aspect": ext_aspect,
                 "largest_frac": largest_frac, "nb_frac": n["nb_frac"]}
        feats.update(_extra_features(orig, path, sk, L, feret, n["other"], n["d_comp"], n["not_end"], params))
        _feat(dbg, sample_log, **feats)
        _log(dbg, sample_log, stats=(f"component: path={L*px:.0f}m  area={A*px*px/1e4:.1f}ha  W={W_m:.0f}m  "
                                     f"max_w={2*max_hw*px:.0f}m  med_w={med_w_m:.0f}m  ext_aspect={ext_aspect:.1f}  "
                                     f"fill={fill:.2f}  path_burn={path_burn:.2f}  largest_frac={largest_frac:.2f}  "
                                     f"nb_frac={n['nb_frac']:.2f}  fused={fused}"))

    # ---------- Stage 1: whole component is a firebreak ----------
    rule = stage1_pass(vals)
    if rule:
        o[orig] = 2
        # satellites: burned pieces the main path doesn't run through (crumbs glued on by the closing)
        on_path = set(np.unique(plab[path[:, 0], path[:, 1]]).tolist()) - {0}
        n_demoted = 0
        for pc in regionprops(plab):
            if pc.label in on_path:
                continue
            f_pc = feret_max(pc.image)
            if f_pc < min_len_px or f_pc ** 2 / pc.area < cfg['min_sat_aspect']:
                o[plab == pc.label] = 3
                n_demoted += 1
        if dbg:
            _log(dbg, sample_log, decision=f"stage 1: passed rule {rule} -> linear "
                                           f"({n_demoted} satellite crumbs demoted)")
        return emit()
    if dbg:
        _log(dbg, sample_log, decision="stage 1 failed: " + flip_note(vals))

    # ---------- Stage 2: blob core + arms ----------
    r_open = (int(np.clip(np.ceil(cfg['open_factor'] * med_hw) + 1, r_open_min, cfg['r_open_max']))
              if fused else r_open_min)
    core = binary_opening(solid, disk(r_open))
    blob = (ndi.binary_dilation(core, disk(cfg['blob_grow'])) if cfg['blob_grow'] else core) & orig

    if not blob.any():                         # nothing thick survives: fragment
        o[orig] = 3
        if dbg:
            _log(dbg, sample_log, stage2="no blob core -> small fragment")
        return emit()

    o[blob] = 1
    for p in dbg:
        if blob[loc(p)]:
            _log([p], sample_log, stage2=f"pixel is in the blob core (opening radius {r_open}px = {(2*r_open+1)*px:.0f}m wide)")

    blab = label(blob, connectivity=2)
    bdiam = {b.label: 2 * np.sqrt(_area_filled(b) / np.pi) for b in regionprops(blab)}
    rlab = label(orig & ~blob, connectivity=2)
    for ar in regionprops(rlab):
        a0, b0, a1, b1 = ar.bbox
        sub = (slice(max(a0 - 1, 0), a1 + 1), slice(max(b0 - 1, 0), b1 + 1))
        pm = rlab[sub] == ar.label
        L_arm = skel_len(ar.image)
        W_arm = ar.area / max(L_arm, 1)
        elongated = L_arm / max(W_arm, 1e-9) >= cfg['arm_aspect']

        touch = ndi.binary_dilation(pm, structure=np.ones((3, 3), bool))
        ids = np.unique(blab[sub][touch & blob[sub]])
        ids = ids[ids > 0]
        if len(ids):                           # arm attached to a blob
            need = max(min_len_px, cfg['len_ratio'] * max(bdiam[i] for i in ids))
            o[sub][pm] = 2 if (L_arm >= need and elongated) else 1
            info = (f"pixel is on an arm attached to a blob: arm length {L_arm*px:.0f}m "
                    f"(needs >= {need*px:.0f}m), arm aspect {L_arm/max(W_arm,1e-9):.1f} (needs >= {cfg['arm_aspect']})")
        else:                                  # free-standing piece
            o[sub][pm] = 2 if (L_arm >= min_len_px and elongated) else 3
            info = (f"pixel is on a free-standing piece: length {L_arm*px:.0f}m "
                    f"(needs >= {cfg['min_len_m']}m and aspect >= {cfg['arm_aspect']})")
        for p in dbg:
            if rlab[loc(p)] == ar.label:
                _log([p], sample_log, stage2=info)
    return emit()
    
def classify_firebreaks(params, mask, wid=None, coord_checker=None, sample_log=None):
    """mask: bool array for one DOY window. Returns uint8: 0 none, 1 blob, 2 linear, 3 small fragment."""
    coord_checker = coord_checker or []
    sample_log = sample_log if sample_log is not None else {}
    cfg, n_jobs = _firebreak_cfg(params), _n_jobs(params)

    closed = binary_closing(mask, disk(cfg['r_close']))
    lab, n = ndi.label(closed, structure=np.ones((3, 3), int))           # int32 labels
    if n == 0:
        return np.zeros(mask.shape, np.uint8)
    objs = ndi.find_objects(lab)
    bb = np.array([(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in objs])
    hh, ww = bb[:, 1] - bb[:, 0], bb[:, 3] - bb[:, 2]

    # components holding calibration points are always done in this process, with logging
    dbg_arr = np.array(sorted({int(lab[p["rc"]]) for p in coord_checker if p["win"] == wid} - {0}), dtype=int)

    # fast path: too small to reach min_len and too small to hold the opening disk -> always class 3
    tiny = ((np.minimum(hh, ww) < 2 * cfg['r_open_min'] + 1) &
            (np.hypot(hh - 1, ww - 1) + 1 < cfg['min_len_px']))
    if len(dbg_arr):
        tiny[dbg_arr - 1] = False
    lut = np.zeros(n + 1, np.uint8)
    lut[1:][tiny] = 3
    out = lut[lab]
    out[~mask] = 0

    todo = np.flatnonzero(~tiny) + 1
    pool_ids = todo[~np.isin(todo, dbg_arr)]
    pool_ids = pool_ids[np.argsort(-(hh[pool_ids - 1] * ww[pool_ids - 1]), kind='stable')]
    logger.debug(f"window {wid}: {n} components | fast path {int(tiny.sum())} | "
                f"full {len(pool_ids)} | calibration {len(dbg_arr)} | workers {n_jobs}")

    for cid in dbg_arr:
        dbg = [p for p in coord_checker if p["win"] == wid and lab[p["rc"]] == cid]
        rr, cc, vals = _classify_component_firebreak(mask, lab, cid, objs[cid - 1], cfg, dbg, sample_log, wid, params)
        out[rr, cc] = vals

    tasks = _make_tasks(pool_ids, n_jobs)
    _G.update(mask=mask, lab=lab, objs=objs, cfg=cfg)
    try:
        if n_jobs > 1 and len(tasks) > 1:
            with mp.get_context('fork').Pool(min(n_jobs, len(tasks))) as pool:
                for rr, cc, vals in pool.imap_unordered(_worker, tasks):
                    out[rr, cc] = vals
        else:
            for t in tasks:
                rr, cc, vals = _worker(t)
                out[rr, cc] = vals
    finally:
        _G.clear()
        
    return out

CLASSIFIERS = {'firebreaks': classify_firebreaks}      # add new object types here

def write_class_doy_raster(out_path, result_ext, doy, profile, shapes_map, grown=None):
    classes = [k for k in shapes_map if k != 0]
    extra = set(np.unique(result_ext).tolist()) - set(shapes_map)
    if extra:
        logger.warning(f"class values not in SHAPES (dropped from class bands): {sorted(extra)}")

    prof = profile.copy()
    prof.update(count=len(classes) + 1 + int(grown is not None), dtype="uint16", nodata=0, compress="lzw")
    with rio.open(out_path, "w", **prof) as dst:
        for i, k in enumerate(classes, start=1):
            dst.write((result_ext == k).astype(np.uint16), i)
            dst.set_band_description(i, shapes_map[k])
            if grown is not None:
                dst.write(grown.astype(np.uint16), len(classes) + 2)
                dst.set_band_description(len(classes) + 2, "ext_added")
        dst.write(doy.astype(np.uint16), len(classes) + 1)
        dst.set_band_description(len(classes) + 1, "doy")
        
def classify_day(params, m, w, coord_checker, sample_log):
    key = params['iter_models']['optimize_on']
    if key not in CLASSIFIERS:
        raise ValueError(f"no classifier registered for optimize_on={key!r}; known: {list(CLASSIFIERS)}")
    return CLASSIFIERS[key](params, m, w, coord_checker, sample_log)

def get_poly_stats(params, in_ras, sample_log, quiet=False):

    window_days = params['polygonize']['day_grouping']
    bands = params['polygonize']['ras_bands']
    calibration_pts = params['polygonize']['calibration_pts']
    mod_type = params['iter_models']['optimize_on']
    mod_year = params['polygonize']['mod_year']

    with timed('read + merge bands'):
        with rio.open(in_ras) as src:
            logger.debug(f'bands in input raster: {src.descriptions}') 
            stack = src.read(bands)              # shape (3, rows, cols)
            profile = src.profile
        
        big = np.iinfo(stack.dtype).max
        stack = np.where(stack > 0, stack, big)          # stays int16
        ## input ras is in doy. take the earliest burn from the stack (after excluding 0)
        doy = stack.min(axis=0)
        del stack
        doy[doy == big] = 0
        doy = doy.astype(np.uint16)
        ## chunk temporally by window_days parameter
        win = np.where(doy > 0, (doy.astype(np.int32) - 1) // window_days + 1, 0).astype(np.int16)
        
    coord_checker = setup_calibration_pts(calibration_pts, profile["transform"], doy, win, mod_year, mod_type, quiet)
            
    result = np.zeros(doy.shape, np.uint8)

    with timed('classification, all windows'):
        for w in np.unique(win[win > 0]):
            logger.debug(f'working on temporal window starting with doy {w}...')
            m = win == w
            res = classify_day(params, m, w, coord_checker, sample_log)
            result[m] = res[m]

        with timed('extension of paths'):
            result_ext = extend_linear(params, result, win)
            grown = (result_ext == 2) & (result != 2)          # pixels the extension step added to the linear class
            for p in coord_checker:
                sample_log.setdefault(p["name"], {})["got"] = int(result_ext[p["rc"]])

    return result_ext, sample_log, coord_checker, profile, doy, grown


def _origin(ef):
    return np.where(ef <= 1e-6, "first_pass", np.where(ef >= 1 - 1e-6, "extension", "mixed"))


def array_to_polygons(result, doy, transform, crs, window_days, shapes_map, grown=None, classes=None,
                      split_by_window=True, min_area_ha=0.0, simplify_m=0.0,
                      keep_satellites=True, min_area_classes=None):
    win = np.where(doy > 0, (doy.astype(np.int32) - 1) // window_days + 1, 0).astype(np.int32)

    parts = []
    for k in (k for k in shapes_map if k != 0):
        if classes is not None and k not in classes:
            continue
        band = result == k
        if not band.any():
            continue

        key = np.where(band, win, 0) if split_by_window else band.astype(np.int32)
        lab = label(key, background=0, connectivity=2).astype(np.int32)
        valid = lab > 0
        if not valid.any():
            continue

        l, d = lab[valid], doy[valid].astype(np.float32)
        idx = np.arange(1, lab.max() + 1)
        d_min, d_max, d_med = ndi.minimum(d, l, idx), ndi.maximum(d, l, idx), ndi.median(d, l, idx)
        if grown is not None and (grown & valid).any():        # share of each region added by the extension
            ext = ndi.mean(grown[valid].astype(np.float32), l, idx)
        else:
            ext = np.zeros(len(idx), np.float32)

        geoms, ids = [], []
        for g, v in features.shapes(lab, mask=valid, transform=transform, connectivity=8):
            geoms.append(shape(g))
            ids.append(int(v))
        ids = np.asarray(ids) - 1
        ef = ext[ids].astype(np.float32)
        parts.append(gpd.GeoDataFrame(
            {"cls": np.uint8(k), "cls_name": shapes_map[k],
             "doy": np.rint(d_med[ids]).astype(np.int16),
             "doy_min": d_min[ids].astype(np.int16),
             "doy_max": d_max[ids].astype(np.int16),
             "ext_frac": ef, "origin": _origin(ef)},
            geometry=geoms, crs=crs))
        del band, key, lab, valid, l, d

    if not parts:
        return None
    gdf = pd.concat(parts, ignore_index=True)
    gdf["area_ha"] = gdf.geometry.area / 1e4                    # needs a projected CRS in metres
    return _finish(gdf, min_area_ha, simplify_m, keep_satellites, min_area_classes)

def classify_cell_with_polygons(params, cell):
    """Classify one cell with its buffer, keep the core, write the core raster + polygons."""
    
    object_type = params['iter_models']['optimize_on']
    mod_year = params['polygonize']['mod_year']
    mod_name = params['polygonize']['mod_name']

    ppaths_cell = ProjectPaths(params, grid=int(cell))
    in_ras = ppaths_cell.comp / f'{int(cell):06d}_{mod_year}_{mod_name}.tif'
    if not in_ras.is_file():
        logger.warning(f'no raster found for {in_ras}. SKIPPING {int(cell):06d}')
        return None

    tmp_poly_dir = Path(params['scratch_dir']) / 'polys' / f'{mod_year}'
    tmp_poly_dir.mkdir(exist_ok=True,parents=True)
    
    sample_log = {}
    result_ext, sample_log, coord_checker, profile, doy, grown = get_poly_stats(params, in_ras, sample_log, quiet=True)
    info = {"cell": cell, "n_polys": 0, "sample_log": sample_log, "coord_checker": coord_checker}
    if not (doy > 0).any():                                    # nothing burned in this cell
        Path(tmp_poly_dir / f'{int(cell):06d}_{object_type}_{mod_year}.empty').touch()
        return info

    if (params['polygonize']['save_ras']) and (params['polygonize']['save_ras'] not in ('False', '')):
        comp_dir = ppaths_cell.scratch / 'comp'
        comp_dir.mkdir(exist_ok=True,parents=True)
        out_ras = comp_dir / f'{object_type}_{mod_year}.tif'
        write_class_doy_raster(out_ras, result_ext, doy, profile, SHAPES[object_type], grown=grown)

    ## removing buffer prior to polygonizing (not does for raster because done in mosaic step)
    res_x, res_y = abs(profile['transform'].a), abs(profile['transform'].e)
    bx = math.ceil(float(params['buffer']) / res_x)       # same as your mosaic code
    by = math.ceil(float(params['buffer']) / res_y)
    rows = slice(by, doy.shape[0] - by)
    cols = slice(bx, doy.shape[1] - bx)
    core_tf = profile['transform'] * Affine.translation(bx, by)
    h, w = doy.shape[0] - 2 * by, doy.shape[1] - 2 * bx
    logger.debug(f'core_tf = {core_tf}')

    ## make polygons without saving for now 
    gdf = array_to_polygons(result_ext[rows, cols], doy[rows, cols], core_tf, profile['crs'],
                            int(params['polygonize']['day_grouping']), SHAPES[object_type], grown=grown[rows, cols])
    if gdf is None or len(gdf) == 0:
        logger.info(f'no polygons created for cell {int(cell):06d}')
        Path(tmp_poly_dir / f'{int(cell):06d}_{object_type}_{mod_year}.empty').touch()
        return info

    west, south, east, north = array_bounds(h, w, core_tf)
    bnd, tol = gdf.bounds, 1e-3
    gdf['edge'] = ((bnd.minx <= west + tol) | (bnd.maxx >= east - tol) |
                   (bnd.miny <= south + tol) | (bnd.maxy >= north - tol))      # touches the cell seam
    gdf['cell'] = cell
    logger.debug(f'gdf = {gdf} \n geometry={gdf.geometry}')
    
    tmp = tmp_poly_dir / f'{int(cell):06d}_{object_type}_{mod_year}.partial'   # write then rename, so a crash can't leave a corrupt file
    gdf.to_file(tmp, driver='GPKG')
    tmp.replace(tmp_poly_dir / f'{int(cell):06d}_{object_type}_{mod_year}.gpkg')
    logger.info(f'polygons saved to {tmp}')
    info["n_polys"] = len(gdf)
    return info

def _append_gpkg(gdf, path, layer='polygons'):
    if gdf is None or len(gdf) == 0:
        return
    gdf.to_file(path, driver='GPKG', layer=layer, mode='a' if Path(path).exists() else 'w')

def _note(path, text):
    if path is None:
        return
    with open(path, 'a') as f:
        f.write(text + '\n')
        
def _cell_job(cell):
    try:
        return cell, classify_cell_with_polygons(_CELL_CTX['params'], cell), None
    except Exception:
        #log = _CELL_CTX['log_file']
        tb = traceback.format_exc()
        #_note(log, f"cell {cell} FAILED (pid {os.getpid()}):\n{tb}")
        print(f"cell {cell} FAILED:\n{tb}", file=sys.stderr, flush=True)
        return cell, None, tb
    #except BaseException:                               # SystemExit etc.: record the line where it hit, then let it go
    #    _note(log, f"cell {cell} interrupted by {sys.exc_info()[0].__name__} (pid {os.getpid()}):\n{traceback.format_exc()}")
        raise
        
def _iter_cells(todo, n_jobs):
    global _FORCE_SERIAL
    if n_jobs == 1:                                     # no pool: ordinary tracebacks, nothing forked
        _FORCE_SERIAL = True
        yield from map(_cell_job, todo)
        return
    with mp.get_context('fork').Pool(min(n_jobs, len(todo)), initializer=_init_worker) as pool:
        yield from pool.imap_unordered(_cell_job, todo)
        
def _finish(gdf, min_area_ha=0.0, simplify_m=0.0, keep_satellites=True, min_area_classes=None):
    """Size filter and simplification. Small polygons are dropped unless they contain
    pixels added by the extension step (when keep_satellites is on)."""
    if min_area_ha and len(gdf):
        keep = gdf["area_ha"] >= min_area_ha
        if keep_satellites and "origin" in gdf.columns:
            keep |= gdf["origin"] != "first_pass"
        if min_area_classes is not None:                       # classes outside this list are never size-filtered
            keep |= ~gdf["cls"].isin(min_area_classes)
        gdf = gdf[keep]
    if simplify_m and len(gdf):
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(simplify_m, preserve_topology=True)
    return gdf

def merge_seam_polygons(edges, window_days, tol=0.01):
    """Union polygons that touch across cell seams (same class and window)."""
    g = edges.reset_index(drop=True)
    g['win'] = (g['doy_min'].astype(int) - 1) // window_days + 1
    probe = gpd.GeoDataFrame(geometry=g.geometry.buffer(tol), crs=g.crs)
    pairs = gpd.sjoin(probe, g[['geometry']], predicate='intersects')
    i, j = pairs.index.to_numpy(), pairs['index_right'].to_numpy()
    cls, win = g['cls'].to_numpy(), g['win'].to_numpy()
    ok = (cls[i] == cls[j]) & (win[i] == win[j])
    graph = coo_matrix((np.ones(int(ok.sum())), (i[ok], j[ok])), shape=(len(g), len(g)))
    _, g['comp'] = connected_components(graph, directed=False)
    g['doy_w'] = g['doy'] * g['area_ha']
    g['ext_w'] = g['ext_frac'] * g['area_ha']
    m = g.dissolve(by='comp', aggfunc={'cls': 'first', 'cls_name': 'first', 'doy_min': 'min',
                                       'doy_max': 'max', 'doy_w': 'sum', 'ext_w': 'sum', 'area_ha': 'sum'})
    m['doy'] = np.rint(m['doy_w'] / m['area_ha']).astype(np.int16)
    m['ext_frac'] = (m['ext_w'] / m['area_ha']).astype(np.float32)
    m['origin'] = _origin(m['ext_frac'].to_numpy())
    return m.drop(columns=['doy_w', 'ext_w']).reset_index(drop=True)


def assemble_polygons(out_path, tmp_ply_dir, window_days, min_area_ha=0.0, simplify_m=0.0, keep_satellites=True, min_area_classes=None):
    out_path = Path(out_path).with_suffix('.gpkg')
    if out_path.exists():
        out_path.unlink()
    edge_parts, n_int = [], 0
    for f in sorted(Path(tmp_ply_dir).glob('*.gpkg')):
        g = gpd.read_file(f)
        e = g['edge'].astype(bool)
        edge_parts.append(g[e])
        inter = _finish(g[~e].drop(columns=['edge', 'cell']), min_area_ha, simplify_m, keep_satellites, min_area_classes)   # complete polygons: write now
        _append_gpkg(inter, out_path)
        n_int += len(inter)
    n_pieces = n_seam = 0
    if edge_parts:
        edges = pd.concat(edge_parts, ignore_index=True).drop(columns=['edge', 'cell'])
        n_pieces = len(edges)
        if n_pieces:
            merged = _finish(merge_seam_polygons(edges, window_days), min_area_ha, simplify_m, keep_satellites, min_area_classes)
            _append_gpkg(merged, out_path)
            n_seam = len(merged)
    logger.info(f'wrote {out_path}: {n_int} interior + {n_seam} seam polygons ({n_pieces} pieces before merging)')
    return out_path
    
    
def polygons_from_classified_raster(params):

    ppaths = ProjectPaths(params, grid=None)
    agg_prefix = params['polygonize']['agg_prefix']
    calibrate_type = params['polygonize']['calibration_type']
    
    object_type = params['iter_models']['optimize_on']
    mod_name = params['polygonize']['mod_name']
    mod_year = params['polygonize']['mod_year']
    
    save_ras = params['polygonize']['save_ras']
    n_jobs = int(_n_jobs(params))
    
    caltab_dir = Path(params['polygonize']['report_dir'])
    if caltab_dir is None or caltab_dir in ('None',''):
        caltab_dir = ppaths.optimization/'polygons'
    caltab_dir.mkdir(exist_ok=True,parents=True)
    caltab_name = f'{object_type}_poly_calibration_table.csv'
    
    calibrate_only = params['polygonize']['calibrate_only'] 
    if not calibrate_only or calibrate_only == 'False':
        calibrate_only = False
    caltab_path = Path(caltab_dir)/f'{agg_prefix}_{object_type}_{mod_year}_polygon_calibration.csv'

    if calibrate_only:
        # re-tune thresholds on a saved table -- does not need to view rasters
        df = pd.read_csv(caltab_dir / caltab_name, index_col='name')
        s = dict(STAGE1, **_stage1_overrides(params['polygonize']['stage1_overrides']))
        calibration_report(df, caltab_dir.with_name(caltab_dir.name + '_retune'), s=s, log=logger.info, mod_name=caltab_name)
        return

    sample_log, coord_checker = {}, []
    
    if calibrate_type.lower() == 'prelim':
        ## get polygons for the cells only covered by the calibration set
        for cell in get_cells_with_pts(params['grid_file'], params['polygonize']['calibration_pts'], params['grids']):
            ppaths_cell = ProjectPaths(params, grid=int(cell))
            in_ras = ppaths_cell.comp / f'{int(cell):06d}_{mod_year}_{mod_name}.tif'
            if not in_ras.is_file():
                logger.warning(f'no raster found for {in_ras}. SKIPPING {int(cell):06d}')
                continue
            logger.info(f'getting data from sample points in cell{int(cell):06d}')
            result_ext,sample_log, pts, profile, doy, _ = get_poly_stats(params, in_ras, sample_log, quiet=True)
            if save_ras and save_ras not in ('False',''):
                out_dir = ppaths_cell.comp
                out_dir.mkdir(parents=True, exist_ok=True)
                out_ras = out_dir / f'{int(cell):06d}_{object_type}_{mod_year}.tif'
                write_class_doy_raster(out_ras, result_ext, doy, profile, SHAPES[object_type])
            coord_checker += pts
            
        logger.info(f'sample results: \n {sample_log}')
    
    else:
        if params['polygonize']['ras_in_dir'] and str(params['polygonize']['ras_in_dir']) not in ('None',''):
            in_ras_dir = params['polygonize']['ras_in_dir']
        else:
            in_ras_dir = ppaths.mosaics
        in_ras = Path(in_ras_dir) / f'{agg_prefix}_{mod_year}_{mod_name}.tif'

        if params['polygonize']['ras_out_dir'] and str(params['polygonize']['ras_out_dir']) not in ('None',''):
            out_ras_dir = params['polygonize']['ras_out_dir']
        else:
            out_ras_dir = ppaths.mosaics

        cells = get_cell_list_from_grid_param(params['grids'])
        tmp_poly_dir = Path(params['scratch_dir']) / f'polys/{mod_year}'
        done = lambda c: any((tmp_poly_dir / f'{c:06d}_{object_type}_{mod_year}{e}').exists() for e in ('.gpkg', '.empty'))
        resume = False
        if params['polygonize']['resume'] and params['polygonize']['resume'] not in ('false',''):
            resume=True
        todo = [c for c in cells if not (resume and done(c))]
        logger.info(f'{len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run on {n_jobs} workers')
        if not todo:
            return {}

        log_file = Path(params['scratch_dir']) / 'polys' / f'{mod_year}' / 'errors.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _CELL_CTX.update(params=params, log_file=log_file)

        results, failed = {}, {}
        for k, (cell, res, err) in enumerate(_iter_cells(todo, n_jobs), start=1):
            if err:
                failed[cell] = err
                status = 'FAILED'
            else:
                results[cell] = res
                status = 'skipped (no raster)' if res is None else f"{res['n_polys']} polygons"
            logger.info(f'[{k}/{len(todo)}] cell {cell}: {status}')

        _CELL_CTX.clear()
        if failed:
            logger.warning(f'{len(failed)} cells failed: {sorted(failed)}')

        sample_log, pts = {}, {}
        for r in results.values():
            if r:
                sample_log.update(r['sample_log'])
                pts.update({p['name']: p for p in r['coord_checker']})
        coord_checker = list(pts.values())
        
        if save_ras and save_ras not in ('False',''):
            out_ras = Path(out_ras_dir) / f'{agg_prefix}_{object_type}_{mod_year}.tif'
            params['classify']['comp_dir'] = 'tmp'
            params['classify']['name'] = f'{object_type}_{mod_year}'
            mosaic_cells(params, out_path=out_ras)
            
        out_poly_dir = params['polygonize']['poly_out_dir']
        if out_poly_dir and str(out_poly_dir) not in ('None', ''):
            out_poly_dir = Path(out_poly_dir)
            out_poly_dir.mkdir(parents=True, exist_ok=True)
            polys_out = out_poly_dir / f'{agg_prefix}_{object_type}_{mod_year}_polys'
            if polys_out.is_file():
                if params['polygonize']['treat_existing_polys'] == 'skip':
                    logger.warning(f" WARNING: Not overwriting the polygon file already existing at {polys_out}. \n") 
                    logger.info("Change polygonize:treat_existing_polys' to 'overwrite' to rewrite polygon file.")
                elif params['polygonize']['treat_existing_polys'] == 'overwrite':
                    polys_out.unlink()
            if not polys_out.is_file():
                logger.info(f"appending polygons to existing set at {polys_out}. \n") 
                logger.info ("If appending is not intended behavior, Change polygonize_treat_existing_polys' to 'overwrite' or 'skip'")
                assemble_polygons(polys_out, tmp_poly_dir, int(params['polygonize']['day_grouping']),
                        min_area_ha=params['polygonize']['min_area_ha'], simplify_m=params['polygonize']['simplify_m'], 
                              keep_satellites=params['polygonize']['keep_satellites'], min_area_classes=params['polygonize']['min_area_classes'])

    if calibrate_type.lower() in('full', 'prelim') and coord_checker:
        #log_tiny_linear(result, result_ext, max_px=5)
        df = calib_table(coord_checker, sample_log, SHAPES[object_type])
        if df.empty:
            logger.info("calibration: no features logged (points outside raster, unburned, or in a hole)")
        else:
            calibration_report(df, caltab_path, log=logger.info, mod_name=caltab_name)
            logger.info(f'saved calibration report at {caltab_path}')
        
    elif calibrate_type.lower() == 'simple' and coord_checker:
        logger.info(f'sample results: \n {sample_log}')

    logger.info('done!')