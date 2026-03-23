#!/usr/bin/env python3
"""
Sapiens Antibody Humanization Dashboard  (Performance-Optimized, Biotech Theme)
================================================================================
A Plotly Dash web application for batch antibody humanization using:
  - sapiens   : Deep learning humanization (Merck, pip install sapiens)
  - abnumber  : Antibody numbering & CDR identification (pip install abnumber)

Theme: Mouse → Human transformation
  - Warm amber/orange  = parental mouse antibody (X0)
  - Cool teal/cyan     = humanized sequence (X1)
  - Violet/purple      = Vernier back-mutated variant (X2)

Requirements:
    pip install dash dash-bootstrap-components dash-ag-grid plotly pandas \
                sapiens abnumber openpyxl xlsxwriter diskcache

Run:
    python biophi_dash_app.py
    Then open http://127.0.0.1:8050
"""

import base64
import io
import logging
import os
import traceback
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import dash
import dash_bootstrap_components as dbc
import diskcache
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, DiskcacheManager, dcc, html
import dash_ag_grid as dag

import sapiens
from abnumber import Chain

# ---------------------------------------------------------------------------
# Logging & GPU detection
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GPU_AVAILABLE = False
GPU_NAME = "N/A"
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
except ImportError:
    pass

MAX_WORKERS = max(1, cpu_count() - 1) if not GPU_AVAILABLE else min(2, cpu_count())

NUMBERING_SCHEMES = ["imgt", "chothia", "kabat"]
CDR_DEFINITIONS = ["imgt", "chothia", "kabat", "north"]

VERNIER_KABAT = {
    "H": {"2", "27", "28", "29", "30", "47", "48", "49",
           "67", "69", "71", "73", "78", "93", "94", "103"},
    "L": {"2", "4", "35", "36", "46", "47", "48", "49",
           "64", "66", "68", "69", "71", "98"},
}

# ---------------------------------------------------------------------------
# Theme palette
# ---------------------------------------------------------------------------
THEME = {
    "bg_dark":      "#0f1923",
    "bg_card":      "#162029",
    "bg_sidebar":   "#0d1520",
    "bg_input":     "#1a2a38",
    "border":       "#1e3448",
    "border_glow":  "#0d9488",
    "text":         "#e2e8f0",
    "text_muted":   "#7494ab",
    "text_heading":  "#f0fdfa",
    # Transformation colors
    "mouse":        "#f59e0b",   # amber — parental
    "mouse_bg":     "#451a03",
    "human":        "#14b8a6",   # teal — humanized
    "human_bg":     "#042f2e",
    "vernier":      "#a78bfa",   # violet — back-mutated
    "vernier_bg":   "#1e1b4b",
    # Accents
    "accent":       "#0d9488",
    "accent_glow":  "rgba(13,148,136,0.25)",
    "cdr_yellow":   "#fbbf24",
    "cdr_bg":       "#78350f",
    "fw_dim":       "#334155",
    "mutation_red":  "#ef4444",
    "success":      "#10b981",
    "danger":       "#ef4444",
}

# Plotly chart template
PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1923",
        font=dict(color="#e2e8f0", family="JetBrains Mono, monospace", size=12),
        xaxis=dict(gridcolor="#1e3448", zerolinecolor="#1e3448"),
        yaxis=dict(gridcolor="#1e3448", zerolinecolor="#1e3448"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        title=dict(font=dict(color="#f0fdfa", size=15)),
    )
)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# ---------------------------------------------------------------------------
# Pre-warm models
# ---------------------------------------------------------------------------
print("\n  Loading Sapiens models...")
try:
    _dummy_vh = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDKILWFGEPVFDYWGQGTLVTVSS"
    _dummy_vl = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
    sapiens.predict_scores(_dummy_vh, "H")
    sapiens.predict_scores(_dummy_vl, "L")
    print("  Models loaded.\n")
except Exception as e:
    print(f"  Warning: {e}\n")


# ---------------------------------------------------------------------------
# Core helpers (unchanged logic)
# ---------------------------------------------------------------------------

def get_cdr_positions(chain: Chain) -> set:
    full_seq = chain.seq
    cdr_pos_set = set()
    search_start = 0
    for cdr_attr in ("cdr1_seq", "cdr2_seq", "cdr3_seq"):
        cdr_seq = getattr(chain, cdr_attr, None)
        if cdr_seq:
            idx = full_seq.find(cdr_seq, search_start)
            if idx >= 0:
                for j in range(idx, idx + len(cdr_seq)):
                    cdr_pos_set.add(j)
                search_start = idx + len(cdr_seq)
    return cdr_pos_set


def get_vernier_positions(sequence: str, chain_type: str) -> set:
    vernier_set = VERNIER_KABAT.get(chain_type, set())
    vernier_indices = set()
    try:
        kabat_chain = Chain(sequence, scheme="kabat")
        for seq_idx, (pos, aa) in enumerate(kabat_chain):
            pos_str = str(pos)
            num_part = ""
            for ch in pos_str:
                if ch.isdigit():
                    num_part += ch
                elif num_part:
                    break
            if num_part in vernier_set:
                vernier_indices.add(seq_idx)
    except Exception as e:
        logger.warning(f"Kabat Vernier failed: {e}")
    return vernier_indices


def annotate_positions(chain: Chain) -> list:
    chain_type = chain.chain_type
    full_seq = chain.seq
    vernier_set = VERNIER_KABAT.get(chain_type, set())
    cdr_indices = set()
    search_start = 0
    for cdr_attr in ("cdr1_seq", "cdr2_seq", "cdr3_seq"):
        cdr_seq = getattr(chain, cdr_attr, None)
        if cdr_seq:
            idx = full_seq.find(cdr_seq, search_start)
            if idx >= 0:
                for j in range(idx, idx + len(cdr_seq)):
                    cdr_indices.add(j)
                search_start = idx + len(cdr_seq)
    kabat_vernier_indices = set()
    try:
        kabat_chain = Chain(full_seq, scheme="kabat")
        for seq_idx, (pos, aa) in enumerate(kabat_chain):
            pos_str = str(pos)
            num_part = ""
            for ch in pos_str:
                if ch.isdigit():
                    num_part += ch
                elif num_part:
                    break
            if num_part in vernier_set:
                kabat_vernier_indices.add(seq_idx)
    except Exception:
        pass
    annotations = []
    for i, aa in enumerate(full_seq):
        if i in cdr_indices:
            region = "CDR"
        elif i in kabat_vernier_indices:
            region = "Vernier"
        else:
            region = "Framework"
        annotations.append({"idx": i, "region": region, "aa": aa})
    return annotations


def _calc_mean_sapiens_score(sequence, scores_df):
    total = 0.0
    for i, aa in enumerate(sequence):
        if aa in scores_df.columns and i < len(scores_df):
            total += scores_df.iloc[i][aa]
    return total / len(sequence) if len(sequence) > 0 else 0.0


def humanize_single_chain(sequence, chain_type, scheme, cdr_definition,
                          humanize_cdrs, iterations):
    result = {
        "original_seq": sequence, "humanized_seq": None,
        "vernier_backmut_seq": None,
        "mutations_h1": None, "num_mutations_h1": 0,
        "mutations_h2": None, "num_mutations_h2": 0,
        "vernier_mutations_in_h1": None,
        "sapiens_score_orig": None, "sapiens_score_hum": None,
        "sapiens_score_backmut": None, "error": None,
    }
    try:
        chain = Chain(sequence, scheme=scheme, cdr_definition=cdr_definition)
    except Exception as e:
        result["error"] = f"Numbering failed: {e}"
        return result
    detected_ct = chain.chain_type
    try:
        cdr_indices = get_cdr_positions(chain)
    except Exception:
        cdr_indices = set()
    current_seq = chain.seq
    result["original_seq"] = current_seq
    try:
        scores_df = sapiens.predict_scores(current_seq, detected_ct)
        result["sapiens_score_orig"] = round(
            _calc_mean_sapiens_score(current_seq, scores_df), 4)
    except Exception as e:
        result["error"] = f"Sapiens prediction failed: {e}"
        return result
    new_seq_list = list(current_seq)
    for i in range(len(current_seq)):
        if not humanize_cdrs and i in cdr_indices:
            continue
        new_seq_list[i] = scores_df.iloc[i].idxmax()
    current_seq = "".join(new_seq_list)
    for _ in range(1, iterations):
        try:
            scores_df = sapiens.predict_scores(current_seq, detected_ct)
        except Exception as e:
            result["error"] = f"Sapiens iter failed: {e}"
            return result
        new_seq_list = list(current_seq)
        for i in range(len(current_seq)):
            if not humanize_cdrs and i in cdr_indices:
                continue
            new_seq_list[i] = scores_df.iloc[i].idxmax()
        current_seq = "".join(new_seq_list)
    humanized_seq = current_seq
    result["humanized_seq"] = humanized_seq
    try:
        hum_scores = sapiens.predict_scores(humanized_seq, detected_ct)
        result["sapiens_score_hum"] = round(
            _calc_mean_sapiens_score(humanized_seq, hum_scores), 4)
    except Exception:
        pass
    original = result["original_seq"]
    mutations_h1 = [f"{o}{i+1}{h}" for i, (o, h) in
                    enumerate(zip(original, humanized_seq)) if o != h]
    result["mutations_h1"] = "; ".join(mutations_h1) if mutations_h1 else "None"
    result["num_mutations_h1"] = len(mutations_h1)
    vernier_indices = get_vernier_positions(original, detected_ct)
    vernier_muts = [(i, original[i], humanized_seq[i]) for i in vernier_indices
                    if i < len(original) and i < len(humanized_seq)
                    and original[i] != humanized_seq[i]]
    if vernier_muts:
        backmut_list = list(humanized_seq)
        for pos, orig_aa, _ in vernier_muts:
            backmut_list[pos] = orig_aa
        backmut_seq = "".join(backmut_list)
        result["vernier_backmut_seq"] = backmut_seq
        result["vernier_mutations_in_h1"] = "; ".join(
            [f"{o}{p+1}{h}" for p, o, h in vernier_muts])
        try:
            bm_scores = sapiens.predict_scores(backmut_seq, detected_ct)
            result["sapiens_score_backmut"] = round(
                _calc_mean_sapiens_score(backmut_seq, bm_scores), 4)
        except Exception:
            pass
        mutations_h2 = [f"{o}{i+1}{b}" for i, (o, b) in
                        enumerate(zip(original, backmut_seq)) if o != b]
        result["mutations_h2"] = "; ".join(mutations_h2) if mutations_h2 else "None"
        result["num_mutations_h2"] = len(mutations_h2)
    else:
        result["vernier_backmut_seq"] = None
        result["vernier_mutations_in_h1"] = "None (all Vernier preserved)"
    return result


def _worker_humanize(args):
    seq, ct, scheme, cdr_def, humanize_cdrs, iterations = args
    return humanize_single_chain(seq, ct, scheme, cdr_def, humanize_cdrs, iterations)


def parse_upload(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        raw = pd.read_excel(io.BytesIO(decoded))
    elif ext == "csv":
        raw = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    elif ext == "tsv":
        raw = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    name_col = None
    for c in ("name", "id", "antibody_name", "antibody_id", "ab_name"):
        if c in raw.columns:
            name_col = c
            break
    if name_col is None:
        raw["name"] = [f"Ab_{i+1}" for i in range(len(raw))]
        name_col = "name"
    heavy_col = next((c for c in raw.columns if c in (
        "heavy", "vh", "heavy_chain", "heavy_sequence", "vh_sequence")), None)
    light_col = next((c for c in raw.columns if c in (
        "light", "vl", "light_chain", "light_sequence", "vl_sequence")), None)
    if heavy_col or light_col:
        return pd.DataFrame({"name": raw[name_col],
                             "heavy": raw[heavy_col] if heavy_col else None,
                             "light": raw[light_col] if light_col else None})
    if "sequence" in raw.columns:
        return pd.DataFrame({"name": raw[name_col], "heavy": raw["sequence"],
                             "light": None})
    raise ValueError("Could not find sequence columns.")


def run_batch(df, scheme, cdr_def, humanize_cdrs, iterations,
              n_workers=1, set_progress=None):
    task_list = []
    for idx, row in df.iterrows():
        for label, col, ct, prefix in [("VH","heavy","H","H"),("VL","light","L","L")]:
            seq = row.get(col)
            if pd.isna(seq) or seq is None or str(seq).strip() == "":
                continue
            task_list.append((idx, row["name"], label, prefix, str(seq).strip(), ct))
    total_chains = len(task_list)
    if total_chains == 0:
        return []
    unique_keys = OrderedDict()
    for task in task_list:
        key = (task[4], task[5])
        if key not in unique_keys:
            unique_keys[key] = None
    n_unique = len(unique_keys)
    dedup_savings = total_chains - n_unique
    if set_progress:
        dedup_msg = f" ({dedup_savings} cached)" if dedup_savings > 0 else ""
        set_progress((f"Processing {n_unique} unique chains{dedup_msg}...", 0))
    seq_cache = {}
    if n_workers > 1 and n_unique >= 4:
        worker_args = [(s, c, scheme, cdr_def, humanize_cdrs, iterations)
                       for s, c in unique_keys.keys()]
        keys_list = list(unique_keys.keys())
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futs = {executor.submit(_worker_humanize, a): i
                    for i, a in enumerate(worker_args)}
            done_count = 0
            for f in as_completed(futs):
                i = futs[f]
                try:
                    seq_cache[keys_list[i]] = f.result()
                except Exception as e:
                    seq_cache[keys_list[i]] = {
                        "original_seq": keys_list[i][0], "humanized_seq": None,
                        "vernier_backmut_seq": None, "mutations_h1": None,
                        "num_mutations_h1": 0, "mutations_h2": None,
                        "num_mutations_h2": 0, "vernier_mutations_in_h1": None,
                        "sapiens_score_orig": None, "sapiens_score_hum": None,
                        "sapiens_score_backmut": None, "error": str(e),
                    }
                done_count += 1
                if set_progress:
                    set_progress((f"{done_count}/{n_unique} chains...",
                                  int(done_count / n_unique * 95)))
    else:
        for done_count, (seq, ct) in enumerate(unique_keys.keys()):
            if set_progress:
                set_progress((f"Chain {done_count+1}/{n_unique}...",
                              int(done_count / n_unique * 95)))
            seq_cache[(seq, ct)] = humanize_single_chain(
                seq, ct, scheme, cdr_def, humanize_cdrs, iterations)
    results = []
    current_name = None
    rec = None
    for idx, name, label, prefix, seq, ct in task_list:
        if name != current_name:
            if rec is not None:
                results.append(rec)
            rec = {"name": name}
            current_name = name
        hum = seq_cache[(seq, ct)]
        rec[f"{prefix}0_sequence"] = hum["original_seq"]
        rec[f"{prefix}0_sapiens"] = hum["sapiens_score_orig"]
        rec[f"{prefix}1_sequence"] = hum["humanized_seq"]
        rec[f"{prefix}1_sapiens"] = hum["sapiens_score_hum"]
        rec[f"{prefix}1_mutations"] = hum["mutations_h1"]
        rec[f"{prefix}1_num_mutations"] = hum["num_mutations_h1"]
        rec[f"{prefix}1_vernier_mutations"] = hum["vernier_mutations_in_h1"]
        if hum["vernier_backmut_seq"] is not None:
            rec[f"{prefix}2_sequence"] = hum["vernier_backmut_seq"]
            rec[f"{prefix}2_sapiens"] = hum["sapiens_score_backmut"]
            rec[f"{prefix}2_mutations"] = hum["mutations_h2"]
            rec[f"{prefix}2_num_mutations"] = hum["num_mutations_h2"]
            rec[f"{prefix}2_note"] = f"Vernier back-mutated from {prefix}1"
        else:
            rec[f"{prefix}2_sequence"] = None
            rec[f"{prefix}2_note"] = f"Not needed ({prefix}1 has no Vernier mutations)"
        rec[f"{label}_error"] = hum["error"]
    if rec is not None:
        results.append(rec)
    if set_progress:
        set_progress(("Complete!", 100))
    return results


def build_per_antibody_sheets(result_df):
    sheets = {}
    for _, row in result_df.iterrows():
        ab_name = str(row["name"])
        sn = ab_name[:31].replace("/","_").replace("\\","_").replace("?","").replace(
            "*","").replace("[","").replace("]","").replace(":","")
        base_name = sn
        counter = 1
        while sn in sheets:
            sn = f"{base_name[:28]}_{counter}"
            counter += 1
        rows = []
        for pf, cl in [("H","Heavy"),("L","Light")]:
            for ver, desc in [("0","Original (parental)"),("1","Sapiens humanized"),
                              ("2","Vernier back-mutated")]:
                sc = f"{pf}{ver}_sequence"
                if sc in row and pd.notna(row.get(sc)):
                    rows.append({
                        "Variant": f"{pf}{ver}", "Chain": cl, "Description": desc,
                        "Sequence": row[sc],
                        "Sapiens Score": row.get(f"{pf}{ver}_sapiens"),
                        "Mutations vs Original": row.get(f"{pf}{ver}_mutations", "\u2014") if ver != "0" else "\u2014",
                        "# Mutations": row.get(f"{pf}{ver}_num_mutations", 0) if ver != "0" else 0,
                        "Notes": row.get(f"{pf}{ver}_note", "") if ver == "2" else (
                            f"Vernier mutations: {row.get(f'{pf}1_vernier_mutations','')}" if ver == "1" else ""),
                    })
        if rows:
            sheets[sn] = pd.DataFrame(rows)
    return sheets


# ============================================================================
# DASH APP
# ============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark: """ + THEME["bg_dark"] + """;
    --bg-card: """ + THEME["bg_card"] + """;
    --bg-sidebar: """ + THEME["bg_sidebar"] + """;
    --border: """ + THEME["border"] + """;
    --accent: """ + THEME["accent"] + """;
    --accent-glow: """ + THEME["accent_glow"] + """;
    --text: """ + THEME["text"] + """;
    --text-muted: """ + THEME["text_muted"] + """;
    --mouse: """ + THEME["mouse"] + """;
    --human: """ + THEME["human"] + """;
    --vernier: """ + THEME["vernier"] + """;
}

body {
    background-color: var(--bg-dark) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Navbar */
.navbar-brand { font-family: 'JetBrains Mono', monospace !important; letter-spacing: -0.5px; }

/* Sidebar */
.sidebar-custom {
    background: linear-gradient(180deg, #0d1520 0%, #0a1018 100%) !important;
    border-right: 1px solid var(--border) !important;
    padding: 1.5rem;
    height: 100%;
    overflow-y: auto;
}
.sidebar-custom label, .sidebar-custom .form-label {
    color: #7494ab !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}
.sidebar-custom .Select-control, .sidebar-custom .Select-menu-outer {
    background-color: #1a2a38 !important;
}

/* Cards */
.card-bio {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.card-bio .card-header {
    background: linear-gradient(90deg, rgba(13,148,136,0.08), rgba(13,148,136,0.02)) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0.9rem 1.2rem;
}
.card-bio .card-header h5 {
    color: #f0fdfa !important;
    font-weight: 600;
    font-size: 0.95rem;
    font-family: 'JetBrains Mono', monospace;
}
.card-bio .card-body { background: var(--bg-card) !important; }

/* Step badges in card headers */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), #0f766e);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* Upload area */
.upload-zone {
    border: 2px dashed #1e3448 !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, rgba(13,148,136,0.04), rgba(245,158,11,0.03)) !important;
    padding: 50px 20px !important;
    cursor: pointer;
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: var(--accent) !important;
    background: linear-gradient(135deg, rgba(13,148,136,0.08), rgba(245,158,11,0.05)) !important;
    box-shadow: 0 0 30px rgba(13,148,136,0.1);
}

/* Species icons in upload */
.species-arrow { color: var(--accent); font-size: 1.5rem; margin: 0 12px; }

/* Dropdowns (Dash native) */
.Select-control { background-color: #1a2a38 !important; border-color: #1e3448 !important; }
.Select-value-label, .Select-placeholder { color: #94a3b8 !important; }
.Select-menu-outer { background-color: #1a2a38 !important; border-color: #1e3448 !important; }
.VirtualizedSelectOption { background: #1a2a38 !important; color: #e2e8f0 !important; }
.VirtualizedSelectFocusedOption { background: #0d9488 !important; color: white !important; }

/* Slider */
.rc-slider-track { background-color: var(--accent) !important; }
.rc-slider-handle { border-color: var(--accent) !important; background: var(--accent) !important; }
.rc-slider-dot-active { border-color: var(--accent) !important; }
.rc-slider-rail { background-color: #1e3448 !important; }

/* Progress bar */
.progress { background-color: #1e3448 !important; border-radius: 8px !important; }
.progress-bar {
    background: linear-gradient(90deg, var(--mouse), var(--human)) !important;
    border-radius: 8px !important;
}

/* AG Grid dark theme overrides */
.ag-theme-alpine-dark {
    --ag-background-color: #162029 !important;
    --ag-header-background-color: #0f1923 !important;
    --ag-odd-row-background-color: #1a2a38 !important;
    --ag-row-hover-color: rgba(13,148,136,0.12) !important;
    --ag-border-color: #1e3448 !important;
    --ag-foreground-color: #e2e8f0 !important;
    --ag-header-foreground-color: #94a3b8 !important;
    --ag-font-family: 'Outfit', sans-serif !important;
    --ag-font-size: 13px !important;
    --ag-input-focus-border-color: #0d9488 !important;
}

/* Accordion */
.accordion-item { background: var(--bg-card) !important; border-color: var(--border) !important; }
.accordion-button {
    background: var(--bg-card) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.accordion-button:not(.collapsed) {
    background: rgba(13,148,136,0.08) !important;
    color: #14b8a6 !important;
}
.accordion-button::after { filter: invert(0.7); }
.accordion-body { background: var(--bg-card) !important; }

/* Alerts */
.alert-success-bio {
    background: linear-gradient(90deg, rgba(16,185,129,0.12), rgba(13,148,136,0.06)) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 10px;
    color: #6ee7b7 !important;
}
.alert-danger-bio {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px;
    color: #fca5a5 !important;
}

/* Buttons */
.btn-run {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    border: none !important;
    color: white !important;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    border-radius: 10px !important;
    padding: 12px !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(13,148,136,0.25);
}
.btn-run:hover:not(:disabled) {
    box-shadow: 0 6px 25px rgba(13,148,136,0.4) !important;
    transform: translateY(-1px);
}
.btn-run:disabled { opacity: 0.4 !important; }

.btn-dl-csv { background: linear-gradient(135deg, #10b981, #059669) !important; border: none !important; border-radius: 8px !important; }
.btn-dl-xlsx { background: linear-gradient(135deg, #0d9488, #0f766e) !important; border: none !important; border-radius: 8px !important; }
.btn-dl-fasta { background: linear-gradient(135deg, #6366f1, #4f46e5) !important; border: none !important; border-radius: 8px !important; }

/* Switch */
.form-check-input:checked { background-color: var(--accent) !important; border-color: var(--accent) !important; }

/* Alignment block */
.alignment-block {
    background: #0a1018 !important;
    border: 1px solid #1e3448;
    border-radius: 10px;
    padding: 14px;
    overflow-x: auto;
    white-space: nowrap;
}
"""

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap",
    ],
    title="Sapiens Humanization",
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
)

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>''' + CUSTOM_CSS + '''</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def make_card_header(step_num, title):
    return dbc.CardHeader([
        html.Span(str(step_num), className="step-badge"),
        html.Span(title, style={"color": "#f0fdfa", "fontWeight": "600",
                                "fontFamily": "'JetBrains Mono', monospace",
                                "fontSize": "0.9rem"}),
    ])

sidebar = html.Div([
    # Logo area
    html.Div([
        html.Div("SAPIENS", style={
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize": "1.3rem", "fontWeight": "700",
            "color": THEME["human"],
            "letterSpacing": "3px",
        }),
        html.Div("humanization engine", style={
            "fontSize": "0.7rem", "color": THEME["text_muted"],
            "letterSpacing": "2px", "textTransform": "uppercase",
            "marginTop": "2px",
        }),
    ], style={"textAlign": "center", "padding": "8px 0 20px 0",
              "borderBottom": f"1px solid {THEME['border']}", "marginBottom": "20px"}),

    # Transformation visual
    html.Div([
        html.Span("\U0001f42d", style={"fontSize": "1.2rem"}),
        html.Span(" \u2192 ", style={"color": THEME["accent"], "fontWeight": "bold",
                                     "margin": "0 6px"}),
        html.Span("\U0001f9ec", style={"fontSize": "1.2rem"}),
        html.Span(" \u2192 ", style={"color": THEME["accent"], "fontWeight": "bold",
                                     "margin": "0 6px"}),
        html.Span("\U0001f9d1\u200d\u2695\ufe0f", style={"fontSize": "1.2rem"}),
    ], style={"textAlign": "center", "marginBottom": "20px", "padding": "10px",
              "background": "rgba(13,148,136,0.06)", "borderRadius": "10px",
              "border": f"1px solid {THEME['border']}"}),

    html.Div("PARAMETERS", style={
        "fontSize": "0.7rem", "color": THEME["text_muted"],
        "letterSpacing": "2px", "marginBottom": "12px", "fontWeight": "600",
    }),

    dbc.Label("Numbering Scheme", style={"color": THEME["text_muted"]}),
    dcc.Dropdown(id="scheme-dropdown",
                 options=[{"label": s.upper(), "value": s} for s in NUMBERING_SCHEMES],
                 value="kabat", clearable=False),
    html.Div(style={"height": "12px"}),

    dbc.Label("CDR Definition", style={"color": THEME["text_muted"]}),
    dcc.Dropdown(id="cdr-dropdown",
                 options=[{"label": s.upper(), "value": s} for s in CDR_DEFINITIONS],
                 value="kabat", clearable=False),
    html.Div(style={"height": "12px"}),

    dbc.Label("Sapiens Iterations", style={"color": THEME["text_muted"]}),
    dcc.Slider(id="iterations-slider", min=1, max=5, step=1, value=1,
               marks={i: {"label": str(i), "style": {"color": THEME["text_muted"]}}
                      for i in range(1, 6)}),
    html.Div(style={"height": "8px"}),

    dbc.Checklist(id="humanize-cdrs-check",
                  options=[{"label": " Humanize CDRs", "value": "yes"}],
                  value=[], switch=True,
                  style={"color": THEME["text"]}),

    html.Hr(style={"borderColor": THEME["border"], "margin": "18px 0"}),

    html.Div("PERFORMANCE", style={
        "fontSize": "0.7rem", "color": THEME["text_muted"],
        "letterSpacing": "2px", "marginBottom": "12px", "fontWeight": "600",
    }),
    dbc.Label(f"Workers (max {cpu_count()})", style={"color": THEME["text_muted"]}),
    dcc.Slider(id="workers-slider", min=1, max=max(cpu_count(), 2), step=1,
               value=MAX_WORKERS,
               marks={1: {"label": "1", "style": {"color": THEME["text_muted"]}},
                      max(cpu_count(),2): {"label": str(max(cpu_count(),2)),
                                           "style": {"color": THEME["text_muted"]}}}),
    html.Small(
        f"{'GPU: ' + GPU_NAME if GPU_AVAILABLE else 'CPU mode'}",
        style={"color": THEME["text_muted"], "fontSize": "0.7rem"}),

    html.Hr(style={"borderColor": THEME["border"], "margin": "18px 0"}),

    dbc.Button([
        html.Span("\u25b6 ", style={"marginRight": "4px"}),
        "RUN HUMANIZATION",
    ], id="run-btn", className="btn-run w-100", size="lg", disabled=True),
    html.Div(id="cancel-btn-container", className="mt-2"),
], className="sidebar-custom")

upload_area = dcc.Upload(
    id="upload-data",
    children=html.Div([
        html.Div([
            html.Span("\U0001f42d", style={"fontSize": "2rem", "opacity": "0.7"}),
            html.Span(" \u2794 ", style={"fontSize": "1.5rem", "color": THEME["accent"],
                                         "margin": "0 10px"}),
            html.Span("\U0001f9ec", style={"fontSize": "2rem", "opacity": "0.7"}),
        ], style={"marginBottom": "12px"}),
        html.Div([
            html.Span("Drop antibody sequences or ", style={"color": THEME["text_muted"]}),
            html.Span("browse files", style={"color": THEME["human"], "fontWeight": "600",
                                             "textDecoration": "underline",
                                             "cursor": "pointer"}),
        ]),
        html.Div("Accepts .xlsx, .xls, .csv, .tsv",
                 style={"color": THEME["text_muted"], "fontSize": "0.75rem",
                        "marginTop": "6px", "opacity": "0.6"}),
    ], style={"textAlign": "center"}),
    className="upload-zone",
    multiple=False,
)

app.layout = dbc.Container(fluid=True, className="px-0", children=[
    # Navbar
    dbc.Navbar(
        dbc.Container([
            html.Div([
                dbc.NavbarBrand([
                    html.Span("\U0001f9ec ", style={"fontSize": "1.2rem"}),
                    html.Span("SAPIENS", style={"fontWeight": "700",
                                                 "letterSpacing": "2px"}),
                    html.Span(" HUMANIZATION", style={"fontWeight": "300",
                                                       "opacity": "0.7",
                                                       "letterSpacing": "2px"}),
                ], style={"fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.9rem"}),
            ]),
            html.Div([
                html.Span("mouse", style={"color": THEME["mouse"], "fontSize": "0.7rem",
                                          "fontFamily": "'JetBrains Mono', monospace",
                                          "opacity": "0.8"}),
                html.Span(" \u2192 ", style={"color": THEME["text_muted"], "margin": "0 6px"}),
                html.Span("human", style={"color": THEME["human"], "fontSize": "0.7rem",
                                          "fontFamily": "'JetBrains Mono', monospace",
                                          "opacity": "0.8"}),
            ]),
        ], fluid=True, className="d-flex justify-content-between align-items-center"),
        style={"background": "linear-gradient(90deg, #0a1018, #0d1520, #0a1018)",
               "borderBottom": f"1px solid {THEME['border']}",
               "padding": "10px 0"},
        dark=True, className="mb-0",
    ),

    dbc.Row([
        dbc.Col(sidebar, width=3, className="p-0",
                style={"minHeight": "calc(100vh - 56px)"}),
        dbc.Col([
            # Upload card
            dbc.Card([
                make_card_header(1, "UPLOAD SEQUENCES"),
                dbc.CardBody([upload_area,
                              html.Div(id="upload-status", className="mt-3"),
                              html.Div(id="preview-table", className="mt-3")]),
            ], className="card-bio mb-4 mt-3"),

            # Progress
            html.Div(id="progress-container", children=[
                html.Div(id="progress-status-text", className="mb-2",
                         style={"color": THEME["human"], "fontWeight": "500",
                                "fontFamily": "'JetBrains Mono', monospace",
                                "fontSize": "0.8rem"}),
                dbc.Progress(id="progress-bar", value=0, striped=True, animated=True,
                             className="mb-3", style={"height": "8px"}),
            ], style={"display": "none"}),

            html.Div(id="run-status", className="mb-3"),
            html.Div(id="results-container"),

            dcc.Store(id="parsed-data-store"),
            dcc.Store(id="results-store"),
        ], width=9, style={"padding": "0 28px",
                           "background": "linear-gradient(180deg, #0f1923 0%, #0c1219 100%)"}),
    ], className="g-0"),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("parsed-data-store", "data"),
    Output("upload-status", "children"),
    Output("preview-table", "children"),
    Output("run-btn", "disabled"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return dash.no_update, "", "", True
    try:
        df = parse_upload(contents, filename)
    except Exception as e:
        return None, html.Div(str(e), className="alert-danger-bio p-3 mt-2"), "", True
    n = len(df)
    has_h = df["heavy"].notna().sum()
    has_l = df["light"].notna().sum() if "light" in df.columns else 0
    uniq_h = df["heavy"].dropna().nunique() if has_h else 0
    uniq_l = df["light"].dropna().nunique() if has_l else 0
    total_u = uniq_h + uniq_l
    total_c = has_h + has_l
    dedup = f" | {total_c - total_u} duplicates will be cached" if total_c > total_u else ""

    status = html.Div([
        html.Div([
            html.Span("\u2705 ", style={"marginRight": "6px"}),
            html.Strong(filename, style={"color": THEME["human"]}),
            html.Span(f" \u2014 {n} antibodies loaded",
                      style={"color": THEME["text"]}),
        ]),
        html.Div(
            f"{has_h} heavy \u00b7 {has_l} light \u00b7 {total_u} unique sequences{dedup}",
            style={"color": THEME["text_muted"], "fontSize": "0.78rem", "marginTop": "4px"},
        ),
    ], className="alert-success-bio p-3 mt-2")

    preview = df.head(5).copy()
    for col in ("heavy", "light"):
        if col in preview.columns:
            preview[col] = preview[col].apply(
                lambda x: (str(x)[:45] + "\u2026") if pd.notna(x) and len(str(x)) > 45 else x)

    table = dag.AgGrid(
        rowData=preview.to_dict("records"),
        columnDefs=[{"field": c, "headerName": c.upper(), "minWidth": 120,
                     "flex": 2 if c in ("heavy","light") else 1} for c in preview.columns],
        defaultColDef={"resizable": True, "sortable": True},
        style={"height": "210px"},
        className="ag-theme-alpine-dark",
    )
    return df.to_json(date_format="iso", orient="split"), status, table, False


@dash.callback(
    Output("results-store", "data"),
    Output("results-container", "children"),
    Output("run-status", "children"),
    Input("run-btn", "n_clicks"),
    State("parsed-data-store", "data"),
    State("scheme-dropdown", "value"),
    State("cdr-dropdown", "value"),
    State("iterations-slider", "value"),
    State("humanize-cdrs-check", "value"),
    State("workers-slider", "value"),
    prevent_initial_call=True,
    background=True,
    running=[
        (Output("run-btn", "disabled"), True, False),
        (Output("run-btn", "children"), "RUNNING...", [
            html.Span("\u25b6 ", style={"marginRight": "4px"}), "RUN HUMANIZATION"]),
        (Output("progress-container", "style"), {"display": "block"}, {"display": "none"}),
    ],
    progress=[Output("progress-status-text", "children"),
              Output("progress-bar", "value")],
)
def run_humanization(set_progress, n_clicks, json_data, scheme, cdr_def,
                     iterations, humanize_cdrs_val, n_workers):
    if json_data is None:
        return dash.no_update, dash.no_update, ""
    df = pd.read_json(io.StringIO(json_data), orient="split")
    humanize_cdrs = "yes" in (humanize_cdrs_val or [])
    n_workers = int(n_workers) if n_workers else 1
    set_progress(("Initializing Sapiens...", 0))
    t0 = time.time()
    try:
        results = run_batch(df, scheme, cdr_def, humanize_cdrs, iterations,
                            n_workers=n_workers, set_progress=set_progress)
    except Exception as e:
        logger.error(traceback.format_exc())
        return None, "", html.Div(f"Error: {e}", className="alert-danger-bio p-3")
    elapsed = time.time() - t0
    result_df = pd.DataFrame(results)
    children = []

    # ── 2. Summary AG Grid ──
    summary_rows = []
    for _, row in result_df.iterrows():
        for pf, cl in [("H","Heavy"),("L","Light")]:
            if f"{pf}0_sequence" not in row or pd.isna(row.get(f"{pf}0_sequence")):
                continue
            base = {"Antibody": row["name"], "Chain": cl}
            summary_rows.append({**base, "Variant": f"{pf}0", "Type": "\U0001f42d Original",
                "Sapiens Score": row.get(f"{pf}0_sapiens"),
                "Mutations": 0, "Vernier Status": "\u2014"})
            summary_rows.append({**base, "Variant": f"{pf}1", "Type": "\U0001f9ec Humanized",
                "Sapiens Score": row.get(f"{pf}1_sapiens"),
                "Mutations": row.get(f"{pf}1_num_mutations", 0),
                "Vernier Status": row.get(f"{pf}1_vernier_mutations", "")})
            if f"{pf}2_sequence" in row and pd.notna(row.get(f"{pf}2_sequence")):
                summary_rows.append({**base, "Variant": f"{pf}2",
                    "Type": "\U0001f9d1\u200d\u2695\ufe0f Vernier BM",
                    "Sapiens Score": row.get(f"{pf}2_sapiens"),
                    "Mutations": row.get(f"{pf}2_num_mutations", 0),
                    "Vernier Status": "All restored"})

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        col_defs = [
            {"field": "Antibody", "pinned": "left", "minWidth": 130,
             "filter": "agTextColumnFilter"},
            {"field": "Chain", "maxWidth": 90, "filter": "agSetColumnFilter"},
            {"field": "Variant", "maxWidth": 80, "filter": "agSetColumnFilter"},
            {"field": "Type", "minWidth": 160, "filter": "agSetColumnFilter"},
            {"field": "Sapiens Score", "minWidth": 130, "filter": "agNumberColumnFilter",
             "valueFormatter": {"function": "params.value != null ? d3.format('.4f')(params.value) : ''"},
             "cellStyle": {"function": """
                params.value == null ? {} :
                params.value >= 0.9 ? {'backgroundColor': '#042f2e', 'color': '#5eead4'} :
                params.value >= 0.7 ? {'backgroundColor': '#451a03', 'color': '#fbbf24'} :
                {'backgroundColor': '#450a0a', 'color': '#fca5a5'}
             """}},
            {"field": "Mutations", "maxWidth": 100, "filter": "agNumberColumnFilter"},
            {"field": "Vernier Status", "minWidth": 180, "filter": "agTextColumnFilter"},
        ]
        row_style = {"styleConditions": [
            {"condition": "params.data.Variant && params.data.Variant.endsWith('0')",
             "style": {"borderLeft": f"3px solid {THEME['mouse']}"}},
            {"condition": "params.data.Variant && params.data.Variant.endsWith('1')",
             "style": {"borderLeft": f"3px solid {THEME['human']}"}},
            {"condition": "params.data.Variant && params.data.Variant.endsWith('2')",
             "style": {"borderLeft": f"3px solid {THEME['vernier']}"}},
        ]}
        children.append(dbc.Card([
            make_card_header(2, "HUMANIZATION SUMMARY"),
            dbc.CardBody([
                dag.AgGrid(id="summary-grid", rowData=sdf.to_dict("records"),
                    columnDefs=col_defs,
                    defaultColDef={"resizable": True, "sortable": True, "filter": True,
                                   "floatingFilter": True},
                    dashGridOptions={"animateRows": True, "pagination": True,
                                     "paginationAutoPageSize": True,
                                     "enableCellTextSelection": True},
                    getRowStyle=row_style,
                    style={"height": "500px"},
                    className="ag-theme-alpine-dark"),
                # Legend
                html.Div([
                    html.Span("\u2588", style={"color": THEME["mouse"], "marginRight": "4px"}),
                    html.Span("X0 Original  ", style={"fontSize": "0.72rem", "marginRight": "14px",
                                                       "color": THEME["text_muted"]}),
                    html.Span("\u2588", style={"color": THEME["human"], "marginRight": "4px"}),
                    html.Span("X1 Humanized  ", style={"fontSize": "0.72rem", "marginRight": "14px",
                                                        "color": THEME["text_muted"]}),
                    html.Span("\u2588", style={"color": THEME["vernier"], "marginRight": "4px"}),
                    html.Span("X2 Vernier BM", style={"fontSize": "0.72rem",
                                                       "color": THEME["text_muted"]}),
                ], style={"marginTop": "10px"}),
            ]),
        ], className="card-bio mb-4"))

    # ── 3. Sapiens score chart ──
    score_rows = []
    color_map = {}
    for _, row in result_df.iterrows():
        for pf, lab in [("H","VH"),("L","VL")]:
            for ver, stg, clr in [("0","Original",THEME["mouse"]),
                                   ("1","Humanized",THEME["human"]),
                                   ("2","Vernier BM",THEME["vernier"])]:
                col = f"{pf}{ver}_sapiens"
                if col in row and pd.notna(row.get(col)):
                    vname = f"{pf}{ver} ({stg})"
                    score_rows.append({"Antibody": row["name"], "Chain": lab,
                                       "Variant": vname, "Sapiens Score": row[col]})
                    color_map[vname] = clr

    if score_rows:
        scdf = pd.DataFrame(score_rows)
        scdf["Label"] = scdf["Antibody"] + " " + scdf["Chain"]
        fig = px.bar(scdf, x="Label", y="Sapiens Score", color="Variant",
                     barmode="group", color_discrete_map=color_map)
        fig.update_layout(template=PLOT_TEMPLATE, height=430,
                          yaxis_title="Mean Sapiens Score", xaxis_title="",
                          legend_title="", margin=dict(l=60, r=20, t=40, b=60))
        children.append(dbc.Card([
            make_card_header(3, "SAPIENS SCORE COMPARISON"),
            dbc.CardBody(dcc.Graph(figure=fig, config={"displayModeBar": False})),
        ], className="card-bio mb-4"))

    # ── 4. Mutation chart ──
    mut_rows = []
    mut_colors = {}
    for _, row in result_df.iterrows():
        for pf, lab in [("H","VH"),("L","VL")]:
            for ver, desc, clr in [("1","Humanized",THEME["human"]),
                                    ("2","Vernier BM",THEME["vernier"])]:
                col = f"{pf}{ver}_num_mutations"
                if col in row and pd.notna(row.get(col)):
                    vn = f"{pf}{ver} ({desc})"
                    mut_rows.append({"Antibody": row["name"], "Variant": vn,
                                     "Mutations": row[col]})
                    mut_colors[vn] = clr
    if mut_rows:
        mdf = pd.DataFrame(mut_rows)
        fig2 = px.bar(mdf, x="Antibody", y="Mutations", color="Variant",
                      barmode="group", color_discrete_map=mut_colors)
        fig2.update_layout(template=PLOT_TEMPLATE, height=380,
                           yaxis_title="# Mutations", xaxis_title="",
                           margin=dict(l=60, r=20, t=40, b=60))
        children.append(dbc.Card([
            make_card_header(4, "MUTATION COUNTS"),
            dbc.CardBody(dcc.Graph(figure=fig2, config={"displayModeBar": False})),
        ], className="card-bio mb-4"))

    # ── 5. Detail accordion ──
    show_alignment = len(result_df) <= 50
    details = []
    for _, row in result_df.iterrows():
        body_parts = []
        for pf, lab in [("H","VH"),("L","VL")]:
            oc = f"{pf}0_sequence"
            if oc not in row or pd.isna(row.get(oc)):
                continue
            orig_seq = str(row[oc])
            body_parts.append(html.H6([
                html.Span(lab, style={"color": THEME["human"], "fontFamily": "'JetBrains Mono', monospace"}),
                " Chain",
            ], className="mt-3 mb-2", style={"color": THEME["text"]}))

            variants = []
            for ver in ("1","2"):
                sc = f"{pf}{ver}_sequence"
                if sc in row and pd.notna(row.get(sc)):
                    desc = "Humanized" if ver == "1" else "Vernier back-mutated"
                    variants.append((f"{pf}{ver}", desc, str(row[sc])))

            for vid, desc, vseq in variants:
                mc = f"{vid}_mutations"
                if mc in row and pd.notna(row.get(mc)):
                    tag_color = THEME["human"] if vid.endswith("1") else THEME["vernier"]
                    body_parts.append(html.P([
                        html.Span(vid, style={"background": tag_color, "color": "white",
                                              "padding": "2px 8px", "borderRadius": "4px",
                                              "fontSize": "0.72rem", "fontWeight": "600",
                                              "fontFamily": "'JetBrains Mono', monospace",
                                              "marginRight": "8px"}),
                        html.Span(f"{desc}: ", style={"color": THEME["text_muted"],
                                                       "fontSize": "0.82rem"}),
                        html.Span(str(row[mc]), style={"color": THEME["text"],
                                                        "fontSize": "0.82rem",
                                                        "fontFamily": "'JetBrains Mono', monospace"}),
                    ], style={"marginBottom": "6px"}))

            vc = f"{pf}1_vernier_mutations"
            if vc in row and pd.notna(row.get(vc)):
                body_parts.append(html.P([
                    html.Span("\u26a0 Vernier in {p}1: ".format(p=pf),
                              style={"color": THEME["vernier"], "fontWeight": "600",
                                     "fontSize": "0.82rem"}),
                    html.Span(str(row[vc]), style={"color": THEME["text"],
                                                    "fontSize": "0.82rem",
                                                    "fontFamily": "'JetBrains Mono', monospace"}),
                ]))

            if show_alignment:
                annotations = None
                try:
                    ac = Chain(orig_seq, scheme=scheme, cdr_definition=cdr_def)
                    annotations = annotate_positions(ac)
                except Exception:
                    try:
                        ac = Chain(orig_seq, scheme=scheme)
                        annotations = annotate_positions(ac)
                    except Exception:
                        try:
                            ac = Chain(orig_seq, scheme="imgt")
                            annotations = annotate_positions(ac)
                        except Exception:
                            pass

                for vid, desc, vseq in variants:
                    if annotations and len(orig_seq) == len(vseq):
                        tag_color = THEME["human"] if vid.endswith("1") else THEME["vernier"]
                        body_parts.append(html.Div([
                            html.Span(f"{pf}0 vs {vid}", style={
                                "color": tag_color, "fontSize": "0.78rem",
                                "fontWeight": "600",
                                "fontFamily": "'JetBrains Mono', monospace"}),
                        ], className="mt-3 mb-2"))

                        # Legend
                        body_parts.append(html.Div([
                            html.Span(" CDR ", style={"background": THEME["cdr_yellow"],
                                "color": "#000", "padding": "1px 6px", "borderRadius": "3px",
                                "fontSize": "0.68rem", "marginRight": "6px", "fontWeight": "600"}),
                            html.Span(" VER ", style={"background": THEME["vernier"],
                                "color": "#fff", "padding": "1px 6px", "borderRadius": "3px",
                                "fontSize": "0.68rem", "marginRight": "6px", "fontWeight": "600"}),
                            html.Span(" FW ", style={"background": THEME["fw_dim"],
                                "color": "#94a3b8", "padding": "1px 6px", "borderRadius": "3px",
                                "fontSize": "0.68rem", "marginRight": "6px"}),
                            html.Span(" MUT ", style={"background": THEME["mutation_red"],
                                "color": "#fff", "padding": "1px 6px", "borderRadius": "3px",
                                "fontSize": "0.68rem"}),
                        ], className="mb-2"))

                        cs = {"display": "inline-block", "width": "14px",
                              "textAlign": "center", "fontFamily": "'JetBrains Mono', monospace",
                              "fontSize": "0.68rem", "lineHeight": "1.7"}
                        r_sp, o_sp, m_sp, h_sp = [], [], [], []
                        for i, ann in enumerate(annotations):
                            if i >= len(orig_seq) or i >= len(vseq):
                                break
                            reg = ann["region"]
                            oa, ha = orig_seq[i], vseq[i]
                            mut = oa != ha
                            bg = {
                                "CDR": THEME["cdr_yellow"],
                                "Vernier": THEME["vernier"],
                            }.get(reg, THEME["fw_dim"])
                            rc = {"CDR": "C", "Vernier": "V"}.get(reg, "\u00b7")
                            rc_color = "#000" if reg == "CDR" else ("#fff" if reg == "Vernier" else "#64748b")
                            mb = THEME["mutation_red"] if mut else "transparent"
                            mc = "#fff" if mut else THEME["text"]
                            r_sp.append(html.Span(rc, style={**cs, "backgroundColor": bg, "color": rc_color}))
                            o_sp.append(html.Span(oa, style={**cs, "backgroundColor": mb, "color": mc}))
                            h_sp.append(html.Span(ha, style={**cs, "backgroundColor": mb, "color": mc}))
                            m_sp.append(html.Span("*" if mut else "\u2502",
                                style={**cs, "color": THEME["mutation_red"] if mut else "#334155"}))

                        ls = {"fontFamily": "'JetBrains Mono', monospace",
                              "fontSize": "0.68rem", "display": "inline-block",
                              "width": "74px", "color": THEME["text_muted"], "fontWeight": "500"}
                        body_parts.append(html.Div([
                            html.Div([html.Span("Region  ", style=ls)] + r_sp),
                            html.Div([html.Span(f"{pf}0     ", style=ls)] + o_sp),
                            html.Div([html.Span("        ", style=ls)] + m_sp),
                            html.Div([html.Span(f"{vid}     ", style=ls)] + h_sp),
                        ], className="alignment-block"))
                    else:
                        if vseq:
                            body_parts.append(html.Details([
                                html.Summary(f"{vid} Sequence", style={
                                    "cursor": "pointer", "color": THEME["text_muted"],
                                    "fontSize": "0.82rem"}),
                                html.Code(vseq, style={"wordBreak": "break-all",
                                    "fontSize": "0.75rem", "color": THEME["human"]}),
                            ], className="mb-2"))
            else:
                for vid, desc, vseq in variants:
                    if vseq:
                        body_parts.append(html.Details([
                            html.Summary(f"{vid} ({desc})", style={
                                "cursor": "pointer", "color": THEME["text_muted"],
                                "fontSize": "0.82rem"}),
                            html.Code(vseq, style={"wordBreak": "break-all",
                                "fontSize": "0.75rem", "color": THEME["human"]}),
                        ], className="mb-2"))

            ec = f"{lab}_error"
            if ec in row and pd.notna(row.get(ec)):
                body_parts.append(html.Div(f"{lab} Error: {row[ec]}",
                    className="alert-danger-bio p-2 mt-1",
                    style={"fontSize": "0.8rem"}))

        details.append(dbc.AccordionItem(html.Div(body_parts), title=row["name"]))

    if details:
        hn = "" if show_alignment else " (alignment skipped for large batches)"
        children.append(dbc.Card([
            make_card_header(5, f"DETAILED RESULTS{hn}"),
            dbc.CardBody(dbc.Accordion(details, start_collapsed=True)),
        ], className="card-bio mb-4"))

    # ── 6. Downloads ──
    children.append(dbc.Card([
        make_card_header(6, "DOWNLOAD RESULTS"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Button(["\u2913 CSV"], id="dl-csv-btn",
                    className="btn-dl-csv w-100 text-white"), width=3),
                dbc.Col(dbc.Button(["\u2913 Excel (per-antibody sheets)"], id="dl-xlsx-btn",
                    className="btn-dl-xlsx w-100 text-white"), width=5),
                dbc.Col(dbc.Button(["\u2913 FASTA"], id="dl-fasta-btn",
                    className="btn-dl-fasta w-100 text-white"), width=4),
            ], className="g-2"),
            html.Div("Each antibody gets its own Excel sheet with H0/L0, H1/L1, H2/L2 variants.",
                     style={"color": THEME["text_muted"], "fontSize": "0.72rem",
                            "marginTop": "10px"}),
        ]),
    ], className="card-bio mb-5"))
    children.append(dcc.Download(id="download-csv"))
    children.append(dcc.Download(id="download-xlsx"))
    children.append(dcc.Download(id="download-fasta"))

    rate = len(result_df) / elapsed if elapsed > 0 else 0
    status = html.Div([
        html.Div([
            html.Span("\u2705 ", style={"marginRight": "4px"}),
            html.Span(f"Humanization complete \u2014 {len(result_df)} antibodies",
                      style={"color": THEME["human"], "fontWeight": "600"}),
        ]),
        html.Div(
            f"{scheme.upper()} \u00b7 CDR={cdr_def.upper()} \u00b7 "
            f"iter={iterations} \u00b7 {elapsed:.1f}s \u00b7 "
            f"{rate:.1f} ab/sec \u00b7 {n_workers}w \u00b7 "
            f"{'GPU' if GPU_AVAILABLE else 'CPU'}",
            style={"color": THEME["text_muted"], "fontSize": "0.72rem",
                   "fontFamily": "'JetBrains Mono', monospace", "marginTop": "4px"},
        ),
    ], className="alert-success-bio p-3")
    return result_df.to_json(date_format="iso", orient="split"), children, status


# ── Downloads ──

@app.callback(Output("download-csv","data"), Input("dl-csv-btn","n_clicks"),
              State("results-store","data"), prevent_initial_call=True)
def dl_csv(n, j):
    if j is None: return dash.no_update
    df = pd.read_json(io.StringIO(j), orient="split")
    return dcc.send_data_frame(df.to_csv, "humanized_results.csv", index=False)

@app.callback(Output("download-xlsx","data"), Input("dl-xlsx-btn","n_clicks"),
              State("results-store","data"), prevent_initial_call=True)
def dl_xlsx(n, j):
    if j is None: return dash.no_update
    df = pd.read_json(io.StringIO(j), orient="split")
    sheets = build_per_antibody_sheets(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        sc = [c for c in df.columns if not c.endswith("_sequence")]
        if sc: df[sc].to_excel(w, index=False, sheet_name="Summary")
        for sn, sd in sheets.items():
            sd.to_excel(w, index=False, sheet_name=sn)
            ws = w.sheets[sn]
            for ci, cn in enumerate(sd.columns):
                ml = max(sd[cn].astype(str).str.len().max(), len(cn))
                ws.set_column(ci, ci, 80 if cn == "Sequence" else min(ml+2, 30))
    return dcc.send_bytes(buf.getvalue(), "humanized_results.xlsx")

@app.callback(Output("download-fasta","data"), Input("dl-fasta-btn","n_clicks"),
              State("results-store","data"), prevent_initial_call=True)
def dl_fasta(n, j):
    if j is None: return dash.no_update
    df = pd.read_json(io.StringIO(j), orient="split")
    lines = []
    for _, row in df.iterrows():
        for pf in ("H","L"):
            for ver in ("0","1","2"):
                col = f"{pf}{ver}_sequence"
                if col in row and pd.notna(row.get(col)):
                    desc = {"0":"original","1":"humanized","2":"vernier_backmut"}[ver]
                    lines.append(f">{row['name']}_{pf}{ver}_{desc}")
                    lines.append(str(row[col]))
    return dict(content="\n".join(lines)+"\n", filename="humanized_sequences.fasta")


if __name__ == "__main__":
    print("\n  \U0001f9ec Sapiens Humanization Dashboard")
    print(f"  GPU: {'Yes (' + GPU_NAME + ')' if GPU_AVAILABLE else 'CPU'}")
    print(f"  Workers: {MAX_WORKERS}")
    print("  http://127.0.0.1:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)
