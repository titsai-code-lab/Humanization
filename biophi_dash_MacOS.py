#!/usr/bin/env python3
"""
Sapiens Antibody Humanization Dashboard
========================================
A Plotly Dash web application for batch antibody humanization using:
  - sapiens   : Deep learning humanization (Merck, pip install sapiens)
  - abnumber  : Antibody numbering & CDR identification (pip install abnumber)

Requirements:
    pip install dash dash-bootstrap-components plotly pandas sapiens abnumber \
                openpyxl xlsxwriter diskcache

Run:
    python biophi_dash_app.py
    Then open http://127.0.0.1:8050
"""

import base64
import io
import logging
import traceback
import time

import dash
import dash_bootstrap_components as dbc
import diskcache
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, DiskcacheManager, dash_table, dcc, html

import sapiens
from abnumber import Chain

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMBERING_SCHEMES = ["imgt", "chothia", "kabat"]
CDR_DEFINITIONS = ["imgt", "chothia", "kabat", "north"]

# Vernier zone positions (Foote & Winter, 1992, J. Mol. Biol. 224:487-499)
# Defined in Kabat numbering
VERNIER_KABAT = {
    "H": {"2", "27", "28", "29", "30", "47", "48", "49",
           "67", "69", "71", "73", "78", "93", "94", "103"},
    "L": {"2", "4", "35", "36", "46", "47", "48", "49",
           "64", "66", "68", "69", "71", "98"},
}

# ---------------------------------------------------------------------------
# Background callback manager (prevents browser timeout)
# ---------------------------------------------------------------------------
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# ---------------------------------------------------------------------------
# Pre-warm Sapiens models at startup
# ---------------------------------------------------------------------------
print("\n  Loading Sapiens models (first time downloads from HuggingFace)...")
try:
    _dummy_vh = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDKILWFGEPVFDYWGQGTLVTVSS"
    _dummy_vl = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
    sapiens.predict_scores(_dummy_vh, "H")
    sapiens.predict_scores(_dummy_vl, "L")
    print("  Models loaded successfully!\n")
except Exception as e:
    print(f"  Warning: Model pre-load failed ({e}). Will retry on first request.\n")


# ---------------------------------------------------------------------------
# Core humanization helpers
# ---------------------------------------------------------------------------

def get_cdr_positions(chain: Chain) -> set:
    """
    Return the set of 0-based sequence indices that fall inside CDR regions.
    Uses CDR sequence properties to find positions.
    """
    full_seq = chain.seq
    cdr_pos_set = set()

    # Get CDR sequences and find their positions in the full sequence
    # We search sequentially to handle repeated subsequences correctly
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


def annotate_positions(chain: Chain) -> list:
    """
    Annotate each position as CDR, Vernier, or Framework.
    Returns list of dicts with region label for each residue.
    """
    chain_type = chain.chain_type  # 'H' or 'L'
    full_seq = chain.seq
    vernier_set = VERNIER_KABAT.get(chain_type, set())

    # Step 1: Identify CDR positions using CDR seq properties
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

    # Step 2: Identify Vernier positions via Kabat numbering
    kabat_vernier_indices = set()
    try:
        kabat_chain = Chain(full_seq, scheme="kabat")
        for seq_idx, (pos, aa) in enumerate(kabat_chain):
            pos_str = str(pos)
            # Extract just the number (e.g. "H27" -> "27", "L35A" -> "35")
            num_part = ""
            for ch in pos_str:
                if ch.isdigit():
                    num_part += ch
                elif num_part:
                    break
            if num_part in vernier_set:
                kabat_vernier_indices.add(seq_idx)
    except Exception as e:
        logger.warning(f"Kabat numbering for Vernier failed: {e}")

    # Step 3: Build annotation list
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


def humanize_single_chain(
    sequence: str,
    chain_type: str,
    scheme: str,
    cdr_definition: str,
    humanize_cdrs: bool,
    iterations: int,
) -> dict:
    """
    Humanize one antibody chain using the Sapiens language model.

    Strategy:
      1. sapiens.predict_scores() → per-position probability matrix
      2. Pick the amino acid with the highest Sapiens score at each position
      3. If humanize_cdrs=False, keep original CDR residues
      4. Repeat for `iterations` rounds
    """
    result = {
        "original_seq": sequence,
        "humanized_seq": None,
        "mutations": None,
        "num_mutations": 0,
        "sapiens_score_orig": None,
        "sapiens_score_hum": None,
        "error": None,
    }

    # Number the sequence (abnumber auto-detects chain type)
    try:
        chain = Chain(sequence, scheme=scheme, cdr_definition=cdr_definition)
    except Exception as e:
        result["error"] = f"Numbering failed: {e}"
        return result

    detected_ct = chain.chain_type  # 'H' or 'L'

    # Get CDR positions (0-based indices)
    try:
        cdr_indices = get_cdr_positions(chain)
    except Exception:
        cdr_indices = set()

    current_seq = chain.seq

    # Original mean Sapiens score
    try:
        orig_scores = sapiens.predict_scores(current_seq, detected_ct)
        result["sapiens_score_orig"] = round(
            _calc_mean_sapiens_score(current_seq, orig_scores), 4
        )
    except Exception:
        pass

    # Iterative humanization
    for _ in range(iterations):
        try:
            scores_df = sapiens.predict_scores(current_seq, detected_ct)
        except Exception as e:
            result["error"] = f"Sapiens prediction failed: {e}"
            result["original_seq"] = chain.seq
            return result

        new_seq_list = list(current_seq)
        for i in range(len(current_seq)):
            if not humanize_cdrs and i in cdr_indices:
                continue
            best_aa = scores_df.iloc[i].idxmax()
            new_seq_list[i] = best_aa
        current_seq = "".join(new_seq_list)

    humanized_seq = current_seq

    # Humanized mean Sapiens score
    try:
        hum_scores = sapiens.predict_scores(humanized_seq, detected_ct)
        result["sapiens_score_hum"] = round(
            _calc_mean_sapiens_score(humanized_seq, hum_scores), 4
        )
    except Exception:
        pass

    # Collect mutations
    original = chain.seq
    mutations = []
    for i, (orig_aa, hum_aa) in enumerate(zip(original, humanized_seq)):
        if orig_aa != hum_aa:
            mutations.append(f"{orig_aa}{i+1}{hum_aa}")

    result["original_seq"] = original
    result["humanized_seq"] = humanized_seq
    result["mutations"] = "; ".join(mutations) if mutations else "None"
    result["num_mutations"] = len(mutations)
    return result


def _calc_mean_sapiens_score(sequence: str, scores_df: pd.DataFrame) -> float:
    """Mean probability the model assigns to each actual residue."""
    total = 0.0
    for i, aa in enumerate(sequence):
        if aa in scores_df.columns and i < len(scores_df):
            total += scores_df.iloc[i][aa]
    return total / len(sequence) if len(sequence) > 0 else 0.0


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_upload(contents: str, filename: str) -> pd.DataFrame:
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
        return pd.DataFrame({
            "name": raw[name_col],
            "heavy": raw[heavy_col] if heavy_col else None,
            "light": raw[light_col] if light_col else None,
        })

    if "sequence" in raw.columns:
        return pd.DataFrame({
            "name": raw[name_col],
            "heavy": raw["sequence"],
            "light": None,
        })

    raise ValueError(
        "Could not find sequence columns. Expected: "
        "heavy / VH / heavy_chain and optionally light / VL / light_chain"
    )


# ---------------------------------------------------------------------------
# Batch processing with progress
# ---------------------------------------------------------------------------

def run_batch(df: pd.DataFrame, scheme: str, cdr_def: str,
              humanize_cdrs: bool, iterations: int,
              set_progress=None) -> list:
    results = []
    # Count total chains to process
    total_chains = 0
    for _, row in df.iterrows():
        for col in ("heavy", "light"):
            seq = row.get(col)
            if pd.notna(seq) and seq is not None and str(seq).strip():
                total_chains += 1

    done = 0
    for idx, row in df.iterrows():
        rec = {"name": row["name"]}

        for label, col, ct in [("VH", "heavy", "H"), ("VL", "light", "L")]:
            seq = row.get(col)
            if pd.isna(seq) or seq is None or str(seq).strip() == "":
                continue
            seq = str(seq).strip()

            if set_progress:
                set_progress((
                    f"Processing {row['name']} {label} ... "
                    f"({done + 1}/{total_chains} chains)",
                    int(done / total_chains * 100),
                ))

            hum = humanize_single_chain(seq, ct, scheme, cdr_def, humanize_cdrs, iterations)
            rec[f"{label}_original"] = hum["original_seq"]
            rec[f"{label}_humanized"] = hum["humanized_seq"]
            rec[f"{label}_mutations"] = hum["mutations"]
            rec[f"{label}_num_mutations"] = hum["num_mutations"]
            rec[f"{label}_sapiens_orig"] = hum["sapiens_score_orig"]
            rec[f"{label}_sapiens_hum"] = hum["sapiens_score_hum"]
            rec[f"{label}_error"] = hum["error"]
            done += 1

        results.append(rec)

    if set_progress:
        set_progress(("Complete!", 100))

    return results


# ============================================================================
# DASH APPLICATION
# ============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Sapiens Humanization Dashboard",
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

SIDEBAR_STYLE = {
    "padding": "1.5rem",
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
    "height": "100%",
    "overflowY": "auto",
}

sidebar = html.Div([
    html.H5("Parameters", className="mb-3"),
    dbc.Label("Numbering Scheme"),
    dcc.Dropdown(
        id="scheme-dropdown",
        options=[{"label": s.upper(), "value": s} for s in NUMBERING_SCHEMES],
        value="imgt", clearable=False,
    ),
    html.Br(),
    dbc.Label("CDR Definition"),
    dcc.Dropdown(
        id="cdr-dropdown",
        options=[{"label": s.upper(), "value": s} for s in CDR_DEFINITIONS],
        value="imgt", clearable=False,
    ),
    html.Br(),
    dbc.Label("Sapiens Iterations"),
    dcc.Slider(
        id="iterations-slider", min=1, max=5, step=1, value=1,
        marks={i: str(i) for i in range(1, 6)},
    ),
    html.Br(),
    dbc.Checklist(
        id="humanize-cdrs-check",
        options=[{"label": " Humanize CDRs", "value": "yes"}],
        value=[], switch=True,
    ),
    html.Hr(),
    dbc.Button("Run Humanization", id="run-btn", color="primary",
               className="w-100", size="lg", disabled=True),
    html.Div(id="cancel-btn-container", className="mt-2"),
], style=SIDEBAR_STYLE)

upload_area = dcc.Upload(
    id="upload-data",
    children=html.Div([
        html.Br(),
        html.Span("Drag & Drop or ", style={"fontSize": "1rem"}),
        html.A("Browse", style={"fontWeight": "bold", "color": "#2c3e50",
                                "cursor": "pointer"}),
        html.Br(),
        html.Small("Accepts .xlsx, .xls, .csv, .tsv", className="text-muted"),
        html.Br(), html.Br(),
    ], style={"textAlign": "center"}),
    style={
        "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "12px",
        "borderColor": "#adb5bd", "padding": "40px 20px", "cursor": "pointer",
        "backgroundColor": "#fdfdfe",
    },
    multiple=False,
)

app.layout = dbc.Container(fluid=True, className="px-0", children=[
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Sapiens Antibody Humanization Dashboard",
                            className="fw-bold"),
        ], fluid=True),
        color="primary", dark=True, className="mb-0",
    ),
    dbc.Row([
        dbc.Col(sidebar, width=3, className="p-0",
                style={"minHeight": "calc(100vh - 56px)"}),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("1. Upload Antibody Sequences",
                                       className="mb-0")),
                dbc.CardBody([
                    upload_area,
                    html.Div(id="upload-status", className="mt-3"),
                    html.Div(id="preview-table", className="mt-3"),
                ]),
            ], className="mb-4 mt-3 shadow-sm"),

            # Progress bar (shown during long callback)
            html.Div(id="progress-container", children=[
                html.Div(id="progress-status-text", className="mb-1 fw-bold"),
                dbc.Progress(id="progress-bar", value=0, striped=True,
                             animated=True, className="mb-3",
                             style={"height": "24px"}),
            ], style={"display": "none"}),

            html.Div(id="run-status", className="mb-3"),
            html.Div(id="results-container"),

            dcc.Store(id="parsed-data-store"),
            dcc.Store(id="results-store"),
        ], width=9, className="px-4"),
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
        return None, dbc.Alert(f"Error parsing file: {e}", color="danger"), "", True

    n = len(df)
    has_heavy = df["heavy"].notna().sum()
    has_light = df["light"].notna().sum() if "light" in df.columns else 0

    status = dbc.Alert([
        html.Strong(f"{filename}"),
        f" — {n} antibodies loaded ({has_heavy} heavy, {has_light} light chains)",
    ], color="success")

    preview = df.head(5).copy()
    for col in ("heavy", "light"):
        if col in preview.columns:
            preview[col] = preview[col].apply(
                lambda x: (str(x)[:40] + "...") if pd.notna(x) and len(str(x)) > 40 else x
            )

    table = dash_table.DataTable(
        data=preview.to_dict("records"),
        columns=[{"name": c.upper(), "id": c} for c in preview.columns],
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
        style_cell={"textAlign": "left", "padding": "6px 12px", "fontSize": "0.85rem"},
        page_size=5,
    )
    return df.to_json(date_format="iso", orient="split"), status, table, False


# ── This is the LONG callback: runs in the background, won't time out ──
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
    prevent_initial_call=True,
    background=True,
    running=[
        (Output("run-btn", "disabled"), True, False),
        (Output("run-btn", "children"), "Running...", "Run Humanization"),
        (Output("progress-container", "style"), {"display": "block"}, {"display": "none"}),
    ],
    progress=[
        Output("progress-status-text", "children"),
        Output("progress-bar", "value"),
    ],
)
def run_humanization(set_progress, n_clicks, json_data, scheme, cdr_def,
                     iterations, humanize_cdrs_val):
    if json_data is None:
        return dash.no_update, dash.no_update, dbc.Alert("No data loaded.", color="warning")

    df = pd.read_json(io.StringIO(json_data), orient="split")
    humanize_cdrs = "yes" in (humanize_cdrs_val or [])

    set_progress(("Initializing Sapiens model...", 0))

    try:
        results = run_batch(df, scheme, cdr_def, humanize_cdrs, iterations,
                            set_progress=set_progress)
    except Exception as e:
        logger.error(traceback.format_exc())
        return None, "", dbc.Alert(f"Humanization failed: {e}", color="danger")

    result_df = pd.DataFrame(results)
    children = []

    # ── 2. Summary table ──
    summary_cols = ["name"]
    optional = [
        "VH_num_mutations", "VH_sapiens_orig", "VH_sapiens_hum",
        "VL_num_mutations", "VL_sapiens_orig", "VL_sapiens_hum",
    ]
    summary_cols += [c for c in optional if c in result_df.columns]

    display_names = {
        "name": "Antibody",
        "VH_num_mutations": "VH Mutations",
        "VH_sapiens_orig": "VH Sapiens (orig)",
        "VH_sapiens_hum": "VH Sapiens (hum)",
        "VL_num_mutations": "VL Mutations",
        "VL_sapiens_orig": "VL Sapiens (orig)",
        "VL_sapiens_hum": "VL Sapiens (hum)",
    }

    children.append(
        dbc.Card([
            dbc.CardHeader(html.H5("2. Humanization Summary", className="mb-0")),
            dbc.CardBody(
                dash_table.DataTable(
                    data=result_df[summary_cols].to_dict("records"),
                    columns=[{"name": display_names.get(c, c), "id": c}
                             for c in summary_cols],
                    style_table={"overflowX": "auto"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                    style_cell={"textAlign": "center", "padding": "8px",
                                "fontSize": "0.85rem"},
                    page_size=20,
                )
            ),
        ], className="mb-4 shadow-sm")
    )

    # ── 3. Sapiens score comparison chart ──
    score_rows = []
    for _, row in result_df.iterrows():
        name = row["name"]
        for label in ("VH", "VL"):
            orig_col = f"{label}_sapiens_orig"
            hum_col = f"{label}_sapiens_hum"
            if orig_col in row and pd.notna(row.get(orig_col)):
                score_rows.append({"Antibody": name, "Chain": label,
                                   "Stage": "Original",
                                   "Sapiens Score": row[orig_col]})
            if hum_col in row and pd.notna(row.get(hum_col)):
                score_rows.append({"Antibody": name, "Chain": label,
                                   "Stage": "Humanized",
                                   "Sapiens Score": row[hum_col]})

    if score_rows:
        score_df = pd.DataFrame(score_rows)
        score_df["Label"] = score_df["Antibody"] + " " + score_df["Chain"]
        fig_bar = px.bar(
            score_df, x="Label", y="Sapiens Score", color="Stage",
            barmode="group",
            color_discrete_map={"Original": "#e74c3c", "Humanized": "#27ae60"},
            title="Mean Sapiens Score: Original vs Humanized",
        )
        fig_bar.update_layout(
            yaxis_title="Mean Sapiens Score", xaxis_title="", legend_title="",
            template="plotly_white", height=420,
        )
        children.append(
            dbc.Card([
                dbc.CardHeader(html.H5("3. Sapiens Score Comparison",
                                       className="mb-0")),
                dbc.CardBody(dcc.Graph(figure=fig_bar)),
            ], className="mb-4 shadow-sm")
        )

    # ── 4. Mutation count chart ──
    mut_rows = []
    for _, row in result_df.iterrows():
        for label in ("VH", "VL"):
            col = f"{label}_num_mutations"
            if col in row and pd.notna(row.get(col)):
                mut_rows.append({"Antibody": row["name"], "Chain": label,
                                 "Mutations": row[col]})

    if mut_rows:
        mut_df = pd.DataFrame(mut_rows)
        fig_mut = px.bar(
            mut_df, x="Antibody", y="Mutations", color="Chain",
            barmode="group",
            color_discrete_map={"VH": "#3498db", "VL": "#e67e22"},
            title="Number of Mutations Introduced",
        )
        fig_mut.update_layout(template="plotly_white", height=380,
                              yaxis_title="# Mutations", xaxis_title="")
        children.append(
            dbc.Card([
                dbc.CardHeader(html.H5("4. Mutation Counts", className="mb-0")),
                dbc.CardBody(dcc.Graph(figure=fig_mut)),
            ], className="mb-4 shadow-sm")
        )

    # ── 5. Detail accordion with Vernier-annotated alignment ──
    details = []
    for _, row in result_df.iterrows():
        body_parts = []
        for label in ("VH", "VL"):
            orig_col = f"{label}_original"
            hum_col = f"{label}_humanized"

            if orig_col not in row or pd.isna(row.get(orig_col)):
                continue

            orig_seq = str(row[orig_col])
            hum_seq = str(row.get(hum_col, "")) if pd.notna(row.get(hum_col)) else ""

            body_parts.append(html.H6(f"{label} Chain", className="mt-3 mb-2"))

            # Mutation summary
            if f"{label}_mutations" in row and pd.notna(row.get(f"{label}_mutations")):
                body_parts.append(html.P([
                    html.Strong("Mutations: "),
                    str(row[f"{label}_mutations"]),
                ], style={"fontSize": "0.85rem"}))

            # Compute annotations on-the-fly from original sequence
            annotations = None
            ann_err_msg = ""
            try:
                ann_chain = Chain(orig_seq, scheme=scheme, cdr_definition=cdr_def)
                annotations = annotate_positions(ann_chain)
            except Exception as ann_err:
                ann_err_msg = str(ann_err)
                logger.warning(f"Annotation attempt 1 failed for {row['name']} {label}: {ann_err}")
                # Fallback: try without cdr_definition
                try:
                    ann_chain = Chain(orig_seq, scheme=scheme)
                    annotations = annotate_positions(ann_chain)
                    ann_err_msg = ""
                except Exception as ann_err2:
                    ann_err_msg = str(ann_err2)
                    logger.warning(f"Annotation attempt 2 failed: {ann_err2}")
                    # Fallback: try with just imgt
                    try:
                        ann_chain = Chain(orig_seq, scheme="imgt")
                        annotations = annotate_positions(ann_chain)
                        ann_err_msg = ""
                    except Exception as ann_err3:
                        ann_err_msg = str(ann_err3)
                        logger.warning(f"Annotation attempt 3 failed: {ann_err3}")

            # Build color-coded alignment
            if annotations and hum_seq and len(orig_seq) == len(hum_seq):
                # Color legend
                body_parts.append(html.Div([
                    html.Span("Legend: ", style={"fontWeight": "bold", "fontSize": "0.8rem"}),
                    html.Span(" CDR ", style={
                        "backgroundColor": "#ffeb3b", "padding": "1px 6px",
                        "borderRadius": "3px", "fontSize": "0.75rem", "marginRight": "6px"}),
                    html.Span(" Vernier ", style={
                        "backgroundColor": "#ce93d8", "padding": "1px 6px",
                        "borderRadius": "3px", "fontSize": "0.75rem", "marginRight": "6px"}),
                    html.Span(" Framework ", style={
                        "backgroundColor": "#e0e0e0", "padding": "1px 6px",
                        "borderRadius": "3px", "fontSize": "0.75rem", "marginRight": "6px"}),
                    html.Span(" Mutated ", style={
                        "backgroundColor": "#ef5350", "color": "white",
                        "padding": "1px 6px", "borderRadius": "3px",
                        "fontSize": "0.75rem"}),
                ], className="mb-2"))

                # Build alignment rows
                region_spans = []
                orig_spans = []
                hum_spans = []
                match_spans = []

                char_style = {
                    "display": "inline-block",
                    "width": "14px",
                    "textAlign": "center",
                    "fontFamily": "monospace",
                    "fontSize": "0.7rem",
                    "lineHeight": "1.6",
                }

                for i, ann in enumerate(annotations):
                    if i >= len(orig_seq) or i >= len(hum_seq):
                        break

                    region = ann["region"]
                    o_aa = orig_seq[i]
                    h_aa = hum_seq[i]
                    is_mutated = o_aa != h_aa

                    if region == "CDR":
                        bg = "#ffeb3b"
                        region_char = "C"
                    elif region == "Vernier":
                        bg = "#ce93d8"
                        region_char = "V"
                    else:
                        bg = "#e0e0e0"
                        region_char = "."

                    mut_bg = "#ef5350" if is_mutated else "transparent"
                    mut_color = "white" if is_mutated else "#333"

                    region_spans.append(html.Span(
                        region_char,
                        style={**char_style, "backgroundColor": bg, "color": "#555"},
                    ))
                    orig_spans.append(html.Span(
                        o_aa,
                        style={**char_style, "backgroundColor": mut_bg, "color": mut_color},
                    ))
                    hum_spans.append(html.Span(
                        h_aa,
                        style={**char_style, "backgroundColor": mut_bg, "color": mut_color},
                    ))
                    match_spans.append(html.Span(
                        "*" if is_mutated else "|",
                        style={**char_style,
                               "color": "#ef5350" if is_mutated else "#aaa"},
                    ))

                lbl_style = {"fontFamily": "monospace", "fontSize": "0.7rem",
                             "display": "inline-block", "width": "80px",
                             "fontWeight": "bold", "color": "#666"}

                alignment_div = html.Div([
                    html.Div([html.Span("Region:   ", style=lbl_style)] + region_spans),
                    html.Div([html.Span("Original: ", style=lbl_style)] + orig_spans),
                    html.Div([html.Span("Match:    ", style=lbl_style)] + match_spans),
                    html.Div([html.Span("Humanized:", style=lbl_style)] + hum_spans),
                ], style={"overflowX": "auto", "whiteSpace": "nowrap",
                          "padding": "10px", "backgroundColor": "#fafafa",
                          "borderRadius": "6px", "border": "1px solid #eee"})

                body_parts.append(alignment_div)

                # Vernier mutation summary
                vernier_muts = []
                for i, ann in enumerate(annotations):
                    if i < len(orig_seq) and i < len(hum_seq):
                        if ann["region"] == "Vernier" and orig_seq[i] != hum_seq[i]:
                            vernier_muts.append(f"{orig_seq[i]}{i+1}{hum_seq[i]}")
                if vernier_muts:
                    body_parts.append(html.P([
                        html.Strong("Vernier zone mutations: ",
                                    style={"color": "#7b1fa2"}),
                        ", ".join(vernier_muts),
                    ], style={"fontSize": "0.85rem", "marginTop": "6px"}))
                else:
                    body_parts.append(html.P([
                        html.Strong("Vernier zone: ",
                                    style={"color": "#7b1fa2"}),
                        "All preserved (no mutations)",
                    ], style={"fontSize": "0.85rem", "marginTop": "6px"}))

            else:
                # Fallback: show raw sequences if annotation failed
                if hum_seq:
                    body_parts.append(html.Details([
                        html.Summary(f"{label} Humanized Sequence",
                                     style={"cursor": "pointer"}),
                        html.Code(hum_seq,
                                  style={"wordBreak": "break-all", "fontSize": "0.8rem"}),
                    ], className="mb-2"))
                if not annotations:
                    body_parts.append(
                        dbc.Alert(f"Vernier annotation unavailable: {ann_err_msg}",
                                  color="info", className="py-1 mt-1"))

            if f"{label}_error" in row and pd.notna(row.get(f"{label}_error")):
                body_parts.append(
                    dbc.Alert(f"{label} Error: {row[f'{label}_error']}",
                              color="warning", className="py-1")
                )
        details.append(dbc.AccordionItem(html.Div(body_parts), title=row["name"]))

    if details:
        children.append(
            dbc.Card([
                dbc.CardHeader(html.H5("5. Detailed Results", className="mb-0")),
                dbc.CardBody(dbc.Accordion(details, start_collapsed=True)),
            ], className="mb-4 shadow-sm")
        )

    # ── 6. Downloads ──
    children.append(
        dbc.Card([
            dbc.CardHeader(html.H5("6. Download Results", className="mb-0")),
            dbc.CardBody(
                dbc.Row([
                    dbc.Col(dbc.Button("Download CSV", id="dl-csv-btn",
                                       color="success", className="w-100"), width=4),
                    dbc.Col(dbc.Button("Download Excel", id="dl-xlsx-btn",
                                       color="info", className="w-100"), width=4),
                    dbc.Col(dbc.Button("Download FASTA", id="dl-fasta-btn",
                                       color="secondary", className="w-100"), width=4),
                ], className="g-2"),
            ),
        ], className="mb-5 shadow-sm")
    )
    children.append(dcc.Download(id="download-csv"))
    children.append(dcc.Download(id="download-xlsx"))
    children.append(dcc.Download(id="download-fasta"))

    status = dbc.Alert(
        f"Humanization complete for {len(result_df)} antibodies "
        f"(scheme={scheme.upper()}, CDR={cdr_def.upper()}, iter={iterations})",
        color="success",
    )
    return result_df.to_json(date_format="iso", orient="split"), children, status


# ── Download callbacks ──

@app.callback(
    Output("download-csv", "data"),
    Input("dl-csv-btn", "n_clicks"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def download_csv(n, json_data):
    if json_data is None:
        return dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient="split")
    return dcc.send_data_frame(df.to_csv, "humanized_results.csv", index=False)


@app.callback(
    Output("download-xlsx", "data"),
    Input("dl-xlsx-btn", "n_clicks"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def download_xlsx(n, json_data):
    if json_data is None:
        return dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient="split")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Humanized")
    return dcc.send_bytes(buf.getvalue(), "humanized_results.xlsx")


@app.callback(
    Output("download-fasta", "data"),
    Input("dl-fasta-btn", "n_clicks"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def download_fasta(n, json_data):
    if json_data is None:
        return dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient="split")
    lines = []
    for _, row in df.iterrows():
        for label in ("VH", "VL"):
            col = f"{label}_humanized"
            if col in row and pd.notna(row.get(col)):
                lines.append(f">{row['name']}_{label}_humanized")
                lines.append(str(row[col]))
    return dict(content="\n".join(lines) + "\n", filename="humanized_sequences.fasta")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("  Sapiens Humanization Dashboard")
    print("  ──────────────────────────────")
    print("  Open http://127.0.0.1:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)

