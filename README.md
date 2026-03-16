# Sapiens Antibody Humanization Dashboard

A Plotly Dash web application for batch antibody humanization using **Sapiens** (deep learning) with interactive visualization, **Vernier zone** annotation, and downloadable results.

Built on top of:
- [Sapiens](https://github.com/Merck/Sapiens) — BERT-based antibody language model for humanization (Merck)
- [AbNumber](https://github.com/prihoda/AbNumber) — Antibody numbering using ANARCI
- [BioPhi](https://github.com/Merck/BioPhi) — Platform reference for Sapiens and OASis

> Prihoda et al. (2022) *BioPhi: A platform for antibody design, humanization, and humanness evaluation based on natural antibody repertoires and deep learning*, mAbs 14:1, [DOI: 10.1080/19420862.2021.2020203](https://doi.org/10.1080/19420862.2021.2020203)

---

## Features

- **File Upload** — Drag-and-drop `.xlsx`, `.xls`, `.csv`, or `.tsv` files with heavy/light chain sequences
- **Sapiens Humanization** — Deep learning-based framework humanization with configurable iterations
- **Numbering Schemes** — IMGT, Chothia, or Kabat numbering with matching CDR definitions (IMGT, Chothia, Kabat, North)
- **Vernier Zone Annotation** — Color-coded alignment view highlighting CDR, Vernier, and Framework regions per Foote & Winter (1992)
- **Sapiens Score Comparison** — Bar chart comparing mean Sapiens scores before and after humanization
- **Mutation Tracking** — Per-residue mutation list, mutation count charts, and Vernier-specific mutation summary
- **Background Processing** — Long-running jobs run in the background with a live progress bar (no browser timeout)
- **Export** — Download results as CSV, Excel, or FASTA

---

## Screenshots

After uploading sequences and running humanization, the dashboard shows:

1. **Summary Table** — Mutation counts and Sapiens scores for each antibody
2. **Sapiens Score Chart** — Original vs. humanized scores
3. **Mutation Count Chart** — Per-chain mutation counts
4. **Detailed Alignment** — Color-coded residue-level view with Vernier zones
5. **Download Buttons** — CSV, Excel, and FASTA export

---

## Installation

### Prerequisites

- **macOS (Apple Silicon)** or **Linux** — Windows is not supported by AbNumber/ANARCI
- **Miniforge / Conda** — [Install Miniforge](https://github.com/conda-forge/miniforge) (recommended for Apple Silicon)
- **Python 3.10–3.11** recommended

### Step-by-step setup

```bash
# 1. Create a fresh conda environment
conda create -n biophi python=3.11 -y
conda activate biophi

# 2. Install PyTorch (required by Sapiens)
#    Apple Silicon (M1/M2/M3):
pip install torch torchvision
#    Linux (CPU only):
#    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install ANARCI and HMMER (required by AbNumber for antibody numbering)
conda install -c bioconda anarci hmmer -y

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "
import torch; print('PyTorch', torch.__version__)
import sapiens; print('Sapiens OK')
import abnumber; print('AbNumber OK')
import dash; print('Dash OK')
print('All dependencies installed successfully!')
"
```

### Verify PyTorch version

Sapiens requires **PyTorch >= 2.4**. Check with:

```bash
python -c "import torch; print(torch.__version__)"
```

If you see a version below 2.4, your Python may be running under Rosetta (x86) instead of native ARM64. See [Troubleshooting](#troubleshooting) below.

---

## Usage

### 1. Start the dashboard

```bash
conda activate biophi
python biophi_dash_app.py
```

On first launch, Sapiens will download model weights from HuggingFace (~100MB). Subsequent launches are instant.

Open **http://127.0.0.1:8050** in your browser.

### 2. Prepare your input file

Create a CSV or Excel file with columns:

| name | heavy | light |
|------|-------|-------|
| Trastuzumab | EVQLVES... | DIQMTQ... |
| Pembrolizumab | QVQLVQ... | EIVLTQ... |

Accepted column names (case-insensitive):
- **Name**: `name`, `id`, `antibody_name`, `antibody_id`
- **Heavy chain**: `heavy`, `VH`, `heavy_chain`, `heavy_sequence`
- **Light chain**: `light`, `VL`, `light_chain`, `light_sequence`

A sample file (`sample_antibodies.csv`) is included.

### 3. Configure parameters

In the sidebar:
- **Numbering Scheme** — IMGT (default), Chothia, or Kabat
- **CDR Definition** — IMGT (default), Chothia, Kabat, or North
- **Sapiens Iterations** — 1 (conservative) to 5 (aggressive). 3 is recommended for thorough humanization.
- **Humanize CDRs** — Off by default (preserves parental CDRs). Turn on to also mutate CDR residues.

### 4. Run and review

Click **Run Humanization** and monitor the progress bar. Results appear in six sections:

1. **Summary Table** — Overview with mutation counts and Sapiens scores
2. **Sapiens Score Comparison** — Bar chart of humanness improvement
3. **Mutation Counts** — Per-antibody, per-chain mutation counts
4. **Detailed Results** — Expand each antibody to see:
   - Color-coded alignment (CDR = yellow, Vernier = purple, Framework = gray, Mutated = red)
   - Vernier zone mutation summary
5. **Download** — Export as CSV, Excel (.xlsx), or FASTA

---

## Vernier Zone Annotation

Vernier zone residues are framework positions that structurally support CDR loop conformations (Foote & Winter, 1992). Mutations in these positions during humanization may affect binding affinity.

The dashboard highlights Vernier positions using the canonical Kabat-numbered positions:

- **Heavy chain (16 positions)**: H2, H27, H28, H29, H30, H47, H48, H49, H67, H69, H71, H73, H78, H93, H94, H103
- **Light chain (14 positions)**: L2, L4, L35, L36, L46, L47, L48, L49, L64, L66, L68, L69, L71, L98

Regardless of which numbering scheme you select for humanization, Vernier detection always uses Kabat internally (the positions are defined in Kabat numbering).

> Foote, J. & Winter, G. (1992) Antibody framework residues affecting the conformation of the hypervariable loops. *J. Mol. Biol.* 224:487–499.

---

## How Sapiens Humanization Works

1. The input sequence is fed to the Sapiens BERT model
2. The model outputs a probability matrix: for each position, the probability of each of the 20 amino acids being "human-like"
3. At each framework position, the amino acid with the highest probability replaces the original
4. CDR residues are preserved (unless "Humanize CDRs" is enabled)
5. This process repeats for the configured number of iterations

More iterations produce more humanized sequences at the cost of more mutations from the parental sequence.

---

## Project Structure

```
├── biophi_dash_app.py      # Main Dash application
├── requirements.txt         # Python dependencies
├── sample_antibodies.csv    # Example input file
├── README.md                # This file
└── cache/                   # Auto-created for background job queue
```

---

## Troubleshooting

### PyTorch stuck at 2.2.2

If `pip install torch` keeps installing 2.2.2, your Python is running as x86 (Intel) via Rosetta instead of native ARM64. Verify with:

```bash
python -c "import platform; print(platform.machine())"
```

If it prints `x86_64` on an Apple Silicon Mac, you need native ARM64 Python:

```bash
brew install miniforge
conda init zsh
# Close and reopen terminal
conda create -n biophi python=3.11 -y
conda activate biophi
python -c "import platform; print(platform.machine())"  # Should print: arm64
```

### `No module named 'anarci'`

AbNumber requires ANARCI for antibody numbering:

```bash
conda install -c bioconda anarci hmmer -y
```

### `ModuleNotFoundError: No module named 'biophi.humanization.sapiens'`

This app uses the standalone `sapiens` pip package, **not** the `biophi` conda package. They have different APIs. Make sure you installed `sapiens` (not `biophi`) via pip.

### Callback timeout / server not responding

The app uses background callbacks via `diskcache` to prevent browser timeouts. If you still see timeouts, ensure `diskcache` is installed:

```bash
pip install diskcache
```

### Vernier annotation unavailable

If Vernier zones don't appear, check that ANARCI is working:

```bash
python -c "from abnumber import Chain; c = Chain('EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDKILWFGEPVFDYWGQGTLVTVSS', scheme='kabat'); print(c.seq[:20])"
```

---

## License

This dashboard is provided as-is for research use. Sapiens and BioPhi are open-source under the MIT license by Merck.

---

## References

1. Prihoda, D. et al. (2022) BioPhi: A platform for antibody design, humanization, and humanness evaluation. *mAbs* 14:1. [DOI: 10.1080/19420862.2021.2020203](https://doi.org/10.1080/19420862.2021.2020203)

2. Foote, J. & Winter, G. (1992) Antibody framework residues affecting the conformation of the hypervariable loops. *J. Mol. Biol.* 224:487–499.

3. Dunbar, J. & Deane, C.M. (2016) ANARCI: Antigen receptor numbering and receptor classification. *Bioinformatics* 32(2):298–300.
