import streamlit as st
import requests
import yaml
import pandas as pd
import subprocess
import urllib.parse
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import html

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LabChronicle | Lab Database",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS STYLES ─────────────────────────────────────────────────────────────
# Use a dedicated function for CSS to keep the main script clean
def apply_custom_css():
    """Applies custom CSS to the Streamlit app."""
    st.markdown("""
    <style>
    /* Gradient Header */
    .gradient-header {
        background: linear-gradient(90deg, #2159a6, #23a3b5);
        padding: 18px 0 10px 18px;
        border-radius: 12px;
        margin-bottom: 28px;
    }

    .gradient-header h1 {
        color: #fff;
        margin-bottom: 0;
    }

    .gradient-header span {
        color: #e0ecff;
        font-size: 1.1em;
    }

    /* Data Tables */
    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0 8px; /* Adds vertical spacing between rows */
    }

    .stDataFrame th, .stDataFrame td {
        text-align: center !important;
        padding: 8px 12px; /* Add padding for better spacing */
    }

    .stDataFrame th {
        background-color: #f0f2f6; /* Lighter background for header */
        color: #333;
        font-weight: bold;
    }

    .stDataFrame tr:hover {
        background-color: #e6f7ff; /* Highlight rows on hover */
    }

    /* Dropdowns & Text Inputs */
    div[data-baseweb="select"] > div:first-child,
    input[data-baseweb="input"],
    input[type="text"], input[type="password"], textarea {
        border: 2px solid #ccc !important;
        border-radius: 1px !important;
        background-color: #fff !important;
        padding: 1rem !important;
        color: #333 !important; /* 
    }

/* center everything in data tables */
table th, table td {
    text-align: center !important;
}

    /* Full Detail Card */
    .detail-card {
        background: #fafafa;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Subtle shadow for a card effect */
    }

    .detail-card h3 {
        color: #2159a6;
        border-bottom: 2px solid #2159a6;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    .detail-card dl {
        display: grid;
        grid-template-columns: 250px 1fr; /* Widen the key column */
        gap: 1rem;
        align-items: start;
        margin: 0;
    }

    .detail-card dt {
        font-weight: 600;
        color: #555;
    }

    .detail-card dd {
        margin: 0;
        padding: 0;
        line-height: 1.6;
        word-wrap: break-word; /* Prevents long URLs from overflowing */
    }

    </style>
    """, unsafe_allow_html=True)

# Apply the CSS function
apply_custom_css()

# ─── HEADER MARKDOWN ────────────────────────────────────────────────────────
st.markdown("""
<div class="gradient-header">
  <h1>
    LabChronicle <span style="font-size:0.7em;font-weight:300;">| Lab Database</span>
  </h1>
  <span style="color:#e0ecff;font-size:1.1em;">
    🔬 All your lab reagents, in one place
  </span>
</div>
""", unsafe_allow_html=True)


# ─── CONFIGS ────────────────────────────────────────────────────────────────
field_config = {
    "antibodies": [
        "antibody_target", "antibody_name", "location", "clonality",
        "host_species", "catalog_id", "source", "applications",
        "notes", "website", "status"
    ],
    "cell_lines": [
        "cell_line_name", "organism", "tissue", "type", "derived_from",
        "resistance", "modifications", "notes", "source", "catalog_id",
        "publication", "authentication"
    ],
    "plasmids": [
        "plasmid_name", "backbone", "promoter", "tag", "origin", "catalog_id", "website",
        "expression_host", "sequencing_validation", "location", "notes", "status", "MTA", "publications"
    ],
    "animal_models": [
        "model_name", "organism", "background", "strain", "genotype",
        "source", "catalogue_number", "website", "sex", "notes"
    ],
}

def prettify_col(col: str) -> str:
    manual = {
        "antibody_target": "Antibody Target",
        "antibody_name": "Antibody Name",
        "host_species": "Host Species",
        "catalog_id": "Catalog ID",
        "cell_line_name": "Cell Line Name",
        "derived_from": "Derived From",
        "source_file": "YAML File",
        "selection_marker": "Selection Marker",
        "amplification_host": "Amplification Host",
        "sequencing_validation": "Sequencing Validated",
        "expression_host": "Expression Host",
        "publications": "Publication DOI(s)",
        "publication": "Publication DOI",
        "model_name": "Model Name",
        "catalogue_number": "Catalogue #",
    }
    return manual.get(col, col.replace("_", " ").title())

prettified_field_config = {
    k: [prettify_col(c) for c in cols]
    for k, cols in field_config.items()
}

def pretty_val(v: Any) -> Any:
    if isinstance(v, dict):
        return "; ".join(f"**{k}**: {v[k]}" for k in v)
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return v

def make_all_links_clickable(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if any(k in col.lower() for k in ("website", "url", "link", "publication")):
            df2[col] = df2[col].apply(
                lambda v: (
                    f'<a href="{v}" target="_blank" '
                    'style="text-decoration:none; color:#2159a6; font-weight:bold;">'
                    'Link</a>'
                ) if isinstance(v, str) and v.startswith("http") else v
            )
    return df2


# ─── YAML → DataFrame LOADER ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(dataset: str, configs_path: Path) -> pd.DataFrame:
    folder = configs_path / dataset
    if not folder.exists():
        return pd.DataFrame()

    records = []
    for file in folder.glob("*.yaml"):
        raw = yaml.safe_load(file.read_text()) or {}
        if dataset == "antibodies":
            tgt = raw.get("antibody_target","")
            for entry in raw.get("antibodies", []):
                rec = entry.copy()
                rec["antibody_target"] = tgt
                rec["source_file"] = file.name
                records.append(rec)
        else:
            entries = (
                raw.get("models", []) if dataset=="animal_models"
                else raw if isinstance(raw, list)
                else raw.get("plasmids", raw)
            )
            for entry in entries:
                rec = {}
                if dataset=="cell_lines":
                    rec = flatten_cell_line(raw, file.name)
                elif dataset=="plasmids":
                    rec = flatten_plasmid(entry, file.name)
                elif dataset=="animal_models":
                    rec = flatten_animal_model({**entry, "organism": raw.get("organism",file.stem)}, file.name)
                records.append(rec)
    return pd.DataFrame(records)

# ─── FLATTEN HELPERS ─────────────────────────────────────────────────────────
def flatten_cell_line(d, src):
    h = d.get("history",{}) or {}
    mod = "; ".join(f"{m['category'].capitalize()}: {m['description']}"
                     for m in d.get("modifications",[]))
    res = ", ".join(d.get("resistance",[])) if isinstance(d.get("resistance",[]),list) else d.get("resistance","")
    pub = h.get("publication","")
    if "/" in pub: pub = f'https://doi.org/{pub}' # Store as raw URL
    return {
        "cell_line_name": d.get("cell_line_name",""),
        "organism": d.get("organism",""),
        "sex": d.get("sex",""),
        "tissue": d.get("tissue",""),
        "type": d.get("type",""),
        "derived_from": h.get("derived_from",""),
        "resistance": res,
        "modifications": mod,
        "notes": d.get("notes",""),
        "source": h.get("source",""),
        "catalog_id": h.get("catalog_id",""),
        "publication": pub,
        "authentication": (d.get("authentication") or {}).get("method",""),
        "source_file": src
    }

def flatten_plasmid(d, src):
    status = ", ".join(f"{k}: {v}" for k, v in (d.get("status") or {}).items())

    # publications → bold “Link”
    pubs = d.get("publications", d.get("publication", ""))
    if isinstance(pubs, list):
        pubs = [str(p) for p in pubs]
    if isinstance(pubs, list):
        pubs = ", ".join(
            f'<strong><a href="https://doi.org/{p}" target="_blank">Link</a></strong>'
            if "/" in p else html.escape(p)
            for p in pubs
        )
    elif isinstance(pubs, str) and "/" in pubs:
        pubs = '<strong><a href="https://doi.org/{0}" target="_blank">Link</a></strong>'.format(pubs)
    else:
        pubs = html.escape(str(pubs))

    # website → bold “Link”
    web = d.get("website", "")
    if isinstance(web, str) and web.startswith("http"):
        web = f'<strong><a href="{web}" target="_blank">Link</a></strong>'
    else:
        web = html.escape(str(web))

    return {
        **d,
        "status": status,
        "publications": pubs,
        "website": web,
        "source_file": src
    }

def flatten_animal_model(d, src):
    return {**d, "source_file":src}

# ─── GIT CLONE HELPER ────────────────────────────────────────────────────────
def clone_repo(repo_url: str, private: bool, token: str) -> Optional[Path]:
    try:
        owner, name = repo_url.rstrip("/").split("/")[-2:]
    except:
        st.error("Invalid GitHub URL")
        return None
    headers = {"Accept":"application/vnd.github.v3+json"}
    if private:
        if not token:
            st.error("Token required for private repos")
            return None
        headers["Authorization"] = f"token {token}"

    check = requests.get(f"https://api.github.com/repos/{owner}/{name}/contents", headers=headers)
    if check.status_code != 200:
        st.error(f"Cannot access repo ({check.status_code})")
        return None

    if "tmpdir" not in st.session_state:
        st.session_state.tmpdir_obj = tempfile.TemporaryDirectory()
        st.session_state.tmpdir = st.session_state.tmpdir_obj.name

    path = Path(st.session_state.tmpdir)/name
    if not path.exists():
        with st.spinner("Cloning repository…"):
            url = repo_url
            if private:
                p = urllib.parse.urlparse(repo_url)
                token_safe = urllib.parse.quote(token)
                url = f"https://{token_safe}@{p.netloc}{p.path}"
            res = subprocess.run(["git","clone",url,str(path)], capture_output=True, text=True)
            if res.returncode:
                st.error(f"Git clone failed: {res.stderr}")
                return None
    return path

# ─── APP LOGIC ───────────────────────────────────────────────────────────────
if "repo_path" not in st.session_state:
    # Show download form
    owner = st.text_input("GitHub Repository URL", help="e.g., https://github.com/user/repo")
    private = st.checkbox("Private repo?")
    token = st.text_input("Personal Access Token", type="password", disabled=not private)
    if st.button("🔄 Load Database"):
        if owner:
            rp = clone_repo(owner, private, token)
            if rp:
                st.session_state.repo_path = rp
                st.rerun()
        else:
            st.warning("Please enter a GitHub repository URL to load the database.")
else:
    repo_path = st.session_state.repo_path
    st.success(f"Loaded: {repo_path.name}")
    configs = repo_path/"configs"
    if not configs.exists():
        st.error("`configs/` folder not found in repo.")
        st.stop()

    choices = [k for k in field_config if (configs/k).is_dir()]
    if not choices:
        st.warning("No datasets found.")
        st.stop()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ds = st.selectbox("Dataset", choices)
    with col2:
        query = st.text_input("🔍 Filter rows…", key=f"search_{ds}", placeholder="Search across all columns...")
    
    df = load_dataset(ds, configs)
    
    # ─── LIVE FILTER ───────────────────────────────────────────────────────
    if query:
        df = df[df.apply(
            lambda row: row.astype(str).str.contains(query, case=False).any(),
            axis=1
        )]
    if ds == "cell_lines":
        df = df.drop_duplicates(subset=["cell_line_name"])

    st.info(f"Showing {len(df)} records")
    if df.empty:
        st.warning("No matching records.")
    else:
        # pretty columns + values
        cols = field_config[ds]
        pretty_cols = prettified_field_config[ds]
        disp = df[cols].copy()
        disp.columns = pretty_cols

        # Apply pretty_val to each value, but first, handle the URLs to keep them as is for links later
        for c in pretty_cols:
            disp[c] = disp[c].apply(pretty_val)
        
        # Now, make links clickable. We'll use the raw URL from the DataFrame.
        disp_html = make_all_links_clickable(disp)

        st.markdown(disp_html.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # --- Detail Card ---
    # Moved the detail card selection to only appear when data is available
    if not df.empty:
        st.divider() # Use a divider for better separation
        
        # Use st.expander for a cleaner detail card display
        with st.expander("Show Full Record Details", expanded=False):
            label_col = "antibody_name" if ds == "antibodies" else field_config[ds][0]
            sel_index = st.selectbox(
                "Select a record to view details:",
                df.index,
                format_func=lambda i: df.loc[i].get(label_col, f"Row {i}"),
                key=f"details_select_{ds}"
            )
            
            # Use columns to display key-value pairs
            sel_record = df.loc[sel_index]
            
            # A more robust way to display the detail card using Streamlit
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            st.markdown(f"### Details for: {sel_record.get(label_col, 'Record')}")
            
            # Using st.columns for better alignment without raw HTML dl/dt/dd
            detail_cols = st.columns(2) # Split the card into two columns
            
            # Create a list of items to display
            details_to_show = [(k, pretty_val(v)) for k, v in sel_record.items()]
            
            # Distribute items across the columns
            mid_point = len(details_to_show) // 2
            col1_items = details_to_show[:mid_point]
            col2_items = details_to_show[mid_point:]

            with detail_cols[0]:
                for k, v in col1_items:
                    st.markdown(f"**{prettify_col(k)}:** {v}")
            
            with detail_cols[1]:
                for k, v in col2_items:
                    st.markdown(f"**{prettify_col(k)}:** {v}")
                    
            st.markdown('</div>', unsafe_allow_html=True)
            
    with st.sidebar:
        if "repo_path" in st.session_state:
            if st.button("🔁 Reset / Load another"):
                del st.session_state.repo_path
                st.rerun()
