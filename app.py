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
def apply_custom_css():
    """Applies custom CSS to the Streamlit app for mobile friendliness."""
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
    .stDataFrame {
        /* Allow horizontal scrolling for tables on small screens */
        overflow-x: auto;
    }

    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0 8px; /* Adds vertical spacing between rows */
        width: 100%; /* Ensure table takes full width of its container */
        min-width: 600px; /* Set a minimum width for the table to prevent squishing */
    }

    .stDataFrame th, .stDataFrame td {
        text-align: center !important;
        vertical-align: middle !important; /* Center vertically as well */
        padding: 8px 12px; /* Add padding for better spacing */
        white-space: nowrap; /* Prevent text wrapping in table cells unless explicitly allowed */
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
        border-radius: 8px !important;
        background-color: #fff !important;
        padding: 1rem !important; /* Adjusted padding slightly for better input feel */
        color: #333 !important; /* Ensure text is visible */
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
        grid-template-columns: 250px 1fr; /* Default for desktop */
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
        word-wrap: break-word; /* Prevents long URLs/text from overflowing */
        word-break: break-all; /* Ensures long strings without spaces break */
    }

    /* Mobile-specific adjustments using Media Queries */
    @media (max-width: 768px) {
        .gradient-header {
            padding: 10px 0 5px 10px;
            margin-bottom: 20px;
        }
        .gradient-header h1 {
            font-size: 1.8em;
        }
        .gradient-header span {
            font-size: 0.9em;
        }

        /* Stack columns for dataset selector and search on small screens */
        div[data-testid="stColumn"] {
            width: 100% !important; /* Force columns to take full width */
            margin-bottom: 1rem; /* Add spacing between stacked columns */
        }

        /* Adjust detail card layout for mobile */
        .detail-card dl {
            grid-template-columns: 1fr; /* Single column layout for definition lists */
            gap: 0.5rem; /* Reduce gap on mobile */
        }
        .detail-card dt {
            margin-bottom: 0.2rem; /* Small margin for dt on mobile */
        }
        .detail-card dd {
            margin-bottom: 0.8rem; /* Small margin for dd on mobile */
        }
    }
    /* center any <table> you inject via Markdown */
[data-testid="stMarkdownContainer"] table th,
[data-testid="stMarkdownContainer"] table td {
  text-align: center !important;
  vertical-align: middle !important;
}

    /* Further refine table for very small screens if needed */
    @media (max-width: 480px) {
        .stDataFrame th, .stDataFrame td {
            padding: 6px 8px; /* Smaller padding on very small screens */
            font-size: 0.9em; /* Slightly smaller font for table text */
        }
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
DESKTOP_CONFIG = {
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

MOBILE_CONFIG = {
    "antibodies":      ["antibody_target", "antibody_name", "location"],
    "cell_lines":      ["cell_line_name", "modifications"],
    "plasmids":        ["plasmid_name", "location"],
    "animal_models":   ["model_name", "organism"],
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


def pretty_val(v: Any) -> Any:
    # This function bolds keys for dictionaries, which is generally desired for other dicts.
    # The 'status' field for antibodies will now be pre-flattened in flatten_antibody.
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
                # Call the new flatten_antibody function
                rec = flatten_antibody(entry.copy(), tgt, file.name)
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
def flatten_antibody(d, target, src):
    """Flattens antibody data, specifically handling status and website fields."""
    d["antibody_target"] = target
    d["source_file"] = src

    # If 'status' is a dictionary, flatten it into a string without bolding its internal keys.
    if isinstance(d.get("status"), dict):
        d["status"] = "; ".join(f"{k}: {v}" for k, v in d["status"].items())

    # Ensure website URLs are formatted consistently with the rest of the app for display
    web = d.get("website", "")
    if isinstance(web, str) and web.startswith("http"):
        # Store as raw URL here; make_all_links_clickable will create the <strong><a>Link</a></strong>
        d["website"] = web
    else:
        d["website"] = html.escape(str(web))
    
    return d


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
        # Store raw URLs first, make_all_links_clickable handles HTML formatting later
        pubs = [str(p) if not str(p).startswith("http") else f'https://doi.org/{p}' for p in pubs]
        pubs = ", ".join(pubs) # Join into a single string for pretty_val
    elif isinstance(pubs, str) and "/" in pubs:
        pubs = f'https://doi.org/{pubs}' # Store raw URL
    else:
        pubs = html.escape(str(pubs))

    # website → raw URL
    web = d.get("website", "")
    if isinstance(web, str) and web.startswith("http"):
        web = web # Store as raw URL
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
    # For animal models, ensure website is handled as a raw URL similar to other flatten functions
    web = d.get("website", "")
    if isinstance(web, str) and web.startswith("http"):
        d["website"] = web
    else:
        d["website"] = html.escape(str(web))
    return {**d, "source_file": src}

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
# Initialize mobile_view_checkbox in session_state if not already present
if 'mobile_view_checkbox' not in st.session_state:
    st.session_state.mobile_view_checkbox = False

if "repo_path" not in st.session_state:
    # Show download form
    owner = st.text_input("GitHub Repository URL", help="e.g., https://github.com/user/repo")
    private = st.checkbox("Private repo?")
    token = st.text_input("Personal Access Token", type="password", disabled=not private)
    
    # Mobile view checkbox on the first screen
    # Use a different key for this instance to avoid StreamlitAPIException
    st.session_state.mobile_view_checkbox = st.checkbox(
        "📱 Mobile view (Fewer columns)",
        value=st.session_state.mobile_view_checkbox, # Read from session state
        key='mobile_view_checkbox_initial_screen'
    )
    
    # Determine config based on the current mobile_view_checkbox state
    cfg = MOBILE_CONFIG if st.session_state.mobile_view_checkbox else DESKTOP_CONFIG

    # prettified_field_config needs to be defined here for the initial load, using the current cfg
    prettified_field_config = {
        k: [prettify_col(c) for c in cols]
        for k, cols in cfg.items()
    }
    
    if st.button("🔄 Load Database"):
        if owner:
            rp = clone_repo(owner, private, token)
            if rp:
                st.session_state.repo_path = rp
                st.rerun()
        else:
            st.warning("Please enter a GitHub repository URL.")
else:
    repo_path = st.session_state.repo_path

    # Mobile view checkbox in the sidebar
    with st.sidebar:
        st.subheader("Display Options")
        st.session_state.mobile_view_checkbox = st.checkbox(
            "📱 Mobile view (Fewer columns)",
            value=st.session_state.mobile_view_checkbox, # Read from session state
            key='mobile_view_checkbox_sidebar' # Different key for sidebar instance
        )
        if st.button("🔁 Reset / Load another"):
            del st.session_state.repo_path
            # Reset mobile view checkbox state
            st.session_state.mobile_view_checkbox = False
            st.rerun()

    # Determine which config to use based on the current mobile_view_checkbox state
    cfg = MOBILE_CONFIG if st.session_state.mobile_view_checkbox else DESKTOP_CONFIG

    # Re-calculate prettified_field_config based on the chosen cfg
    prettified_field_config = {
        k: [prettify_col(c) for c in cols]
        for k, cols in cfg.items()
    }

    st.success(f"Loaded: {repo_path.name}")
    configs = repo_path / "configs"

    if not configs.exists():
        st.error("`configs/` folder not found in repo.")
        st.stop()

    choices = [k for k in DESKTOP_CONFIG if (configs/k).is_dir()] # Use DESKTOP_CONFIG keys for initial choices
    if not choices:
        st.warning("No datasets found.")
        st.stop()
    
    col1, col2 = st.columns([1, 3]) # These columns will automatically stack on mobile
    with col1:
        ds = st.selectbox("Dataset", choices)
    with col2:
        query = st.text_input("🔍 Filter rows…", key=f"search_{ds}", placeholder="Search…")
    
    df = load_dataset(ds, configs)
    if query:
        df = df[df.apply(lambda r: r.astype(str).str.contains(query, case=False).any(), axis=1)]
    if ds == "cell_lines":
        df = df.drop_duplicates(subset=["cell_line_name"])

    st.info(f"Showing {len(df)} records")
    if not df.empty:
        # use the chosen config
        cols = cfg[ds] # Use the selected config (mobile or desktop)
        pretty_cols = [prettify_col(c) for c in cols]

        disp = df[cols].copy()
        disp.columns = pretty_cols
        for c in pretty_cols:
            disp[c] = disp[c].apply(pretty_val)

        st.markdown(
            make_all_links_clickable(disp).to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    else:
        st.warning("No matching records found for the selected dataset or filters.")
        
    # --- Detail Card ---
    # Moved the detail card selection to only appear when data is available
    if not df.empty:
        st.divider() # Use a divider for better separation
        
        # Use st.expander for a cleaner detail card display
        with st.expander("Show Full Record Details", expanded=False):
            # Ensure label_col uses the full DESKTOP_CONFIG field to always get a good label
            label_col = "antibody_name" if ds == "antibodies" else DESKTOP_CONFIG[ds][0]
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
            # These columns will automatically stack on mobile due to default Streamlit behavior
            detail_cols = st.columns(2) 
            
            # Create a list of items to display
            # Display all details for the selected record, regardless of mobile/desktop table view
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
            
