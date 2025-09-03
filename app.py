# Inject custom CSS to make sidebar wider and force it open by default
import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
import urllib.parse
import subprocess
import tempfile
import requests
import yaml
import html

import src.streamlit_pandas as streamlit_pandas
from src.css_styles import apply_custom_css

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LabChronicle | Lab Database",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Apply the CSS function
apply_custom_css()

# ─── HEADER MARKDOWN ────────────────────────────────────────────────────────
st.markdown("""
<div class="gradient-header">
  <h1>
    LabChronicle <span style="font-size:0.7em;font-weight:300;">| Database Visualizer</span>
  </h1>
  <span style="color:#e0ecff;font-size:1.1em;">
    🔬 All your lab databases (experiments, reagents, cell lines, etc.), in one place
  </span>
</div>
""", unsafe_allow_html=True)

# ─── CONFIGS ────────────────────────────────────────────────────────────────
DESKTOP_CONFIG = {
    "animal_models": {
        "model_name": "text",
        "organism": "multiselect",
        "background": "text",
        "strain": "text",
        "genotype": "text",
        "source": "text",
        "catalogue_number": "text",
        "website": "text",
        "sex": "multiselect",
        "notes": "text"
    },
    "antibodies": {
        "antibody_name": "text", 
        "antibody_target": "multiselect", 
        "location": "multiselect", 
        "clonality": "multiselect",
        "host_species": "multiselect", 
        "catalog_id": "text", 
        "source": "text", 
        "applications": "multiselect",
        "notes": "-", 
        "website": "-", 
        "status.tube": "multiselect",
        "status.reorder": "multiselect",
        "status.last_checked": "date"
    },
    "cell_lines": {
        "cell_line_name": "text",
        "organism": "text",
        "tissue": "text",
        "type": "text",
        "derived_from": "text",
        "resistance": "text",
        "modifications": "text",
        "notes": "text",
        "source": "text",
        "catalog_id": "text",
        "publication": "text",
        "authentication": "text"
    },
    "plasmids": {
        "plasmid_name": "text",
        "backbone": "text",
        "promoter": "text",
        "tag": "text",
        "origin": "text",
        "catalog_id": "text",
        "website": "text",
        "expression_host": "text",
        "sequencing_validation": "text",
        "location": "text",
        "notes": "text",
        "status": "text",
        "MTA": "text",
        "publications": "text"
    },
    "experiments": {
        "experiment_name": "text",
        "date_started": "text",
        "performed_by": "text",
        "primary_method": "text",
        "cell_lines": "text",
        "animal_models": "text",
        "global_metadata.Microscope": "text",
        "global_metadata.Live Imaging": "text",
        "conditions": "text",
        "device_name": "text",
        "location": "text",
        "lab_chronicle_version": "text"
    }
}

IGNORE_FILTER_COLUMNS = {
    "animal_models": ["source_file", "website", "notes"],
    "antibodies": ["source_file", "website", "notes"],
    "cell_lines": ["source_file", "publication", "notes"],
    "plasmids": ["source_file", "website", "notes"],
    "experiments": ["lab_chronicle_version", "location"]
}

temp_mobile_config = {
    "antibodies":      ["antibody_target", "antibody_name", "location"],
    "cell_lines":      ["cell_line_name", "modifications"],
    "plasmids":        ["plasmid_name", "location"],
    "animal_models":   ["model_name", "organism"],
    "experiments": [
        "experiment_name"
        "date_started",
        "performed_by",
        "device_name",
        "location",
    ]
}

MOBILE_CONFIG = {}
for k, v in temp_mobile_config.items():
    MOBILE_CONFIG[k] = {col: DESKTOP_CONFIG[k][col] for col in v if col in DESKTOP_CONFIG[k]}

PRETTY_DATABASES = {
    "antibodies": "Antibodies",
    "cell_lines": "Cell Lines",
    "plasmids": "Plasmids",
    "animal_models": "Animal Models",
    "experiments": "Experiments"
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
        "status.tube": "Tube's status",
        "status.reorder": "Reorder?",
        "status.last_checked": "Last Time Checked",
        # Experiment related
        "experiment_name": "Experiment Name",
        "date_started": "Start Date",
        "performed_by": "Performed By",
        "primary_method": "Primary Method",
        "cell_lines": "Cell Lines",
        "animal_models": "Animal Models",
        "global_metadata.Microscope": "Microscope",
        "global_metadata.Live Imaging": "Live Imaging",
        "conditions": "Experimental Conditions",
        "device_name": "Device Name",
        "location": "Location",
        "lab_chronicle_version": "LabChronicle Version",
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

# ─── YAML → DataFrame LOADER ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(dataset: str, database_root_path: Path) -> pd.DataFrame:
    if dataset == "experiments":
        database_folder = database_root_path / "experiment_database"
    else:
        database_folder = database_root_path / "lab_database" / dataset

    if not database_folder.exists():
        return pd.DataFrame()

    records = []

    if dataset == "experiments":
        database_file = database_folder / "experiment_database.yaml"
        database = yaml.safe_load(database_file.read_text()) or {}

        for hard_drive_idx, hard_drive_info in database.items():
            hard_drive_file = database_folder / hard_drive_idx / "drive_metadata.yaml"
            hard_drive_content = yaml.safe_load(hard_drive_file.read_text()) or {}
            for exp_idx, exp_metadata in hard_drive_content["experiments"].items():
                exp_info = extract_experiment_info(database_folder / hard_drive_idx / exp_idx)
                exp_info.update({k:v for k,v in exp_metadata.items() if k not in ["experiment_checksum"]})  # Merge dictionaries
                exp_info.update({k:v for k,v in hard_drive_info.items() if k not in ["physical_serial_number", 
                                                                                    "volume_serial_number"]})  # Merge dictionaries
                records.append(exp_info)
    else:
        for file in database_folder.glob("*.yaml"):
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
    df = pd.DataFrame(records)
    return df

# ─── FLATTEN HELPERS ─────────────────────────────────────────────────────────
def extract_experiment_info(src: str) -> Dict[str, Any]:
    experiment_metadata_file = src / "metadata.yaml"
    if not experiment_metadata_file.exists():
        return {}

    metadata = yaml.safe_load(experiment_metadata_file.read_text()) or {}

    filtered_metadata = {}

    for key, value in metadata.items():
        filtered_metadata.update(flatten_experiment_metadata(key, value))

    return filtered_metadata 

def flatten_experiment_metadata(key, value):

    # First do a checking of specific key values
    if key == "conditions":
        return {key: ",\n".join([v["name"] for k, v in value.items()])}
    elif key in ["repeats", "experiment_id"]:
        return {}

    if not isinstance(value, dict) and not isinstance(value, list):
        return {key: value}
    elif isinstance(value, dict):
        output_dict = {}
        for sub_key, sub_value in value.items():
            output_dict.update(flatten_experiment_metadata(f"{key}.{sub_key}", sub_value))
        return output_dict
    elif isinstance(value, list):
        any_list_dict = any(isinstance(item, list) or isinstance(item, dict) for item in value)

        if any_list_dict:
            output_dict = {}
            for i, item in enumerate(value):
                output_dict.update(flatten_experiment_metadata(f"{key}.{i}", item))
            return output_dict
        else:
            return {key: ",\n".join(value)}
    return {}


def flatten_antibody(d, target, src):
    """Flattens antibody data, specifically handling status and website fields."""
    d["antibody_target"] = target
    d["source_file"] = src

    # If 'status' is a dictionary, flatten it into a string without bolding its internal keys.
    if isinstance(d.get("status"), dict):
        # Expand the status dictionary
        for k,v in d["status"].items():
            d[f"status.{k}"] = v
        # Pop the original status field
        d.pop("status", None)

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
        k: [prettify_col(c) for c in cols.keys()]
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

    # Determine which config to use based on the current mobile_view_checkbox state
    cfg = MOBILE_CONFIG if st.session_state.mobile_view_checkbox else DESKTOP_CONFIG

    # Re-calculate prettified_field_config based on the chosen cfg
    prettified_field_config = {
        k: [prettify_col(c) for c in cols.keys()]
        for k, cols in cfg.items()
    }

    st.success(f"Loaded: {repo_path.name}")

    lab_database = repo_path / "lab_database"
    experiment_database = repo_path / "experiment_database"

    if not experiment_database.exists() and not lab_database.exists():
        st.warning("No database found.")
        st.stop()

    # Prepare choices for the database selection
    choices = [k for k in DESKTOP_CONFIG if (lab_database/k).is_dir()] # Use DESKTOP_CONFIG keys for initial choices
    choices.insert(0, "experiments")
    choices.sort()
    pretty_choices = [PRETTY_DATABASES.get(c, c) for c in choices]

    # Choose and load the database
    st.subheader("Choose the database to explore")
    ds = st.selectbox("Database options:", pretty_choices)
    ds = choices[pretty_choices.index(ds)]

    df = load_dataset(ds, repo_path)
    
    with st.sidebar:
        st.markdown("## Options")  # Sidebar title

    if not df.empty:
        # Remove duplicates for cell lines based on cell_line_name
        if ds == "cell_lines":
            df = df.drop_duplicates(subset=["cell_line_name"])

        # Only take relevant columns
        temp_df = df[cfg[ds].keys()].copy()

        # oad the filtering options and create the sidebar widget for filtering
        ignore_columns = [e for e in IGNORE_FILTER_COLUMNS.get(ds, []) if e in cfg.keys()]
        create_data = cfg[ds]      

        with st.sidebar:
            with st.expander("Filter Options"):
               all_widgets = streamlit_pandas.create_widgets(temp_df, 
                                                             create_data, 
                                                             ignore_columns=ignore_columns,
                                                             prettify_function=prettify_col)

        # Filter the dataframe
        filtered_df = streamlit_pandas.filter_df(temp_df, all_widgets)
        pretty_cols = [prettify_col(c) for c in filtered_df.columns]
        filtered_df.columns = pretty_cols

        # Display the filtered dataframe
        st.info(f"Showing {len(filtered_df)} records")
        event = st.dataframe(
            filtered_df, 
            height="auto", 
            hide_index=True,    
            on_select="rerun",
            column_config={
                "Website": st.column_config.LinkColumn("Website", display_text="Link"),
                "Publication DOI": st.column_config.LinkColumn("Publication DOI", display_text="Link")
            }
        )

        st.download_button(
                label="Export Selected Rows",
                data=df.loc[event.selection.rows].to_csv(),
                file_name="selected_rows.csv",
                mime="text/csv",
                disabled=not event.selection.rows
            )
        
    else:
        st.warning("No matching records found for the selected dataset or filters.")
        
    
    # Mobile view checkbox in the sidebar
    with st.sidebar:
        with st.expander("Display Options"):
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

    # --- Detail Card ---
    # Moved the detail card selection to only appear when data is available
    if not df.empty:
        st.divider() # Use a divider for better separation
        
        # Use st.expander for a cleaner detail card display
        with st.expander("Show Full Record Details", expanded=False):
            # Ensure label_col uses the full DESKTOP_CONFIG field to always get a good label
            label_col = "experiment_name" if "experiment_name" in df.columns else df.columns[0]
            sel_index = st.selectbox(
                "Select a record to view details:",
                filtered_df.index,
                format_func=lambda i: df.loc[i].get(label_col, f"Row {i}"),
                key="details_select"
            )

            # Use columns to display key-value pairs
            sel_record = df.loc[sel_index]

            # A more robust way to display the detail card using Streamlit
            st.markdown(f"### Details for: {sel_record.get(label_col, 'Record')}")
            with st.container():
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
                    for key, value in col1_items:
                        if isinstance(value, str) and ",\n" in value:
                            items = value.split(",\n")
                            value = "\n".join([f"- {item}" for item in items])
                            st.markdown(f"**{prettify_col(key)}:**\n{value}")
                        else:
                            st.markdown(f"**{prettify_col(key)}:** {value}")

                with detail_cols[1]:
                    for key, value in col2_items:
                        if isinstance(value, str) and ",\n" in value:
                            items = value.split(",\n")
                            value = "\n".join([f"- {item}" for item in items])
                            st.markdown(f"**{prettify_col(key)}:**\n{value}")
                        else:
                            st.markdown(f"**{prettify_col(key)}:** {value}")