import streamlit as st

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