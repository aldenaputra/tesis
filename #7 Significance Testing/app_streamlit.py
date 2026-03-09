import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from scipy.stats import norm

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Forecast Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SHARED CONSTANTS
# ==========================================
EVENTS = [
    ('2023-08-01', 'US Credit Downgrade', '#FF6B6B', 0.05),
    ('2024-03-20', 'Pengumuman Pemilu RI', '#4ECDC4', 0.10),
    ('2024-04-13', 'Middle-East Escalation', '#FF8C42', 0.05),
    ('2024-09-18', 'BI Rate turun 0.25 bps', '#95E1D3', 0.05),
    ('2024-11-05', 'US Election, Trump Wins', '#F38181', 0.10),
]

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_uq_data():
    try:
        df = pd.read_csv('ALL_UQ_PREDICTED.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    except FileNotFoundError:
        st.error("❌ 'ALL_UQ_PREDICTED.csv' not found.")
        df = pd.DataFrame()

    metrics_path = os.path.join('..', 'Results', 'ALL_UQ_METRICS.csv')
    try:
        metrics_df = pd.read_csv(metrics_path)
        winkler_map = dict(zip(metrics_df['model_name'], metrics_df['winkler_score']))
    except FileNotFoundError:
        st.warning("⚠️ ALL_UQ_METRICS.csv not found. Winkler score filtering unavailable.")
        metrics_df = None
        winkler_map = {}

    return df, metrics_df, winkler_map


@st.cache_data
def load_pf_data():
    try:
        df = pd.read_csv('ALL_PREDICTED.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    except FileNotFoundError:
        st.error("❌ 'ALL_PREDICTED.csv' not found.")
        df = pd.DataFrame()

    metrics_path = os.path.join('..', 'Results', 'ALL_METRICS.csv')
    try:
        metrics_df = pd.read_csv(metrics_path)
    except FileNotFoundError:
        st.warning("⚠️ ALL_METRICS.csv not found. Metric filtering unavailable.")
        metrics_df = None

    return df, metrics_df


# ==========================================
# COLUMN PARSING
# ==========================================
@st.cache_data
def parse_uq_columns(df):
    all_cols = [c for c in df.columns if c != 'actual']
    base_cols = [c for c in all_cols if not c.endswith('_L') and not c.endswith('_U')]
    models, schemes, seeds, col_meta = set(), set(), set(), {}
    for col in base_cols:
        parts = col.split('_')
        if len(parts) >= 3:
            m, s, sd = parts[0], parts[1], parts[2]
            models.add(m); schemes.add(s); seeds.add(sd)
            col_meta[col] = (m, s, sd)
    return (
        base_cols, col_meta,
        sorted(models),
        sorted(schemes),
        sorted(seeds, key=lambda x: int(x) if x.isdigit() else x),
    )


@st.cache_data
def parse_pf_columns(df):
    """Parse point-forecast columns: {model}_{tuning}_{seed}"""
    base_cols = [c for c in df.columns if c != 'actual']
    models, tunings, seeds, col_meta = set(), set(), set(), {}
    for col in base_cols:
        parts = col.split('_')
        if len(parts) >= 3:
            m, t, sd = parts[0], parts[1], parts[2]
            models.add(m); tunings.add(t); seeds.add(sd)
            col_meta[col] = (m, t, sd)
    return (
        base_cols, col_meta,
        sorted(models),
        sorted(tunings),
        sorted(seeds, key=lambda x: int(x) if x.isdigit() else x),
    )


# ==========================================
# SHARED HELPER: GEOPOLITICAL EVENTS
# ==========================================
def draw_events(ax, sub_df):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for date_str, label, color, offset in EVENTS:
        event_date = pd.to_datetime(date_str)
        if sub_df.index[0] <= event_date <= sub_df.index[-1]:
            ax.axvline(x=event_date, color=color, linestyle='--', linewidth=2, alpha=0.7, zorder=50)
            ax.text(event_date, y_max - (y_range * offset), label,
                    rotation=0, verticalalignment='bottom', horizontalalignment='center',
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))


# ==========================================
# METRICS TABLE HELPERS
# ==========================================
PF_METRIC_DIRECTIONS = {
    'r2_score': 'max', 'rmse': 'min', 'mae': 'min', 'mape': 'min', 'mse': 'min',
}
UQ_METRIC_DIRECTIONS = {
    'winkler_score': 'min', 'picp': 'max', 'mpiw': 'min',
    'r2_score': 'max', 'rmse': 'min', 'mae': 'min',
}


def enrich_metrics(metrics_df, group_label):
    """Parse 'arch_group_seed' model_name into separate Architecture/group/Seed columns."""
    df = metrics_df.copy()
    extracted = df['model_name'].str.extract(r'^([^_]+)_([^_]+)_(.+)$')
    df['Architecture'] = extracted[0].str.upper()
    df[group_label] = extracted[1].str.upper()
    df['Seed'] = extracted[2]
    return df


def add_best_markers(df, metric_col, group_col, higher_is_better):
    """Add '🥇 Global' and '✅ Scenario' marker column based on chosen metric."""
    df = df.copy()
    df['Best'] = ''
    valid = df[metric_col].dropna()
    if valid.empty:
        return df
    global_idx = valid.idxmax() if higher_is_better else valid.idxmin()
    df.loc[global_idx, 'Best'] = '🥇 Global'
    for _, grp in df.groupby(group_col):
        v = grp[metric_col].dropna()
        if v.empty:
            continue
        local_idx = v.idxmax() if higher_is_better else v.idxmin()
        if df.loc[local_idx, 'Best'] == '':
            df.loc[local_idx, 'Best'] = '✅ Scenario'
    return df


def apply_gradient_style(df, metric_directions):
    """Background-gradient each metric column; color scale clipped to 5th–95th percentile."""
    fmt = {c: '{:.4f}' for c in metric_directions if c in df.columns}
    styler = df.style.format(fmt, na_rep='—')
    for col, direction in metric_directions.items():
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if valid.empty:
            continue
        cmap = 'RdYlGn' if direction == 'max' else 'RdYlGn_r'
        vmin, vmax = valid.quantile(0.05), valid.quantile(0.95)
        if vmin == vmax:
            vmin, vmax = valid.min(), valid.max()
        styler = styler.background_gradient(
            cmap=cmap, subset=[col], axis=0, vmin=vmin, vmax=vmax
        )
    return styler


def apply_dual_highlight(display_df, conv_df, metric_cols, direction_map, group_col=None):
    """
    Per-metric, per-group cell highlighting:
      Blue   = best mean  (direction-aware: min or max)
      Red    = lowest std (always min — lower std = more consistent)
      Purple = same row wins BOTH
    display_df and conv_df must share a matching RangeIndex.
    group_col=None treats the whole table as one group.
    """
    BLUE   = 'background-color: #adc6ff; color: #003399; font-weight: bold'
    RED    = 'background-color: #ffadad; color: #990000; font-weight: bold'
    PURPLE = 'background-color: #d4aaff; color: #4b0082; font-weight: bold'

    styles = pd.DataFrame('', index=display_df.index, columns=display_df.columns)

    if group_col is not None and group_col in conv_df.columns:
        group_map = {val: conv_df.index[conv_df[group_col] == val].tolist()
                     for val in conv_df[group_col].unique()}
    else:
        group_map = {'__all__': list(conv_df.index)}

    for grp_indices in group_map.values():
        for metric in metric_cols:
            mean_col  = f'{metric}_mean'
            std_col   = f'{metric}_std'
            direction = direction_map.get(metric, 'min')

            mean_vals = (conv_df.loc[grp_indices, mean_col].dropna()
                         if mean_col in conv_df.columns else pd.Series(dtype=float))
            std_vals  = (conv_df.loc[grp_indices, std_col].dropna()
                         if std_col  in conv_df.columns else pd.Series(dtype=float))

            best_mean_idx = ((mean_vals.idxmin() if direction == 'min' else mean_vals.idxmax())
                             if not mean_vals.empty else None)
            best_std_idx  = (std_vals.idxmin() if not std_vals.empty else None)

            # Which display columns to colour
            mean_targets = []
            if metric   in display_df.columns: mean_targets.append(metric)
            elif mean_col in display_df.columns: mean_targets.append(mean_col)
            std_targets = [std_col] if std_col in display_df.columns else []

            for idx in grp_indices:
                if idx not in display_df.index:
                    continue
                is_bm = (best_mean_idx is not None and idx == best_mean_idx)
                is_bs = (best_std_idx  is not None and idx == best_std_idx)

                # Colour the mean / combined column
                if mean_targets:
                    c = (PURPLE if (is_bm and is_bs) else
                         BLUE   if is_bm else
                         RED    if is_bs else '')
                    if c:
                        for col in mean_targets:
                            styles.loc[idx, col] = c

                # Colour the std column (if visible)
                if std_targets:
                    c = (PURPLE if (is_bm and is_bs) else
                         RED    if is_bs else
                         BLUE   if is_bm else '')
                    if c:
                        for col in std_targets:
                            styles.loc[idx, col] = c

    return display_df.style.apply(lambda _: styles, axis=None)


def build_convergence_table(df, group_col, metric_cols):
    """Group by (Architecture, group_col), compute mean / std / CV% across seeds."""
    records = []
    for (arch, grp), subset in df.groupby(['Architecture', group_col]):
        row = {'Architecture': arch, group_col: grp, 'N Seeds': len(subset)}
        for col in metric_cols:
            if col not in subset.columns:
                continue
            vals = subset[col].dropna()
            if vals.empty:
                continue
            m = vals.mean()
            s = vals.std(ddof=1) if len(vals) > 1 else 0.0
            cv = (s / abs(m) * 100) if m != 0 else np.nan
            row[f'{col}_mean'] = round(m, 4)
            row[f'{col}_std'] = round(s, 4)
            row[f'{col}_cv%'] = round(cv, 2)
        records.append(row)
    return pd.DataFrame(records)


# ==========================================
# PAGE NAVIGATION
# ==========================================
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["📈 Point Forecast", "🎯 UQ Analysis",
     "📋 PF Training Results", "📋 UQ Training Results",
     "🔬 DM Significance Test", "🔎 EDA Statistics",
     "🔀 Uncertainty Decomposition"],
    index=0
)
st.sidebar.markdown("---")


# ==================================================================================
# PAGE 1 — POINT FORECAST
# ==================================================================================
if page == "📈 Point Forecast":
    st.title("📈 Point Forecast Dashboard")
    st.markdown("Visualize point predictions from **Baseline**, **Random Search**, and **Optuna** tuning runs.")
    st.markdown("---")

    pf_df, pf_metrics_df = load_pf_data()

    if pf_df.empty:
        st.stop()

    pf_base_cols, pf_col_meta, pf_models, pf_tunings, pf_seeds = parse_pf_columns(pf_df)

    # Build metric maps for quality filter and labels
    r2_map = {}
    if pf_metrics_df is not None and 'model_name' in pf_metrics_df.columns:
        if 'r2_score' in pf_metrics_df.columns:
            r2_map = dict(zip(pf_metrics_df['model_name'], pf_metrics_df['r2_score']))

    # --- Sidebar controls ---
    st.sidebar.header("⚙️ Configuration")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pf_sel_model = st.selectbox("Model:", ['All'] + pf_models, key='pf_model')
    with col2:
        pf_sel_tuning = st.selectbox("Tuning:", ['All'] + pf_tunings, key='pf_tuning')
    pf_sel_seed = st.sidebar.selectbox("Seed:", ['All'] + pf_seeds, key='pf_seed')

    st.sidebar.header("📊 Display Options")
    pf_show_events = st.sidebar.checkbox("Show Geopolitical Events", value=False, key='pf_events')

    st.sidebar.header("🏆 Quality Filter")
    pf_top_r2 = st.sidebar.checkbox("Filter Top 10% Highest R² Score", value=False, key='pf_top_r2')

    st.sidebar.header("📅 Date Range")
    pf_date_range = st.sidebar.slider(
        "Select date range:",
        min_value=0, max_value=len(pf_df) - 1,
        value=(0, len(pf_df) - 1), step=1, key='pf_date_range'
    )

    # --- Filtering ---
    cols_to_plot_final = list(pf_base_cols)
    if pf_top_r2 and r2_map:
        r2_series = pd.Series(r2_map)
        threshold = r2_series.quantile(0.9)
        top_cols = set(r2_series[r2_series >= threshold].index)
        cols_to_plot_final = [c for c in pf_base_cols if c in top_cols]
        if not cols_to_plot_final:
            st.warning(f"No models in top 10% R² (threshold: {threshold:.4f})")

    start_idx, end_idx = pf_date_range
    sub_df = pf_df.iloc[start_idx: end_idx + 1]

    cols_to_plot = []
    for col in cols_to_plot_final:
        m, t, sd = pf_col_meta.get(col, (None, None, None))
        if pf_sel_model != 'All' and m != pf_sel_model:
            continue
        if pf_sel_tuning != 'All' and t != pf_sel_tuning:
            continue
        if pf_sel_seed != 'All' and sd != pf_sel_seed:
            continue
        cols_to_plot.append(col)

    # --- Plot ---
    st.subheader("📈 Interactive Visualization")
    with st.spinner("Generating plot..."):
        fig, ax1 = plt.subplots(figsize=(16, 7))

        ax1.plot(sub_df.index, sub_df['actual'],
                 label='Actual Data', color='black', linewidth=2.5, alpha=0.9, zorder=100)

        colors = sns.color_palette("bright", max(len(cols_to_plot), 1))

        if not cols_to_plot:
            ax1.text(0.5, 0.5, "No models match these filters",
                     ha='center', transform=ax1.transAxes, fontsize=14)
        else:
            for i, col in enumerate(cols_to_plot):
                r2_val = r2_map.get(col, None)
                label = f"{col} (R²: {r2_val:.4f})" if r2_val is not None else col
                ax1.plot(sub_df.index, sub_df[col], label=label,
                         color=colors[i], linewidth=1.8, alpha=0.9)

        if pf_show_events and len(sub_df) > 0:
            draw_events(ax1, sub_df)

        ax1.set_title(f"Point Forecast Analysis: {len(cols_to_plot)} Models", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Prediction Value', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)

        if len(cols_to_plot) <= 8:
            ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        else:
            ax1.text(1.02, 0.95, f"Legend hidden\n({len(cols_to_plot)} models)\nFilter to < 8",
                     transform=ax1.transAxes, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        st.pyplot(fig)

    # --- Info Panel ---
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", len(pf_df))
    with c2:
        st.metric("Available Models", len(pf_models))
    with c3:
        st.metric("Tuning Methods", len(pf_tunings))
    with c4:
        st.metric("Seeds", len(pf_seeds))

    if st.checkbox("Show sample data", key='pf_sample'):
        pf_min_date = pf_df.index.min().date()
        pf_max_date = pf_df.index.max().date()
        pf_preview_range = st.date_input(
            "Preview date range:",
            value=(pf_min_date, pf_max_date),
            min_value=pf_min_date,
            max_value=pf_max_date,
            key='pf_preview_range'
        )
        if isinstance(pf_preview_range, (list, tuple)) and len(pf_preview_range) == 2:
            pf_preview = pf_df.loc[str(pf_preview_range[0]):str(pf_preview_range[1])]
        else:
            pf_preview = pf_df
        st.dataframe(pf_preview, use_container_width=True)

    if st.checkbox("Show model list", key='pf_model_list'):
        st.write("**Model Coverage Summary:**")
        rows = []
        for col, (m, t, sd) in pf_col_meta.items():
            rows.append({'Architecture': m.upper(), 'Tuning': t.capitalize(), 'Seed': sd})
        pf_summary = pd.DataFrame(rows)

        n_arch  = pf_summary['Architecture'].nunique()
        n_tuning = pf_summary['Tuning'].nunique()
        n_seeds  = pf_summary['Seed'].nunique()
        total    = len(pf_summary)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Architectures", n_arch)
        mc2.metric("Tuning Scenarios", n_tuning)
        mc3.metric("Seeds", n_seeds)
        mc4.metric("Total Runs", total)

        st.write("**Architectures:**", " | ".join(sorted(pf_summary['Architecture'].unique())))
        st.write("**Tuning Methods:**", " | ".join(sorted(pf_summary['Tuning'].unique())))
        st.write("**Seeds:**", " | ".join(sorted(pf_summary['Seed'].unique(),
                                                    key=lambda x: int(x) if x.isdigit() else x)))

        pivot_pf = pf_summary.groupby(['Tuning', 'Architecture'])['Seed'].apply(
            lambda s: ', '.join(sorted(s, key=lambda x: int(x) if x.isdigit() else x))
        ).unstack(fill_value='—')
        st.write("**Runs Matrix (Tuning × Architecture → Seeds):**")
        st.dataframe(pivot_pf, use_container_width=True)


# ==================================================================================
# PAGE 2 — UQ ANALYSIS
# ==================================================================================
elif page == "🎯 UQ Analysis":
    st.title("🎯 Uncertainty Quantification Analysis Dashboard")
    st.markdown("Visualize prediction intervals from **MCD**, **HLLLA**, and **CQR** uncertainty methods.")
    st.markdown("---")

    df, metrics_df, winkler_map = load_uq_data()

    if df.empty:
        st.stop()

    base_cols, col_meta, sorted_models, sorted_schemes, sorted_seeds = parse_uq_columns(df)

    # --- Sidebar controls ---
    st.sidebar.header("⚙️ Configuration")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_model = st.selectbox("Model:", ['All'] + sorted_models, key='uq_model')
    with col2:
        selected_scheme = st.selectbox("UQ Scheme:", ['All'] + sorted_schemes, key='uq_scheme')
    selected_seed = st.sidebar.selectbox("Seed:", ['All'] + sorted_seeds, key='uq_seed')

    st.sidebar.header("📊 Display Options")
    show_band = st.sidebar.checkbox("Show Uncertainty Bands", value=True, key='uq_band')
    show_coverage = st.sidebar.checkbox("Show Band Coverage (Green=Inside, Red=Outside)", value=False, key='uq_coverage')
    show_rolling = st.sidebar.checkbox("Show Rolling PICP & n-MPIW (30-day window)", value=False, key='uq_rolling')
    show_events = st.sidebar.checkbox("Show Geopolitical Events", value=False, key='uq_events')

    st.sidebar.header("🏆 Quality Filter")
    top_winkler = st.sidebar.checkbox("Filter Top 10% Smallest Winkler Score", value=False, key='uq_winkler')

    st.sidebar.header("📅 Date Range")
    date_range = st.sidebar.slider(
        "Select date range:",
        min_value=0, max_value=len(df) - 1,
        value=(0, len(df) - 1), step=1, key='uq_date_range'
    )

    # --- Filtering ---
    cols_to_plot_final = list(base_cols)
    if top_winkler and metrics_df is not None:
        winkler_threshold = metrics_df['winkler_score'].quantile(0.1)
        top_10_models = metrics_df[metrics_df['winkler_score'] <= winkler_threshold]['model_name'].unique()
        cols_to_plot_final = [col for col in base_cols if col in top_10_models]
        if not cols_to_plot_final:
            st.warning(f"No models in top 10% (threshold: {winkler_threshold:.4f})")

    start_idx, end_idx = date_range
    sub_df = df.iloc[start_idx: end_idx + 1]

    cols_to_plot = []
    for col in cols_to_plot_final:
        m, s, sd = col_meta.get(col, (None, None, None))
        if selected_model != 'All' and m != selected_model:
            continue
        if selected_scheme != 'All' and s != selected_scheme:
            continue
        if selected_seed != 'All' and sd != selected_seed:
            continue
        cols_to_plot.append(col)

    # --- Plot ---
    st.subheader("📈 Interactive Visualization")
    with st.spinner("Generating plot..."):
        if show_rolling and len(cols_to_plot) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                            gridspec_kw={'height_ratios': [2, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(16, 7))

        ax1.plot(sub_df.index, sub_df['actual'],
                 label='Actual Data', color='black', linewidth=2.5, alpha=0.9, zorder=100)

        colors = sns.color_palette("bright", max(len(cols_to_plot), 1))

        if not cols_to_plot:
            ax1.text(0.5, 0.5, "No models match these filters",
                     ha='center', transform=ax1.transAxes, fontsize=14)
        else:
            ax2_picp = ax2_nmpiw = None
            for i, col in enumerate(cols_to_plot):
                winkler_score = winkler_map.get(col, None)
                label = f"{col} (W: {winkler_score:.4f})" if winkler_score else col
                ax1.plot(sub_df.index, sub_df[col], label=label,
                         color=colors[i], linewidth=2, alpha=1.0)

                if show_band:
                    col_L, col_U = f"{col}_L", f"{col}_U"
                    if col_L in sub_df.columns and col_U in sub_df.columns:
                        ax1.fill_between(sub_df.index, sub_df[col_L], sub_df[col_U],
                                         color=colors[i], alpha=0.2, label='_nolegend_')

                        if show_coverage:
                            actual_vals = sub_df['actual']
                            inside_mask = (actual_vals >= sub_df[col_L]) & (actual_vals <= sub_df[col_U])
                            outside_mask = ~inside_mask
                            if inside_mask.any():
                                ax1.plot(sub_df.index[inside_mask], actual_vals[inside_mask],
                                         marker='o', markersize=5, color='limegreen',
                                         linestyle='None', alpha=0.7, zorder=200)
                            if outside_mask.any():
                                ax1.plot(sub_df.index[outside_mask], actual_vals[outside_mask],
                                         marker='x', markersize=7, color='red',
                                         linestyle='None', alpha=0.9, zorder=200)

                if show_rolling:
                    col_L, col_U = f"{col}_L", f"{col}_U"
                    if col_L in df.columns and col_U in df.columns:
                        first_valid_idx = df[col].first_valid_index()
                        if first_valid_idx is not None:
                            df_valid = df[df.index >= first_valid_idx].copy()
                            actual_vals = df_valid['actual']
                            lower_bounds = df_valid[col_L]
                            upper_bounds = df_valid[col_U]

                            coverage = ((actual_vals >= lower_bounds) & (actual_vals <= upper_bounds)).astype(int)
                            rolling_picp = coverage.rolling(window=30).mean()
                            rolling_mpiw = (upper_bounds - lower_bounds).rolling(window=30).mean()
                            actual_range = actual_vals.max() - actual_vals.min()
                            rolling_nmpiw = rolling_mpiw / actual_range if actual_range > 0 else rolling_mpiw

                            if ax2_picp is None:
                                ax2_picp = ax2
                                ax2_nmpiw = ax2.twinx()
                                ax2_picp.axhline(y=0.95, color='green', linestyle='--',
                                                 linewidth=2, alpha=0.7, label='95% CI')
                                ax2_picp.axhline(y=0.90, color='orange', linestyle='--',
                                                 linewidth=2, alpha=0.7, label='90% CI')
                                ax2_picp.set_ylabel('PICP (Coverage)', fontsize=10)
                                ax2_nmpiw.set_ylabel('n-MPIW (Width)', fontsize=10)
                                ax2_picp.legend(loc='upper left', fontsize=9)

                            picp_color = tuple(c * 0.8 for c in colors[i][:3]) if isinstance(colors[i], tuple) else colors[i]
                            ax2_picp.plot(rolling_picp.index, rolling_picp,
                                          label=f"{col} PICP", color=picp_color,
                                          linewidth=2, alpha=0.85)
                            ax2_nmpiw.plot(rolling_nmpiw.index, rolling_nmpiw,
                                           label=f"{col} n-MPIW", color=colors[i],
                                           linewidth=2, linestyle='--', alpha=0.85)

        if show_events and len(sub_df) > 0:
            draw_events(ax1, sub_df)

        ax1.set_title(f"Uncertainty Quantification Analysis: {len(cols_to_plot)} Models",
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Prediction Value', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)

        if len(cols_to_plot) <= 8:
            ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        else:
            ax1.text(1.02, 0.95, f"Legend hidden\n({len(cols_to_plot)} models)\nFilter to < 8",
                     transform=ax1.transAxes, fontsize=10, verticalalignment='top')

        if show_rolling and len(cols_to_plot) > 0:
            ax2.set_xlabel('Date', fontsize=11)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.set_title('Rolling 30-Day Metrics', fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)

    # --- Info Panel ---
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", len(df))
    with c2:
        st.metric("Available Models", len(sorted_models))
    with c3:
        st.metric("UQ Schemes", len(sorted_schemes))
    with c4:
        st.metric("Seeds", len(sorted_seeds))

    if st.checkbox("Show sample data", key='uq_sample'):
        uq_min_date = df.index.min().date()
        uq_max_date = df.index.max().date()
        uq_preview_range = st.date_input(
            "Preview date range:",
            value=(uq_min_date, uq_max_date),
            min_value=uq_min_date,
            max_value=uq_max_date,
            key='uq_preview_range'
        )
        if isinstance(uq_preview_range, (list, tuple)) and len(uq_preview_range) == 2:
            uq_preview = df.loc[str(uq_preview_range[0]):str(uq_preview_range[1])]
        else:
            uq_preview = df
        st.dataframe(uq_preview, use_container_width=True)

    if st.checkbox("Show model list", key='uq_model_list'):
        st.write("**Model Coverage Summary:**")
        rows = []
        for col, (m, s, sd) in col_meta.items():
            rows.append({'Architecture': m.upper(), 'UQ Scheme': s.upper(), 'Seed': sd})
        uq_summary = pd.DataFrame(rows)

        n_arch   = uq_summary['Architecture'].nunique()
        n_scheme = uq_summary['UQ Scheme'].nunique()
        n_seeds  = uq_summary['Seed'].nunique()
        total    = len(uq_summary)

        uc1, uc2, uc3, uc4 = st.columns(4)
        uc1.metric("Architectures", n_arch)
        uc2.metric("UQ Schemes", n_scheme)
        uc3.metric("Seeds", n_seeds)
        uc4.metric("Total Runs", total)

        st.write("**Architectures:**", " | ".join(sorted(uq_summary['Architecture'].unique())))
        st.write("**UQ Schemes:**", " | ".join(sorted(uq_summary['UQ Scheme'].unique())))
        st.write("**Seeds:**", " | ".join(sorted(uq_summary['Seed'].unique(),
                                                   key=lambda x: int(x) if x.isdigit() else x)))

        pivot_uq = uq_summary.groupby(['UQ Scheme', 'Architecture'])['Seed'].apply(
            lambda s: ', '.join(sorted(s, key=lambda x: int(x) if x.isdigit() else x))
        ).unstack(fill_value='—')
        st.write("**Runs Matrix (UQ Scheme × Architecture → Seeds):**")
        st.dataframe(pivot_uq, use_container_width=True)


# ==================================================================================
# PAGE 3 — POINT FORECAST TRAINING RESULTS
# ==================================================================================
elif page == "📋 PF Training Results":
    st.title("📋 Point Forecast — Training Results")
    st.markdown(
        "Tabular summary of all **4 architectures × 3 tuning scenarios × 5 seeds = 60 runs**."
    )
    st.markdown("---")

    _, pf_metrics_df = load_pf_data()
    if pf_metrics_df is None or pf_metrics_df.empty:
        st.error("❌ ALL_METRICS.csv not available.")
        st.stop()

    df_pf = enrich_metrics(pf_metrics_df, 'Tuning')
    pf_metric_cols = ['r2_score', 'rmse', 'mae', 'mape']

    tab1, tab2, tab3 = st.tabs(["📋 Full Results", "📊 Scenario Summary", "🎯 Convergence Ranking"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            pfr_arch = st.multiselect(
                "Architecture:", sorted(df_pf['Architecture'].unique()),
                default=sorted(df_pf['Architecture'].unique()), key='pfr_arch')
        with c2:
            pfr_tuning = st.multiselect(
                "Tuning:", sorted(df_pf['Tuning'].unique()),
                default=sorted(df_pf['Tuning'].unique()), key='pfr_tuning')
        with c3:
            pfr_primary = st.selectbox(
                "Primary metric (for 🥇/✅):",
                list(PF_METRIC_DIRECTIONS.keys()), index=0, key='pfr_primary')

        hi = PF_METRIC_DIRECTIONS[pfr_primary] == 'max'
        subset = df_pf[
            df_pf['Architecture'].isin(pfr_arch) & df_pf['Tuning'].isin(pfr_tuning)
        ].copy()
        subset = add_best_markers(subset, pfr_primary, 'Tuning', hi)

        disp_cols = ['Best', 'Architecture', 'Tuning', 'Seed',
                     'r2_score', 'rmse', 'mae', 'mape', 'mse', 'training_time_s']
        disp_cols = [c for c in disp_cols if c in subset.columns]
        show_df = subset[disp_cols].sort_values(
            ['Tuning', 'Architecture', 'Seed']).reset_index(drop=True)

        m_dir = {k: v for k, v in PF_METRIC_DIRECTIONS.items() if k in show_df.columns}
        st.dataframe(apply_gradient_style(show_df, m_dir), use_container_width=True, height=520)
        st.caption(
            "🥇 = Global best  |  ✅ = Best within tuning scenario  |  "
            "Color: 🟢 best → 🔴 worst (scale clipped to 5th–95th percentile)")

    with tab2:
        ctl1, ctl2 = st.columns([1, 2])
        with ctl1:
            pfr_perspective = st.radio(
                "👁 Group perspective:",
                ["By Tuning Scenario", "By Architecture"],
                horizontal=True, key='pfr_perspective')
        with ctl2:
            pfr_mode = st.radio(
                "Display mode:",
                ["Mean only", "Mean ± Std", "Full stats (mean, std, CV%)"],
                horizontal=True, key='pfr_mode')

        pf_group_col   = 'Tuning'       if pfr_perspective == "By Tuning Scenario" else 'Architecture'
        pf_compare_col = 'Architecture' if pf_group_col == 'Tuning'               else 'Tuning'

        conv_pf = build_convergence_table(df_pf, 'Tuning', pf_metric_cols)
        conv_pf = conv_pf.sort_values([pf_group_col, pf_compare_col]).reset_index(drop=True)

        def _pf_display(conv, mode):
            if mode == "Mean only":
                mc = [f'{c}_mean' for c in pf_metric_cols if f'{c}_mean' in conv.columns]
                d  = conv[['Architecture', 'Tuning', 'N Seeds'] + mc].copy()
                d.columns = [c.replace('_mean', '') for c in d.columns]
                return d
            elif mode == "Mean ± Std":
                rows = []
                for _, row in conv.iterrows():
                    r = {'Architecture': row['Architecture'], 'Tuning': row['Tuning'],
                         'N Seeds': int(row['N Seeds'])}
                    for col in pf_metric_cols:
                        mn = row.get(f'{col}_mean', np.nan)
                        sd = row.get(f'{col}_std', np.nan)
                        r[col] = f"{mn:.4f} ± {sd:.4f}" if pd.notna(mn) and pd.notna(sd) else '—'
                    rows.append(r)
                return pd.DataFrame(rows)
            else:
                return conv.copy()

        disp_pf = _pf_display(conv_pf, pfr_mode)
        st.dataframe(
            apply_dual_highlight(disp_pf, conv_pf, pf_metric_cols, PF_METRIC_DIRECTIONS, pf_group_col),
            use_container_width=True
        )
        st.caption(
            "🔵 Best mean  |  🔴 Lowest std (most consistent)  |  🟣 Both  "
            "— highlighted per metric, within each group.  "
            "Aggregated across 5 seeds per (Architecture, Tuning) group."
        )

        st.write(f"**Breakdown by {pf_group_col}:**")
        for group_val, grp in conv_pf.groupby(pf_group_col):
            with st.expander(f"{pf_group_col}: {group_val}"):
                grp_r    = grp.reset_index(drop=True)
                disp_exp = _pf_display(grp_r, pfr_mode)
                st.dataframe(
                    apply_dual_highlight(disp_exp, grp_r, pf_metric_cols,
                                         PF_METRIC_DIRECTIONS, None),
                    use_container_width=True
                )

    with tab3:
        st.write("**Lower Std / CV% = More consistent across seeds = Better seed convergence**")
        pfr_conv_met = st.selectbox(
            "Rank convergence by metric:", pf_metric_cols, key='pfr_conv_met')
        std_c = f'{pfr_conv_met}_std'
        mean_c = f'{pfr_conv_met}_mean'
        cv_c = f'{pfr_conv_met}_cv%'

        conv_pf3 = build_convergence_table(df_pf, 'Tuning', pf_metric_cols)
        if std_c in conv_pf3.columns:
            rank_cols = [c for c in ['Architecture', 'Tuning', 'N Seeds', mean_c, std_c, cv_c]
                         if c in conv_pf3.columns]
            rank_df = conv_pf3[rank_cols].sort_values(std_c, ascending=True).reset_index(drop=True)
            rank_df.insert(0, 'Rank', range(1, len(rank_df) + 1))

            fmt3 = {mean_c: '{:.4f}', std_c: '{:.4f}', cv_c: '{:.2f}'}
            s3 = rank_df.style.format(
                {k: v for k, v in fmt3.items() if k in rank_df.columns}, na_rep='—')
            s3 = s3.background_gradient(cmap='RdYlGn_r', subset=[std_c], axis=0)
            if cv_c in rank_df.columns:
                s3 = s3.background_gradient(cmap='RdYlGn_r', subset=[cv_c], axis=0)

            st.dataframe(s3, use_container_width=True)
            st.caption(
                f"Sorted by Std({pfr_conv_met}) ascending.  "
                "CV% = Std / |Mean| × 100.  🟢 = most stable (lowest spread across seeds).")
        else:
            st.warning(f"Metric '{pfr_conv_met}' not found in grouped table.")


# ==================================================================================
# PAGE 4 — UQ TRAINING RESULTS
# ==================================================================================
elif page == "📋 UQ Training Results":
    st.title("📋 UQ Methods — Training Results")
    st.markdown(
        "Tabular summary of all **4 architectures × 3 UQ methods × 5 seeds = 60 runs**."
    )
    st.markdown("---")

    _, uq_metrics_df, _ = load_uq_data()
    if uq_metrics_df is None or uq_metrics_df.empty:
        st.error("❌ ALL_UQ_METRICS.csv not available.")
        st.stop()

    df_uq = enrich_metrics(uq_metrics_df, 'UQ Method')
    uq_metric_cols = ['winkler_score', 'picp', 'mpiw', 'r2_score', 'rmse']

    # Detect potential failed/diverged runs
    outlier_mask = (df_uq['r2_score'] < -5) | (df_uq['winkler_score'] > 1e5)
    n_outliers = int(outlier_mask.sum())
    if n_outliers > 0:
        with st.expander(
                f"⚠️ {n_outliers} potentially failed / diverged run(s) detected — click to inspect",
                expanded=False):
            st.dataframe(
                df_uq[outlier_mask][['Architecture', 'UQ Method', 'Seed',
                                     'r2_score', 'winkler_score', 'rmse']],
                use_container_width=True)
            st.caption(
                "These runs likely diverged during training. They are included in all tables "
                "but the color gradient is clipped (5th–95th percentile) to prevent them from "
                "masking differences among well-trained models.")

    tab1, tab2, tab3 = st.tabs(["📋 Full Results", "📊 Scenario Summary", "🎯 Convergence Ranking"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            uqr_arch = st.multiselect(
                "Architecture:", sorted(df_uq['Architecture'].unique()),
                default=sorted(df_uq['Architecture'].unique()), key='uqr_arch')
        with c2:
            uqr_method = st.multiselect(
                "UQ Method:", sorted(df_uq['UQ Method'].unique()),
                default=sorted(df_uq['UQ Method'].unique()), key='uqr_method')
        with c3:
            uqr_primary = st.selectbox(
                "Primary metric (for 🥇/✅):",
                list(UQ_METRIC_DIRECTIONS.keys()), index=0, key='uqr_primary')

        hi_uq = UQ_METRIC_DIRECTIONS[uqr_primary] == 'max'
        subset_uq = df_uq[
            df_uq['Architecture'].isin(uqr_arch) & df_uq['UQ Method'].isin(uqr_method)
        ].copy()
        subset_uq = add_best_markers(subset_uq, uqr_primary, 'UQ Method', hi_uq)

        disp_cols_uq = ['Best', 'Architecture', 'UQ Method', 'Seed',
                        'winkler_score', 'picp', 'mpiw', 'r2_score', 'rmse', 'mae', 'training_time_s']
        disp_cols_uq = [c for c in disp_cols_uq if c in subset_uq.columns]
        show_uq = subset_uq[disp_cols_uq].sort_values(
            ['UQ Method', 'Architecture', 'Seed']).reset_index(drop=True)

        m_dir_uq = {k: v for k, v in UQ_METRIC_DIRECTIONS.items() if k in show_uq.columns}
        st.dataframe(apply_gradient_style(show_uq, m_dir_uq), use_container_width=True, height=520)
        st.caption(
            "🥇 = Global best  |  ✅ = Best within UQ method scenario  |  "
            "Color: 🟢 best → 🔴 worst (scale clipped to 5th–95th percentile)")

    with tab2:
        ctl1_uq, ctl2_uq = st.columns([1, 2])
        with ctl1_uq:
            uqr_perspective = st.radio(
                "👁 Group perspective:",
                ["By UQ Method", "By Architecture"],
                horizontal=True, key='uqr_perspective')
        with ctl2_uq:
            uqr_mode = st.radio(
                "Display mode:",
                ["Mean only", "Mean ± Std", "Full stats (mean, std, CV%)"],
                horizontal=True, key='uqr_mode')

        uq_group_col   = 'UQ Method'    if uqr_perspective == "By UQ Method"  else 'Architecture'
        uq_compare_col = 'Architecture' if uq_group_col    == 'UQ Method'      else 'UQ Method'

        conv_uq = build_convergence_table(df_uq, 'UQ Method', uq_metric_cols)
        conv_uq = conv_uq.sort_values([uq_group_col, uq_compare_col]).reset_index(drop=True)

        def _uq_display(conv, mode):
            if mode == "Mean only":
                mc = [f'{c}_mean' for c in uq_metric_cols if f'{c}_mean' in conv.columns]
                d  = conv[['Architecture', 'UQ Method', 'N Seeds'] + mc].copy()
                d.columns = [c.replace('_mean', '') for c in d.columns]
                return d
            elif mode == "Mean ± Std":
                rows = []
                for _, row in conv.iterrows():
                    r = {'Architecture': row['Architecture'], 'UQ Method': row['UQ Method'],
                         'N Seeds': int(row['N Seeds'])}
                    for col in uq_metric_cols:
                        mn = row.get(f'{col}_mean', np.nan)
                        sd = row.get(f'{col}_std', np.nan)
                        r[col] = f"{mn:.4f} ± {sd:.4f}" if pd.notna(mn) and pd.notna(sd) else '—'
                    rows.append(r)
                return pd.DataFrame(rows)
            else:
                return conv.copy()

        disp_uq = _uq_display(conv_uq, uqr_mode)
        st.dataframe(
            apply_dual_highlight(disp_uq, conv_uq, uq_metric_cols, UQ_METRIC_DIRECTIONS, uq_group_col),
            use_container_width=True
        )
        st.caption(
            "🔵 Best mean  |  🔴 Lowest std (most consistent)  |  🟣 Both  "
            "— highlighted per metric, within each group.  "
            "Aggregated across 5 seeds per (Architecture, UQ Method) group."
        )

        st.write(f"**Breakdown by {uq_group_col}:**")
        for group_val, grp in conv_uq.groupby(uq_group_col):
            with st.expander(f"{uq_group_col}: {group_val}"):
                grp_r    = grp.reset_index(drop=True)
                disp_exp = _uq_display(grp_r, uqr_mode)
                st.dataframe(
                    apply_dual_highlight(disp_exp, grp_r, uq_metric_cols,
                                         UQ_METRIC_DIRECTIONS, None),
                    use_container_width=True
                )

    with tab3:
        st.write("**Lower Std / CV% = More consistent across seeds = Better seed convergence**")
        uqr_conv_met = st.selectbox(
            "Rank convergence by metric:", uq_metric_cols, key='uqr_conv_met')
        std_c_uq = f'{uqr_conv_met}_std'
        mean_c_uq = f'{uqr_conv_met}_mean'
        cv_c_uq = f'{uqr_conv_met}_cv%'

        conv_uq3 = build_convergence_table(df_uq, 'UQ Method', uq_metric_cols)
        if std_c_uq in conv_uq3.columns:
            rank_cols_uq = [c for c in
                            ['Architecture', 'UQ Method', 'N Seeds', mean_c_uq, std_c_uq, cv_c_uq]
                            if c in conv_uq3.columns]
            rank_uq = conv_uq3[rank_cols_uq].sort_values(
                std_c_uq, ascending=True).reset_index(drop=True)
            rank_uq.insert(0, 'Rank', range(1, len(rank_uq) + 1))

            fmt_uq = {mean_c_uq: '{:.4f}', std_c_uq: '{:.4f}', cv_c_uq: '{:.2f}'}
            s_uq = rank_uq.style.format(
                {k: v for k, v in fmt_uq.items() if k in rank_uq.columns}, na_rep='—')
            s_uq = s_uq.background_gradient(cmap='RdYlGn_r', subset=[std_c_uq], axis=0)
            if cv_c_uq in rank_uq.columns:
                s_uq = s_uq.background_gradient(cmap='RdYlGn_r', subset=[cv_c_uq], axis=0)

            st.dataframe(s_uq, use_container_width=True)
            st.caption(
                f"Sorted by Std({uqr_conv_met}) ascending.  "
                "CV% = Std / |Mean| × 100.  🟢 = most stable.  "
                "Note: diverged runs inflate Std for some groups.")
        else:
            st.warning(f"Metric '{uqr_conv_met}' not found in grouped table.")


# ==================================================================================
# PAGE 5 — DIEBOLD-MARIANO SIGNIFICANCE TEST (RS vs Optuna)
# ==================================================================================
elif page == "🔬 DM Significance Test":
    st.title("🔬 Diebold-Mariano Significance Test")
    st.markdown(
        "Pairwise DM tests comparing **Random Search (RS)** vs **Optuna** predictions "
        "across all architectures and seeds. A positive DM statistic means Optuna's "
        "loss differential is smaller (Optuna is better); negative means RS is better."
    )
    st.markdown("---")

    # ---- DM test function ----
    def _dm_test(y_true, y_pred1, y_pred2, loss_func='squared_error'):
        """Return dm_statistic and p_value for the two-sided DM test."""
        y_true  = np.array(y_true,  dtype=float)
        y_pred1 = np.array(y_pred1, dtype=float)
        y_pred2 = np.array(y_pred2, dtype=float)
        if loss_func == 'squared_error':
            e1 = (y_true - y_pred1) ** 2
            e2 = (y_true - y_pred2) ** 2
        else:
            e1 = np.abs(y_true - y_pred1)
            e2 = np.abs(y_true - y_pred2)
        d = e1 - e2
        n = len(d)
        mean_d = np.mean(d)
        var_d  = np.var(d, ddof=1)
        c0     = var_d / n
        dm_stat  = mean_d / np.sqrt(c0) if c0 > 0 else 0.0
        p_value  = 2 * (1 - norm.cdf(abs(dm_stat)))
        return float(dm_stat), float(p_value)

    # ---- Load & parse data ----
    @st.cache_data
    def run_dm_tests(loss_func='squared_error'):
        try:
            df = pd.read_csv('ALL_PREDICTED.csv')
        except FileNotFoundError:
            return None, "❌ 'ALL_PREDICTED.csv' not found."

        actual_col = 'actual' if 'actual' in df.columns else df.columns[0]
        y_actual   = df[actual_col].values.astype(float)

        model_cols = [c for c in df.columns if c not in (actual_col, 'Unnamed: 0')]

        # parse {arch}_{tuning}_{seed}
        parsed = {}
        for col in model_cols:
            parts = col.split('_')
            if len(parts) >= 3:
                arch, tuning, seed = parts[0].lower(), parts[1].lower(), parts[2]
                parsed[(arch, tuning, seed)] = col

        archs = sorted({k[0] for k in parsed})
        seeds = sorted({k[2] for k in parsed})

        results = []
        for arch in archs:
            for seed in seeds:
                rs_key     = (arch, 'rs',     seed)
                optuna_key = (arch, 'optuna', seed)
                if rs_key not in parsed or optuna_key not in parsed:
                    continue
                col_rs     = parsed[rs_key]
                col_optuna = parsed[optuna_key]
                y_rs     = df[col_rs].values.astype(float)
                y_optuna = df[col_optuna].values.astype(float)
                mask = ~(np.isnan(y_actual) | np.isnan(y_rs) | np.isnan(y_optuna))
                if mask.sum() < 5:
                    continue
                dm_stat, p_val = _dm_test(y_actual[mask], y_rs[mask], y_optuna[mask], loss_func)
                better = 'Optuna' if dm_stat > 0 else 'RS'
                results.append({
                    'Architecture': arch.upper(),
                    'Seed': seed,
                    'RS Column': col_rs,
                    'Optuna Column': col_optuna,
                    'DM Statistic': dm_stat,
                    'P-value': p_val,
                    'Significant': p_val < 0.05,
                    'Better Model': better,
                })
        return pd.DataFrame(results), None

    # ---- Sidebar controls ----
    st.sidebar.header("⚙️ DM Test Options")
    dm_loss = st.sidebar.radio(
        "Loss function:",
        ["squared_error", "absolute_error"],
        format_func=lambda x: "Squared Error (MSE)" if x == "squared_error" else "Absolute Error (MAE)",
        key='dm_loss'
    )
    dm_alpha = st.sidebar.selectbox(
        "Significance level α:", [0.01, 0.05, 0.10], index=1, key='dm_alpha'
    )

    dm_df, err_msg = run_dm_tests(dm_loss)

    if err_msg:
        st.error(err_msg)
        st.stop()
    if dm_df is None or dm_df.empty:
        st.warning("No RS vs Optuna comparison pairs found in the dataset.")
        st.stop()

    # Re-evaluate significance with chosen alpha
    dm_df = dm_df.copy()
    dm_df['Significant'] = dm_df['P-value'] < dm_alpha

    total_tests = len(dm_df)
    sig_tests   = dm_df['Significant'].sum()
    archs_found = sorted(dm_df['Architecture'].unique())

    # ---- Top KPI row ----
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Pairs Tested",  total_tests)
    kpi2.metric(f"Significant (p < {dm_alpha})", int(sig_tests))
    kpi3.metric("Architectures", len(archs_found))
    kpi4.metric("Seeds", dm_df['Seed'].nunique())

    st.markdown("---")

    # ---- Tabs ----
    tab_detail, tab_group, tab_overall = st.tabs(
        ["📋 Detailed Results", "📊 Grouped by Architecture", "🏆 Overall Summary"]
    )

    # ---- Tab 1: Detailed results ----
    with tab_detail:
        st.subheader("All Pairwise DM Test Results")

        c1, c2 = st.columns(2)
        with c1:
            arch_filter = st.multiselect(
                "Filter Architecture:", archs_found, default=archs_found, key='dm_arch_filter'
            )
        with c2:
            sig_filter = st.selectbox(
                "Show:", ["All", "Significant only", "Non-significant only"], key='dm_sig_filter'
            )

        detail_df = dm_df[dm_df['Architecture'].isin(arch_filter)].copy()
        if sig_filter == "Significant only":
            detail_df = detail_df[detail_df['Significant']]
        elif sig_filter == "Non-significant only":
            detail_df = detail_df[~detail_df['Significant']]

        def _fmt_pvalue(v):
            return f"{v:.2e}" if v < 0.001 else f"{v:.4f}"

        detail_show = detail_df[['Architecture', 'Seed', 'RS Column', 'Optuna Column',
                                  'DM Statistic', 'P-value', 'Significant', 'Better Model']].copy()
        detail_show['P-value'] = detail_show['P-value'].apply(_fmt_pvalue)
        detail_show['DM Statistic'] = detail_show['DM Statistic'].apply(lambda x: f"{x:.4f}")
        detail_show['Significant'] = detail_show['Significant'].map({True: '✅ Yes', False: '❌ No'})
        detail_show = detail_show.sort_values(['Architecture', 'Seed']).reset_index(drop=True)

        def _row_style(row):
            is_sig = row['Significant'] == '✅ Yes'
            winner = row['Better Model']
            style = ''
            if is_sig and winner == 'Optuna':
                style = 'background-color: #1a7a3c; color: #ffffff; font-weight: bold'  # dark green
            elif is_sig and winner == 'RS':
                style = 'background-color: #b22222; color: #ffffff; font-weight: bold'  # dark red
            return [style] * len(row)

        st.dataframe(
            detail_show.style.apply(_row_style, axis=1),
            use_container_width=True, height=480
        )
        st.caption(
            f"🟢 Significant & Optuna better  |  🔴 Significant & RS better  |  "
            f"α = {dm_alpha}  |  Loss: {dm_loss.replace('_', ' ').title()}  |  "
            "Positive DM stat → Optuna has lower loss"
        )

    # ---- Tab 2: Grouped by architecture ----
    with tab_group:
        st.subheader("Significance Wins Grouped by Architecture")
        st.markdown(
            f"Counts are based on **significant** pairs only (p < {dm_alpha}). "
            "A 'win' means that model has significantly lower forecast loss."
        )

        group_records = []
        for arch, grp in dm_df.groupby('Architecture'):
            total_arch    = len(grp)
            sig_arch      = grp['Significant'].sum()
            not_sig_arch  = total_arch - sig_arch
            sig_grp       = grp[grp['Significant']]
            optuna_wins   = (sig_grp['Better Model'] == 'Optuna').sum()
            rs_wins       = (sig_grp['Better Model'] == 'RS').sum()
            if sig_arch > 0:
                winner = 'Optuna' if optuna_wins > rs_wins else ('RS' if rs_wins > optuna_wins else 'Tie')
            else:
                winner = '—'
            group_records.append({
                'Architecture'    : arch,
                'Total Tests'     : int(total_arch),
                f'Significant (p<{dm_alpha})': int(sig_arch),
                'Not Significant' : int(not_sig_arch),
                'Optuna Wins'     : int(optuna_wins),
                'RS Wins'         : int(rs_wins),
                'Sig. Rate %'     : round(100 * sig_arch / total_arch, 1) if total_arch else 0.0,
                'Arch. Winner'    : winner,
            })

        group_df = pd.DataFrame(group_records).sort_values('Architecture').reset_index(drop=True)

        def _arch_style(row):
            winner = row['Arch. Winner']
            if winner == 'Optuna':
                return ['background-color: #1a7a3c; color: #ffffff; font-weight: bold'] * len(row)
            elif winner == 'RS':
                return ['background-color: #b22222; color: #ffffff; font-weight: bold'] * len(row)
            else:
                return [''] * len(row)

        st.dataframe(
            group_df.style.apply(_arch_style, axis=1),
            use_container_width=True
        )
        st.caption("🟢 Optuna wins the architecture  |  🔴 RS wins the architecture")

        # Per-architecture bar chart
        st.markdown("#### Win Counts per Architecture")
        fig_bar, ax_bar = plt.subplots(figsize=(max(6, len(archs_found) * 1.5), 4))
        x = np.arange(len(group_df))
        width = 0.35
        bars1 = ax_bar.bar(x - width/2, group_df['Optuna Wins'], width,
                           label='Optuna Wins', color='#4caf50', alpha=0.85)
        bars2 = ax_bar.bar(x + width/2, group_df['RS Wins'],     width,
                           label='RS Wins',     color='#f44336', alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(group_df['Architecture'], fontsize=11)
        ax_bar.set_ylabel('Number of Significant Wins')
        ax_bar.set_title(f'Significant DM Test Wins per Architecture (α={dm_alpha})',
                         fontweight='bold')
        ax_bar.legend()
        ax_bar.bar_label(bars1, padding=2)
        ax_bar.bar_label(bars2, padding=2)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_bar)

        # Expandable seed-level detail per architecture
        st.markdown("#### Seed-level Detail per Architecture")
        for arch in archs_found:
            arch_detail = dm_df[dm_df['Architecture'] == arch].copy()
            arch_detail_show = arch_detail[['Seed', 'DM Statistic', 'P-value',
                                            'Significant', 'Better Model']].copy()
            arch_detail_show['P-value']      = arch_detail_show['P-value'].apply(_fmt_pvalue)
            arch_detail_show['DM Statistic'] = arch_detail_show['DM Statistic'].apply(lambda x: f"{x:.4f}")
            arch_detail_show['Significant']  = arch_detail_show['Significant'].map({True: '✅ Yes', False: '❌ No'})
            arch_detail_show = arch_detail_show.sort_values('Seed').reset_index(drop=True)
            sig_n   = arch_detail['Significant'].sum()
            total_n = len(arch_detail)
            with st.expander(f"{arch}  —  {sig_n}/{total_n} significant"):
                st.dataframe(
                    arch_detail_show.style.apply(_row_style, axis=1),
                    use_container_width=True
                )

    # ---- Tab 3: Overall summary ----
    with tab_overall:
        st.subheader("Overall Summary")

        overall_sig     = dm_df[dm_df['Significant']]
        total_optuna    = (overall_sig['Better Model'] == 'Optuna').sum()
        total_rs        = (overall_sig['Better Model'] == 'RS').sum()
        overall_winner  = (
            'Optuna' if total_optuna > total_rs else
            ('RS'    if total_rs > total_optuna  else 'Tie')
        )
        non_sig_count   = (~dm_df['Significant']).sum()

        ov1, ov2, ov3, ov4, ov5 = st.columns(5)
        ov1.metric("Total Tests",         total_tests)
        ov2.metric(f"Sig. (p<{dm_alpha})", int(sig_tests))
        ov3.metric("Optuna Wins",          int(total_optuna))
        ov4.metric("RS Wins",              int(total_rs))
        ov5.metric("Overall Winner",       overall_winner)

        st.markdown("---")

        # Pie chart of significant wins
        if int(total_optuna) + int(total_rs) > 0:
            fig_pie, axes_pie = plt.subplots(1, 2, figsize=(12, 4))

            # Left: significant breakdown
            labels_sig  = []
            sizes_sig   = []
            colors_sig  = []
            if total_optuna > 0:
                labels_sig.append(f'Optuna ({total_optuna})')
                sizes_sig.append(total_optuna)
                colors_sig.append('#4caf50')
            if total_rs > 0:
                labels_sig.append(f'RS ({total_rs})')
                sizes_sig.append(total_rs)
                colors_sig.append('#f44336')
            if non_sig_count > 0:
                labels_sig.append(f'Not Significant ({non_sig_count})')
                sizes_sig.append(non_sig_count)
                colors_sig.append('#bdbdbd')

            axes_pie[0].pie(sizes_sig, labels=labels_sig, colors=colors_sig,
                            autopct='%1.1f%%', startangle=140)
            axes_pie[0].set_title(f'Overall Distribution\n(α={dm_alpha})', fontweight='bold')

            # Right: per-architecture stacked bar
            arch_names    = group_df['Architecture'].tolist()
            optuna_counts = group_df['Optuna Wins'].tolist()
            rs_counts     = group_df['RS Wins'].tolist()
            not_sig_cnts  = group_df['Not Significant'].tolist()
            x2 = np.arange(len(arch_names))
            axes_pie[1].bar(x2, optuna_counts, label='Optuna Wins',     color='#4caf50', alpha=0.85)
            axes_pie[1].bar(x2, rs_counts,     label='RS Wins',         color='#f44336', alpha=0.85,
                            bottom=optuna_counts)
            axes_pie[1].bar(x2, not_sig_cnts,  label='Not Significant', color='#bdbdbd', alpha=0.7,
                            bottom=[o + r for o, r in zip(optuna_counts, rs_counts)])
            axes_pie[1].set_xticks(x2)
            axes_pie[1].set_xticklabels(arch_names, fontsize=11)
            axes_pie[1].set_ylabel('Number of Tests')
            axes_pie[1].set_title('Tests per Architecture', fontweight='bold')
            axes_pie[1].legend(loc='upper right', fontsize=9)
            axes_pie[1].grid(axis='y', linestyle='--', alpha=0.4)

            plt.tight_layout()
            st.pyplot(fig_pie)
        else:
            st.info("No significant differences found — cannot draw win pie chart.")

        st.markdown("---")
        st.subheader("Architecture-level Winner Table")
        winner_tbl = group_df[['Architecture', 'Total Tests',
                                f'Significant (p<{dm_alpha})',
                                'Optuna Wins', 'RS Wins',
                                'Sig. Rate %', 'Arch. Winner']].copy()
        st.dataframe(
            winner_tbl.style.apply(_arch_style, axis=1),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### Interpretation Guide")
        st.markdown(
            f"""
| Symbol | Meaning |
|--------|---------|
| **DM Statistic > 0** | Optuna has lower forecast loss (Optuna is better) |
| **DM Statistic < 0** | RS has lower forecast loss (RS is better) |
| **p < {dm_alpha}** | Difference is statistically significant at α={dm_alpha} |
| **Optuna Wins** | Significant tests where Optuna's loss is smaller |
| **RS Wins** | Significant tests where RS's loss is smaller |
| **Not Significant** | No statistically meaningful difference between RS and Optuna |
"""
        )
        st.caption(
            f"Loss function: **{dm_loss.replace('_', ' ').title()}**  |  "
            "Two-sided DM test with sample variance estimator.  "
            "Results are cached — change the loss function in the sidebar to refresh."
        )


# ==================================================================================
# PAGE 6 — EDA STATISTICS
# ==================================================================================
elif page == "🔎 EDA Statistics":
    st.title("🔎 Exploratory Data Analysis — Statistical Tests")
    st.markdown(
        "Summary of pre-modelling statistical analyses: descriptive statistics, outlier detection, "
        "normality tests, and stationarity tests for all 14 input variables."
    )
    st.markdown("---")

    # ---- Missing-data mapping from EDA notebook ----
    MISSING_PCT = {
        'Nickel_Fut':          33.38,
        'Coal_Fut_Newcastle':  29.41,
        'Palm_Oil_Fut':        33.36,
        'USD_IDR':             30.07,
        'CNY_IDR':             28.57,
        'EUR_IDR':             28.57,
        'BTC_USD':              0.00,
        'FTSE100':             30.87,
        'HANGSENG':            32.70,
        'NIKKEI225':           33.14,
        'SNP500':              31.14,
        'DOW30':               31.14,
        'SSE_Composite':       33.52,
        'JKSE':                33.88,
    }

    # ---- Display name mapping ----
    DISPLAY_NAMES = {
        'Nickel_Fut':         'Nickel',
        'Coal_Fut_Newcastle': 'Coal (Newcastle)',
        'Palm_Oil_Fut':       'Palm Oil (CPO)',
        'USD_IDR':            'USD/IDR',
        'CNY_IDR':            'CNY/IDR',
        'EUR_IDR':            'EUR/IDR',
        'BTC_USD':            'Bitcoin (BTC/USD)',
        'FTSE100':            'FTSE 100',
        'HANGSENG':           'Hang Seng',
        'NIKKEI225':          'Nikkei 225',
        'SNP500':             'S&P 500',
        'DOW30':              'Dow Jones 30',
        'SSE_Composite':      'SSE Composite',
        'JKSE':               'JKSE (IDX)',
    }

    # ---- Data loaders ----
    @st.cache_data
    def load_eda_csvs():
        base = '.'
        five    = pd.read_csv(os.path.join(base, 'Five_Number_Summary.csv'),     index_col=0)
        outlier = pd.read_csv(os.path.join(base, 'Outlier_Analysis_Summary.csv'))
        normal  = pd.read_csv(os.path.join(base, 'Normality_Tests_Results.csv'),  index_col=0)
        adf     = pd.read_csv(os.path.join(base, 'ADF_Test_Results.csv'),          index_col=0)
        return five, outlier, normal, adf

    @st.cache_data
    def load_merged_csv():
        df = pd.read_csv(os.path.join('.', 'ALL_MERGED.csv'))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    @st.cache_data
    def load_corr_matrix():
        return pd.read_csv(os.path.join('.', 'Correlation_Matrix.csv'), index_col=0)

    @st.cache_data
    def load_feature_analysis():
        return pd.read_csv(os.path.join('.', 'Feature_Analysis_Results.csv'))

    try:
        five_df, outlier_df, normal_df, adf_df = load_eda_csvs()
    except FileNotFoundError as e:
        st.error(f"❌ Could not load EDA CSV: {e}")
        st.stop()

    try:
        merged_df = load_merged_csv()
    except FileNotFoundError:
        merged_df = None

    try:
        corr_matrix_df = load_corr_matrix()
    except FileNotFoundError:
        corr_matrix_df = None

    try:
        feat_df = load_feature_analysis()
    except FileNotFoundError:
        feat_df = None

    # numeric feature columns
    FEAT_COLS = [
        'Nickel_Fut', 'Coal_Fut_Newcastle', 'Palm_Oil_Fut',
        'USD_IDR', 'CNY_IDR', 'EUR_IDR', 'BTC_USD',
        'FTSE100', 'HANGSENG', 'NIKKEI225', 'SNP500', 'DOW30',
        'SSE_Composite', 'JKSE',
    ]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📐 Five-Number Summary",
        "📦 Outlier Analysis",
        "🔔 Normality Tests",
        "📉 Stationarity (ADF)",
        "🔗 Correlation",
        "⚛️ Causality Tests",
        "🧪 Further Tests",
    ])

    # ──────────────────────────────────────────────
    # TAB 1 — Five-Number Summary
    # ──────────────────────────────────────────────
    with tab1:
        st.subheader("Five-Number Summary + Extended Statistics")
        st.markdown(
            "Descriptive statistics for each variable computed over the full dataset period."
        )

        # ---- Extended stats from ALL_MERGED.csv ----
        if merged_df is not None:
            ext_records = []
            for col in FEAT_COLS:
                if col not in merged_df.columns:
                    continue
                s = merged_df[col].dropna()
                ext_records.append({
                    'Variable':  DISPLAY_NAMES.get(col, col),
                    'Min':       s.min(),
                    'Q1 (25%)':  s.quantile(0.25),
                    'Median':    s.median(),
                    'Q3 (75%)':  s.quantile(0.75),
                    'Max':       s.max(),
                    'Variance':  s.var(ddof=1),
                    'Std Dev':   s.std(ddof=1),
                    'Skewness':  s.skew(),
                    'Kurtosis':  s.kurt(),
                })
            disp5 = pd.DataFrame(ext_records)
        else:
            disp5 = five_df.copy()
            disp5.index = [DISPLAY_NAMES.get(i, i) for i in disp5.index]
            disp5.index.name = "Variable"
            disp5 = disp5.reset_index()

        num_cols5 = [c for c in disp5.columns if c != 'Variable']
        fmt5 = {c: '{:,.2f}' for c in num_cols5}
        grad_cols = ['Min', 'Q1 (25%)', 'Median', 'Q3 (75%)', 'Max',
                     'Variance', 'Std Dev']
        grad_cols = [c for c in grad_cols if c in disp5.columns]

        styler5 = disp5.style.format(fmt5).background_gradient(
            cmap='Blues', subset=grad_cols, axis=0
        )
        # skewness: diverging colormap centred on 0
        if 'Skewness' in disp5.columns:
            styler5 = styler5.background_gradient(cmap='RdBu_r', subset=['Skewness'], axis=0)
        if 'Kurtosis' in disp5.columns:
            styler5 = styler5.background_gradient(cmap='YlOrRd', subset=['Kurtosis'], axis=0)

        st.dataframe(styler5, use_container_width=True, height=(len(disp5) + 1) * 35 + 3)
        st.caption(
            "Blue gradient = magnitude  |  "
            "Red↔Blue = Skewness (red = right-skewed, blue = left-skewed)  |  "
            "Yellow→Red = Kurtosis (redder = heavier tails)"
        )

        # ---- Time-series plot ----
        if merged_df is not None:
            st.markdown("---")
            st.subheader("📈 Time-Series Plot")

            ts_sel = st.multiselect(
                "Select variables to plot:",
                options=FEAT_COLS,
                default=FEAT_COLS[:4],
                format_func=lambda x: DISPLAY_NAMES.get(x, x),
                key='eda_ts_sel'
            )
            normalize_ts = st.checkbox(
                "Standardize (Z-score / StandardScaler) for easier comparison",
                value=True, key='eda_ts_norm'
            )

            if ts_sel:
                plot_df = merged_df[ts_sel].copy()
                if normalize_ts:
                    plot_df = (plot_df - plot_df.mean()) / plot_df.std()

                n_vars = len(ts_sel)
                colors_ts = sns.color_palette("tab10", n_vars)
                fig_ts, ax_ts = plt.subplots(figsize=(14, 5))
                for i, col in enumerate(ts_sel):
                    ax_ts.plot(
                        plot_df.index, plot_df[col],
                        label=DISPLAY_NAMES.get(col, col),
                        color=colors_ts[i], linewidth=1.4, alpha=0.85
                    )
                ax_ts.set_xlabel('Date', fontsize=11)
                ax_ts.set_ylabel('Standardized Value (z-score)' if normalize_ts else 'Value', fontsize=11)
                ax_ts.set_title('Time-Series of Selected Variables', fontweight='bold')
                ax_ts.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
                ax_ts.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig_ts)
            else:
                st.info("Select at least one variable to plot.")

        # ---- Range / IQR chart ----
        st.markdown("---")
        st.markdown("#### Range & IQR per Variable")
        range_src = merged_df[FEAT_COLS] if merged_df is not None else None
        if range_src is not None:
            range_df = pd.DataFrame({
                'Variable': [DISPLAY_NAMES.get(c, c) for c in FEAT_COLS],
                'Range': range_src.max() - range_src.min(),
                'IQR':   range_src.quantile(0.75) - range_src.quantile(0.25),
            }).reset_index(drop=True)
        else:
            range_df = pd.DataFrame({
                'Variable': [DISPLAY_NAMES.get(i, i) for i in five_df.index],
                'Range': five_df['Max'] - five_df['Min'],
                'IQR':   five_df['Q3 (75%)'] - five_df['Q1 (25%)'],
            }).reset_index(drop=True)

        fig_r, ax_r = plt.subplots(figsize=(12, 4))
        x_r = np.arange(len(range_df))
        ax_r.bar(x_r - 0.2, range_df['Range'], 0.4, label='Range (Max−Min)', color='#1f77b4', alpha=0.85)
        ax_r.bar(x_r + 0.2, range_df['IQR'],   0.4, label='IQR (Q3−Q1)',     color='#ff7f0e', alpha=0.85)
        ax_r.set_xticks(x_r)
        ax_r.set_xticklabels(range_df['Variable'], rotation=35, ha='right', fontsize=9)
        ax_r.set_ylabel('Value')
        ax_r.set_title('Range & IQR per Variable', fontweight='bold')
        ax_r.legend()
        ax_r.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_r)

    # ──────────────────────────────────────────────
    # TAB 2 — Outlier Analysis
    # ──────────────────────────────────────────────
    with tab2:
        st.subheader("Outlier Analysis & Missing Data")
        st.markdown(
            "Outliers detected via IQR method. Missing data percentages derived from "
            "the original dataset before imputation."
        )

        out_df = outlier_df.copy()
        out_df['Missing %'] = out_df['Variable'].map(MISSING_PCT)
        out_df['Variable']  = out_df['Variable'].map(lambda x: DISPLAY_NAMES.get(x, x))
        out_df = out_df.rename(columns={
            'Number of Outliers':      'Outlier Count',
            'Percentage of Outliers (%)': 'Outlier %',
            'Distribution Type':       'Distribution',
        })

        col_order = ['Variable', 'Missing %', 'Outlier Count', 'Outlier %', 'Distribution']
        out_df = out_df[[c for c in col_order if c in out_df.columns]].reset_index(drop=True)

        def _outlier_style(val, col):
            if col == 'Missing %':
                if val > 30:
                    return 'background-color: #b22222; color: white; font-weight: bold'
                elif val > 15:
                    return 'background-color: #e07b00; color: white'
                elif val == 0:
                    return 'background-color: #1a7a3c; color: white'
                return ''
            if col == 'Outlier %':
                if val > 5:
                    return 'background-color: #b22222; color: white; font-weight: bold'
                elif val > 2:
                    return 'background-color: #e07b00; color: white'
                elif val == 0:
                    return 'background-color: #1a7a3c; color: white'
                return ''
            return ''

        styler_out = out_df.style.format({'Missing %': '{:.2f}%', 'Outlier %': '{:.2f}%', 'Outlier Count': '{:,}'})
        for col in ['Missing %', 'Outlier %']:
            if col in out_df.columns:
                styler_out = styler_out.applymap(lambda v, c=col: _outlier_style(v, c), subset=[col])

        st.dataframe(styler_out, use_container_width=True, height=(len(out_df) + 1) * 35 + 3)
        st.caption(
            "🟥 > 30% missing / > 5% outliers  |  "
            "🟧 15–30% missing / 2–5% outliers  |  "
            "🟩 0% (no missing / no outliers)"
        )

        st.markdown("#### Missing % vs Outlier % per Variable")
        fig_o, ax_o = plt.subplots(figsize=(12, 4))
        x_o = np.arange(len(out_df))
        ax_o.bar(x_o - 0.2, out_df['Missing %'], 0.4, label='Missing %',  color='#d62728', alpha=0.85)
        ax_o.bar(x_o + 0.2, out_df['Outlier %'], 0.4, label='Outlier %',  color='#ff7f0e', alpha=0.85)
        ax_o.set_xticks(x_o)
        ax_o.set_xticklabels(out_df['Variable'], rotation=35, ha='right', fontsize=9)
        ax_o.set_ylabel('%')
        ax_o.set_title('Missing Data % and Outlier % per Variable', fontweight='bold')
        ax_o.legend()
        ax_o.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_o)

    # ──────────────────────────────────────────────
    # TAB 3 — Normality Tests
    # ──────────────────────────────────────────────
    with tab3:
        st.subheader("Normality Tests")
        st.markdown(
            "Two tests applied: **Shapiro-Wilk** and **Kolmogorov-Smirnov (KS)**. "
            "H₀: the data follows a normal distribution."
        )

        norm_disp = normal_df.copy()
        norm_disp.index = [DISPLAY_NAMES.get(i, i) for i in norm_disp.index]
        norm_disp.index.name = "Variable"
        norm_disp = norm_disp.reset_index()

        rename_norm = {
            'Shapiro-Wilk Statistic':    'SW Stat',
            'Shapiro-Wilk p-value':      'SW p-value',
            'Shapiro-Wilk Conclusion':   'SW Result',
            'KS Statistic':              'KS Stat',
            'KS p-value':                'KS p-value',
            'KS Conclusion':             'KS Result',
        }
        norm_disp = norm_disp.rename(columns=rename_norm)

        def _norm_result_style(val):
            if 'Reject' in str(val):
                return 'background-color: #b22222; color: white; font-weight: bold'
            elif 'Accept' in str(val):
                return 'background-color: #1a7a3c; color: white; font-weight: bold'
            return ''

        fmt_norm = {
            'SW Stat':     '{:.4f}',
            'SW p-value':  '{:.2e}',
            'KS Stat':     '{:.4f}',
            'KS p-value':  '{:.2e}',
        }
        result_cols = [c for c in ['SW Result', 'KS Result'] if c in norm_disp.columns]
        styler_norm = norm_disp.style.format({k: v for k, v in fmt_norm.items() if k in norm_disp.columns})
        for rc in result_cols:
            styler_norm = styler_norm.applymap(_norm_result_style, subset=[rc])

        st.dataframe(styler_norm, use_container_width=True, height=(len(norm_disp) + 1) * 35 + 3)
        st.caption(
            "🟥 Reject H₀ = data is NOT normal  |  🟩 Accept H₀ = data is normal  |  "
            "α = 0.05"
        )

        sw_rej  = normal_df['Shapiro-Wilk Conclusion'].str.contains('Reject').sum()
        ks_rej  = normal_df['KS Conclusion'].str.contains('Reject').sum()
        total_n = len(normal_df)
        n1, n2, n3 = st.columns(3)
        n1.metric("Variables Tested", total_n)
        n2.metric("SW: Reject H₀", int(sw_rej))
        n3.metric("KS: Reject H₀", int(ks_rej))

    # ──────────────────────────────────────────────
    # TAB 4 — Stationarity (ADF)
    # ──────────────────────────────────────────────
    with tab4:
        st.subheader("Stationarity Test — Augmented Dickey-Fuller (ADF)")
        st.markdown(
            "H₀: the time series has a unit root (non-stationary). "
            "Rejection means the series is stationary."
        )

        adf_disp = adf_df.copy()
        adf_disp.index = [DISPLAY_NAMES.get(i, i) for i in adf_disp.index]
        adf_disp.index.name = "Variable"
        adf_disp = adf_disp.reset_index()

        rename_adf = {
            'ADF Statistic':        'ADF Stat',
            'p-value':              'p-value',
            'Critical Value (5%)':  'Crit. Value (5%)',
            'Conclusion':           'Result',
            'Lags Used':            'Lags',
            'Observations Used':    'Obs.',
        }
        adf_disp = adf_disp.rename(columns=rename_adf)

        def _adf_result_style(val):
            if 'Reject' in str(val):
                return 'background-color: #1a7a3c; color: white; font-weight: bold'
            elif 'Accept' in str(val):
                return 'background-color: #b22222; color: white; font-weight: bold'
            return ''

        fmt_adf = {
            'ADF Stat':         '{:.4f}',
            'p-value':          '{:.4e}',
            'Crit. Value (5%)': '{:.4f}',
        }
        styler_adf = adf_disp.style.format({k: v for k, v in fmt_adf.items() if k in adf_disp.columns})
        if 'Result' in adf_disp.columns:
            styler_adf = styler_adf.applymap(_adf_result_style, subset=['Result'])

        st.dataframe(styler_adf, use_container_width=True, height=(len(adf_disp) + 1) * 35 + 3)
        st.caption(
            "🟩 Reject H₀ = Stationary  |  🟥 Accept H₀ = Non-Stationary  |  "
            "Significance level α = 5%"
        )

        stationary_n    = adf_df['Conclusion'].str.contains('Reject').sum()
        nonstationary_n = adf_df['Conclusion'].str.contains('Accept').sum()
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Variables", len(adf_df))
        a2.metric("Stationary",      int(stationary_n))
        a3.metric("Non-Stationary",  int(nonstationary_n))

        st.markdown("#### ADF Statistic vs Critical Value (5%)")
        fig_adf, ax_adf = plt.subplots(figsize=(12, 4))
        x_a      = np.arange(len(adf_disp))
        adf_vals = adf_df['ADF Statistic'].values
        crit_vals = adf_df['Critical Value (5%)'].values
        colors_adf = ['#1a7a3c' if v < c else '#b22222'
                      for v, c in zip(adf_vals, crit_vals)]
        ax_adf.bar(x_a, adf_vals, color=colors_adf, alpha=0.85, label='ADF Statistic')
        ax_adf.plot(x_a, crit_vals, color='black', linestyle='--',
                    linewidth=2, marker='o', markersize=5, label='Critical Value (5%)')
        ax_adf.set_xticks(x_a)
        ax_adf.set_xticklabels(adf_disp['Variable'], rotation=35, ha='right', fontsize=9)
        ax_adf.set_ylabel('Statistic Value')
        ax_adf.set_title('ADF Statistic vs Critical Value per Variable', fontweight='bold')
        ax_adf.legend()
        ax_adf.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_adf)
        st.caption("🟩 Bar below dashed line = Stationary  |  🟥 Bar above = Non-Stationary")

    # ──────────────────────────────────────────────
    # TAB 5 — Correlation
    # ──────────────────────────────────────────────
    with tab5:
        st.subheader("Correlation Analysis — Pearson & Spearman")
        st.markdown(
            "Pearson measures **linear** correlation; Spearman captures **monotonic** (rank-based) "
            "relationships and is more robust to outliers and non-normality."
        )

        if corr_matrix_df is None or feat_df is None:
            st.warning("Correlation data files not found.")
        else:
            # ── 1 & 2. Pearson and Spearman heatmaps side by side ──────────
            col_ph, col_sp = st.columns(2)

            with col_ph:
                st.markdown("#### Pearson Correlation Matrix")
                corr_display = corr_matrix_df.copy()
                corr_display.index   = [DISPLAY_NAMES.get(i, i) for i in corr_display.index]
                corr_display.columns = [DISPLAY_NAMES.get(c, c) for c in corr_display.columns]

                fig_ph, ax_ph = plt.subplots(figsize=(8, 7))
                im = ax_ph.imshow(corr_display.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
                plt.colorbar(im, ax=ax_ph, shrink=0.8)
                ax_ph.set_xticks(range(len(corr_display.columns)))
                ax_ph.set_yticks(range(len(corr_display.index)))
                ax_ph.set_xticklabels(corr_display.columns, rotation=45, ha='right', fontsize=7)
                ax_ph.set_yticklabels(corr_display.index, fontsize=7)
                for i in range(len(corr_display.index)):
                    for j in range(len(corr_display.columns)):
                        val = corr_display.values[i, j]
                        ax_ph.text(j, i, f'{val:.2f}', ha='center', va='center',
                                   fontsize=6, color='black' if abs(val) < 0.7 else 'white')
                ax_ph.set_title('Pearson Correlation Matrix', fontweight='bold', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_ph)

            with col_sp:
                st.markdown("#### Spearman Correlation Matrix")
                if merged_df is not None:
                    spearman_full = merged_df[FEAT_COLS].corr(method='spearman')
                    spear_display = spearman_full.copy()
                    spear_display.index   = [DISPLAY_NAMES.get(i, i) for i in spear_display.index]
                    spear_display.columns = [DISPLAY_NAMES.get(c, c) for c in spear_display.columns]

                    fig_sp, ax_sp = plt.subplots(figsize=(8, 7))
                    im2 = ax_sp.imshow(spear_display.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
                    plt.colorbar(im2, ax=ax_sp, shrink=0.8)
                    ax_sp.set_xticks(range(len(spear_display.columns)))
                    ax_sp.set_yticks(range(len(spear_display.index)))
                    ax_sp.set_xticklabels(spear_display.columns, rotation=45, ha='right', fontsize=7)
                    ax_sp.set_yticklabels(spear_display.index, fontsize=7)
                    for i in range(len(spear_display.index)):
                        for j in range(len(spear_display.columns)):
                            val = spear_display.values[i, j]
                            ax_sp.text(j, i, f'{val:.2f}', ha='center', va='center',
                                       fontsize=6, color='black' if abs(val) < 0.7 else 'white')
                    ax_sp.set_title('Spearman Correlation Matrix', fontweight='bold', fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig_sp)
                else:
                    st.info("ALL_MERGED.csv not found — Spearman matrix unavailable.")

            # ── 3. Pearson vs Spearman bar chart vs JKSE ───────────────────
            st.markdown("#### Pearson vs Spearman Correlation with JKSE (Target)")
            feat_corr = feat_df.copy()
            feat_corr['Feature_Label'] = feat_corr['Feature'].map(
                lambda x: DISPLAY_NAMES.get(x, x))

            x_fc = np.arange(len(feat_corr))
            width = 0.38
            fig_fc, ax_fc = plt.subplots(figsize=(12, 5))
            bars_p = ax_fc.bar(x_fc - width/2, feat_corr['Pearson_Corr'],  width,
                               label='Pearson',  color='#2c7bb6', alpha=0.85)
            bars_s = ax_fc.bar(x_fc + width/2, feat_corr['Spearman_Corr'], width,
                               label='Spearman', color='#d7191c', alpha=0.85)
            ax_fc.set_xticks(x_fc)
            ax_fc.set_xticklabels(feat_corr['Feature_Label'], rotation=35, ha='right', fontsize=9)
            ax_fc.set_ylabel('Correlation Coefficient')
            ax_fc.set_title('Pearson vs Spearman Correlation with JKSE', fontweight='bold')
            ax_fc.axhline(0, color='black', linewidth=0.8)
            ax_fc.legend()
            ax_fc.grid(axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_fc)

            # ── 4. Summary table ────────────────────────────────────────────
            st.markdown("#### Feature-JKSE Correlation Table")
            corr_tbl = feat_corr[['Feature_Label', 'Pearson_Corr', 'Pearson_Pval',
                                   'Spearman_Corr', 'Spearman_Pval']].copy()
            corr_tbl.columns = ['Feature', 'Pearson r', 'Pearson p', 'Spearman ρ', 'Spearman p']

            def _sig_style(val):
                try:
                    return ('background-color: #1a7a3c; color: white; font-weight: bold'
                            if float(val) < 0.05 else '')
                except (ValueError, TypeError):
                    return ''

            styler_ct = (corr_tbl.style
                         .format({'Pearson r': '{:.4f}', 'Pearson p': '{:.4e}',
                                  'Spearman ρ': '{:.4f}', 'Spearman p': '{:.4e}'})
                         .applymap(_sig_style, subset=['Pearson p', 'Spearman p']))
            st.dataframe(styler_ct, use_container_width=True,
                         height=(len(corr_tbl) + 1) * 35 + 3)
            st.caption("🟩 p < 0.05 = statistically significant correlation")

    # ──────────────────────────────────────────────
    # TAB 6 — Causality Tests
    # ──────────────────────────────────────────────
    with tab6:
        st.subheader("Causality Tests — Granger Causality & Transfer Entropy")
        st.markdown(
            "**Granger Causality**: tests whether past values of a feature statistically improve "
            "the prediction of JKSE (H₀: no Granger causality, α = 0.05).  \n"
            "**Transfer Entropy**: measures the asymmetric information flow from each feature to JKSE "
            "(higher = more directional information transfer)."
        )

        if feat_df is None:
            st.warning("Feature_Analysis_Results.csv not found.")
        else:
            feat_ca = feat_df.copy()
            feat_ca['Feature_Label'] = feat_ca['Feature'].map(
                lambda x: DISPLAY_NAMES.get(x, x))

            col_left, col_right = st.columns(2)

            # ── Granger Causality ───────────────────────────────────────────
            with col_left:
                st.markdown("#### Granger Causality")

                granger_tbl = feat_ca[['Feature_Label', 'Granger_Pval', 'Granger_Lag']].copy()
                granger_tbl.columns = ['Feature', 'p-value', 'Optimal Lag']
                granger_tbl = granger_tbl.sort_values('p-value')

                def _granger_style(val):
                    try:
                        return ('background-color: #1a7a3c; color: white; font-weight: bold'
                                if float(val) < 0.05 else
                                'background-color: #b22222; color: white; font-weight: bold')
                    except (ValueError, TypeError):
                        return ''

                styler_gc = (granger_tbl.style
                             .format({'p-value': '{:.4e}', 'Optimal Lag': '{:.0f}'})
                             .applymap(_granger_style, subset=['p-value']))
                st.dataframe(styler_gc, use_container_width=True,
                             height=(len(granger_tbl) + 1) * 35 + 3)
                st.caption("🟩 p < 0.05 = Granger causes JKSE  |  🟥 Not significant")

                # Bar chart: p-values
                fig_gc, ax_gc = plt.subplots(figsize=(6, 5))
                colors_gc = ['#1a7a3c' if p < 0.05 else '#b22222'
                             for p in granger_tbl['p-value']]
                ax_gc.barh(granger_tbl['Feature'], granger_tbl['p-value'],
                           color=colors_gc, alpha=0.85)
                ax_gc.axvline(0.05, color='black', linestyle='--', linewidth=1.5,
                              label='α = 0.05')
                ax_gc.set_xlabel('p-value')
                ax_gc.set_title('Granger Causality p-values\n(→ JKSE)', fontweight='bold')
                ax_gc.invert_yaxis()
                ax_gc.legend(fontsize=8)
                ax_gc.grid(axis='x', linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig_gc)
                _gc_path = os.path.join('.', 'Granger_Causality_Matrix.png')
                if os.path.exists(_gc_path):
                    st.image(_gc_path, use_container_width=True)

            # ── Transfer Entropy ────────────────────────────────────────────
            with col_right:
                st.markdown("#### Transfer Entropy")

                te_tbl = feat_ca[['Feature_Label', 'TE_Score']].copy()
                te_tbl.columns = ['Feature', 'TE Score']
                te_tbl = te_tbl.sort_values('TE Score', ascending=False)

                def _te_style(val):
                    try:
                        v = float(val)
                        th = feat_ca['TE_Score'].median()
                        return ('background-color: #1a7a3c; color: white; font-weight: bold'
                                if v >= th else '')
                    except (ValueError, TypeError):
                        return ''

                styler_te = (te_tbl.style
                             .format({'TE Score': '{:.6f}'})
                             .applymap(_te_style, subset=['TE Score']))
                st.dataframe(styler_te, use_container_width=True,
                             height=(len(te_tbl) + 1) * 35 + 3)
                st.caption("🟩 TE ≥ median = above-average information flow to JKSE")

                # Bar chart: TE scores
                te_med = feat_ca['TE_Score'].median()
                colors_te = ['#1a7a3c' if v >= te_med else '#2c7bb6'
                             for v in te_tbl['TE Score']]
                fig_te, ax_te = plt.subplots(figsize=(6, 5))
                ax_te.barh(te_tbl['Feature'], te_tbl['TE Score'],
                           color=colors_te, alpha=0.85)
                ax_te.axvline(te_med, color='black', linestyle='--', linewidth=1.5,
                              label=f'Median = {te_med:.5f}')
                ax_te.set_xlabel('Transfer Entropy Score')
                ax_te.set_title('Transfer Entropy\n(→ JKSE)', fontweight='bold')
                ax_te.invert_yaxis()
                ax_te.legend(fontsize=8)
                ax_te.grid(axis='x', linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig_te)
                _te_path = os.path.join('.', 'Transfer_Entropy_Matrix.png')
                if os.path.exists(_te_path):
                    st.image(_te_path, use_container_width=True)

            # ── KPI summary ─────────────────────────────────────────────────
            sig_n = (feat_ca['Granger_Pval'] < 0.05).sum()
            top_te = feat_ca.loc[feat_ca['TE_Score'].idxmax(), 'Feature_Label']
            k1, k2, k3 = st.columns(3)
            k1.metric("Granger-Significant Features", int(sig_n))
            k2.metric("Top TE Feature", top_te)
            k3.metric("Median TE Score", f"{te_med:.6f}")

    # ──────────────────────────────────────────────
    # TAB 7 — Further Tests
    # ──────────────────────────────────────────────
    with tab7:
        st.subheader("Further Dependency Tests — Mutual Information & Distance Correlation")
        st.markdown(
            "**Mutual Information (MI)**: measures the total statistical dependency (including "
            "non-linear) between a feature and JKSE.  \n"
            "**Distance Correlation (dCor)**: detects both linear and non-linear associations; "
            "dCor = 0 implies independence, unlike Pearson r."
        )

        if feat_df is None:
            st.warning("Feature_Analysis_Results.csv not found.")
        else:
            feat_ft = feat_df.copy()
            feat_ft['Feature_Label'] = feat_ft['Feature'].map(
                lambda x: DISPLAY_NAMES.get(x, x))
            feat_ft = feat_ft.sort_values('MI_Score', ascending=False)

            col_a, col_b = st.columns(2)

            # ── Mutual Information ──────────────────────────────────────────
            with col_a:
                st.markdown("#### Mutual Information with JKSE")

                mi_tbl = feat_ft[['Feature_Label', 'MI_Score']].copy()
                mi_tbl.columns = ['Feature', 'MI Score']

                mi_med = feat_ft['MI_Score'].median()

                def _mi_style(val):
                    try:
                        return ('background-color: #1a7a3c; color: white; font-weight: bold'
                                if float(val) >= mi_med else '')
                    except (ValueError, TypeError):
                        return ''

                styler_mi = (mi_tbl.style
                             .format({'MI Score': '{:.6f}'})
                             .applymap(_mi_style, subset=['MI Score']))
                st.dataframe(styler_mi, use_container_width=True,
                             height=(len(mi_tbl) + 1) * 35 + 3)
                st.caption("🟩 MI ≥ median = above-average non-linear dependency with JKSE")

                colors_mi = ['#1a7a3c' if v >= mi_med else '#2c7bb6'
                             for v in mi_tbl['MI Score']]
                fig_mi, ax_mi = plt.subplots(figsize=(6, 5))
                ax_mi.barh(mi_tbl['Feature'], mi_tbl['MI Score'],
                           color=colors_mi, alpha=0.85)
                ax_mi.axvline(mi_med, color='black', linestyle='--', linewidth=1.5,
                              label=f'Median = {mi_med:.5f}')
                ax_mi.set_xlabel('Mutual Information Score')
                ax_mi.set_title('Mutual Information\n(→ JKSE)', fontweight='bold')
                ax_mi.invert_yaxis()
                ax_mi.legend(fontsize=8)
                ax_mi.grid(axis='x', linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig_mi)

            # ── Distance Correlation ────────────────────────────────────────
            with col_b:
                st.markdown("#### Distance Correlation (dCor) with JKSE")

                feat_dcor = feat_ft.sort_values('dCor', ascending=False)
                dcor_tbl = feat_dcor[['Feature_Label', 'dCor']].copy()
                dcor_tbl.columns = ['Feature', 'dCor']

                dcor_med = feat_ft['dCor'].median()

                def _dcor_style(val):
                    try:
                        return ('background-color: #1a7a3c; color: white; font-weight: bold'
                                if float(val) >= dcor_med else '')
                    except (ValueError, TypeError):
                        return ''

                styler_dc = (dcor_tbl.style
                             .format({'dCor': '{:.6f}'})
                             .applymap(_dcor_style, subset=['dCor']))
                st.dataframe(styler_dc, use_container_width=True,
                             height=(len(dcor_tbl) + 1) * 35 + 3)

                _pvs_path = os.path.join('.', 'Feature_Analysis_PearsonVsSpearman.png')
                if os.path.exists(_pvs_path):
                    st.image(_pvs_path, use_container_width=True)
                else:
                    st.info("Feature_Analysis_PearsonVsSpearman.png not found.")

            # ── Combined table ──────────────────────────────────────────────
            st.markdown("#### Combined MI & dCor Summary Table")
            combined_tbl = feat_ft[['Feature_Label', 'MI_Score', 'dCor']].copy()
            combined_tbl.columns = ['Feature', 'MI Score', 'dCor']

            def _both_style(val):
                try:
                    return ('background-color: #1a7a3c; color: white; font-weight: bold'
                            if float(val) >= combined_tbl[combined_tbl.columns[
                                combined_tbl.isin([val]).any(axis=1)
                            ]].median().iloc[0] else '')
                except Exception:
                    return ''

            styler_comb = (combined_tbl.style
                           .format({'MI Score': '{:.6f}', 'dCor': '{:.6f}'}))
            st.dataframe(styler_comb, use_container_width=True,
                         height=(len(combined_tbl) + 1) * 35 + 3)

            # ── KPIs ────────────────────────────────────────────────────────
            top_mi   = feat_ft.loc[feat_ft['MI_Score'].idxmax(), 'Feature_Label']
            top_dcor = feat_ft.loc[feat_ft['dCor'].idxmax(), 'Feature_Label']
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Top MI Feature",    top_mi)
            m2.metric("Max MI Score",      f"{feat_ft['MI_Score'].max():.6f}")
            m3.metric("Top dCor Feature",  top_dcor)
            m4.metric("Max dCor",          f"{feat_ft['dCor'].max():.6f}")

# ==================================================================================
# PAGE 7 — UNCERTAINTY DECOMPOSITION COMPARISON
# ==================================================================================
elif page == "🔀 Uncertainty Decomposition":
    st.title("🔀 Uncertainty Decomposition Comparison")
    st.markdown(
        "Visual comparison of predictive interval decomposition across the three "
        "uncertainty quantification methods: **Monte Carlo Dropout (MCD)**, "
        "**Heteroscedastic Log-Likelihood Loss (HLLLA)**, and "
        "**Conformal Quantile Regression (CQR)**."
    )

    _img_dir = '.'

    st.markdown("### Monte Carlo Dropout (MCD)")
    _mcd_path = os.path.join(_img_dir, 'GRU MCD.png')
    if os.path.exists(_mcd_path):
        st.image(_mcd_path, use_container_width=True)
    else:
        st.warning("GRU MCD.png not found.")

    st.markdown("---")
    st.markdown("### Heteroscedastic Log-Likelihood Loss (HLLLA)")
    _hllla_path = os.path.join(_img_dir, 'LSTM HLLLA.png')
    if os.path.exists(_hllla_path):
        st.image(_hllla_path, use_container_width=True)
    else:
        st.warning("LSTM HLLLA.png not found.")

    st.markdown("---")
    st.markdown("### Conformal Quantile Regression (CQR)")
    _cqr_path = os.path.join(_img_dir, 'TCN CQR.png')
    if os.path.exists(_cqr_path):
        st.image(_cqr_path, use_container_width=True)
    else:
        st.warning("TCN CQR.png not found.")

