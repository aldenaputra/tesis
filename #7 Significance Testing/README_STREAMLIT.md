# Streamlit UQ Analysis Dashboard

Quick guide to running your Streamlit web app locally.

## Installation

### Step 1: Install dependencies
Open PowerShell in this directory and run:
```
pip install -r requirements.txt
```

### Step 2: Run the app
```
streamlit run app_streamlit.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Features

✅ **Interactive Filters** - Model, Scheme, Seed selection from sidebars
✅ **Uncertainty Bands** - Visualize prediction bounds
✅ **Band Coverage** - Green dots (inside) vs Red X (outside)
✅ **Rolling Metrics** - PICP & n-MPIW over 30-day windows
✅ **Geopolitical Events** - Annotated timeline markers
✅ **Date Range Zoom** - Slider to focus on specific periods
✅ **Top Performers** - Filter by top 10% Winkler scores
✅ **Data Info** - Summary statistics & sample data preview

## How to Use

1. **Sidebar Controls** (left side):
   - Select Model, UQ Scheme, Seed
   - Toggle display options
   - Adjust date range slider

2. **Main Plot** (center):
   - Black line = Actual data
   - Colored lines = Model predictions
   - Shaded bands = Uncertainty bounds

3. **Coverage Markers**:
   - 🟢 Green dot = Point inside uncertainty band
   - ❌ Red X = Point outside uncertainty band

4. **Rolling Metrics** (bottom panel when enabled):
   - Shows coverage probability (PICP)
   - Shows normalized width (n-MPIW)

## For Your Thesis Defense

- **Local presentation**: Run locally, share screen
- **Remote presentation**: Deploy to Streamlit Cloud
  - Push code to GitHub
  - Connect repo to streamlit.io/cloud
  - Share public URL with committee

## Key Differences from Jupyter

| Jupyter | Streamlit |
|---------|-----------|
| ipywidgets | st.selectbox, st.slider, st.checkbox |
| plt.show() | st.pyplot(fig) |
| Manual reruns | Auto reruns on widget change |
| Sidebar N/A | st.sidebar for controls |

## Troubleshooting

**"Cannot find CSV file"**
- Make sure ALL_UQ_PREDICTED.csv is in the same directory as app_streamlit.py
- Check that '../Results/ALL_UQ_METRICS.csv' path is correct

**App is slow**
- Streamlit caches data with @st.cache_data
- First run loads data, subsequent runs use cache
- Clear cache: Press 'c' key in Streamlit terminal

**Want custom styling?**
- Edit colors in the events list (hex codes like '#FF6B6B')
- Adjust figure sizes in plot_uq_series() function
- Change plot heights with figsize parameter

Enjoy! 🎉
