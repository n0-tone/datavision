# DataVision

DataVision is a password-protected Streamlit analytics app for CSV exploration, data quality auditing, and machine learning visual analysis.

## What It Includes

- Secure login using Streamlit Secrets.
- Dark JetBrains Mono design system.
- Dataset overview with live filtering.
- Advanced statistics (mean, median, std, skewness, kurtosis, IQR, outlier counts).
- Data quality checks (missing values, duplicate rows, datatype profile, missing heatmap).
- Relationship analysis (correlation heatmap, top correlated pairs, scatter matrix).
- Clustering (K-Means, elbow curve, silhouette score, cluster profiles, PCA projection).
- PCA analysis (explained variance and projections).
- Feature importance with Random Forest (classification/regression).
- Time series analysis (date parsing, resampling, rolling trend).
- Auto insights summary.

## Project Structure

- `streamlit.py`: primary Streamlit entrypoint.
- `main.py`: secondary entrypoint that runs the same modular app.
- `src/dashboard/`: modular app logic.
- `src/components/sidebar.py`: shared sidebar rendering.
- `data/`: sample datasets.

## Local Run

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Set credentials as environment variables.

PowerShell:

```powershell
$env:APP_USERNAME = "admin"
$env:APP_PASSWORD = "change-me"
```

CMD:

```bat
set APP_USERNAME=admin
set APP_PASSWORD=change-me
```

3. Start the app.

```bash
streamlit run streamlit.py
```

## TOML Snippets

Use these snippets where needed (for example Streamlit Cloud settings, or your own local secret file if you choose to create one).

Secrets TOML:

```toml
APP_USERNAME = "admin"
APP_PASSWORD = "change-me"
```

Theme TOML:

```toml
[theme]
base = "dark"
primaryColor = "#39e0c0"
backgroundColor = "#05070c"
secondaryBackgroundColor = "#0f1726"
textColor = "#e6edf8"
font = "monospace"

[browser]
gatherUsageStats = false
```

## Streamlit Cloud Deploy

1. Set Python to 3.11.
2. Set entrypoint to `streamlit.py`.
3. In Streamlit Cloud Secrets, paste:

```toml
APP_USERNAME = "admin"
APP_PASSWORD = "your-strong-password"
```

## Notes

- This repo no longer stores a `.streamlit/` folder.
- Authentication reads Streamlit Secrets first, then falls back to environment variables.
- Theme styling is implemented in `src/dashboard/theme.css` and can be complemented by TOML theme values in your deploy environment.
