# Thesis Scraper & Analysis

A small toolkit to scrape BINUS graduation pages for thesis titles and run simple text analysis (word clouds, keyword counts, clustering).

## Prerequisites
- Python 3.9+ (Windows supported)
- Recommended: a virtual environment

## Project Layout
```
/Thesis-Scraper
├── /data
│     ├── <csv files>
├── analysis.py
├── scraper.py
├── requirements.txt
```

## Setup (Windows PowerShell)
```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Scrape Data
The scraper will ask for an edition year (e.g., `70`, `72`, `73`). It fetches graduates and outstanding graduates for School of Computer Science (S1) using the preset filters.

```powershell
# Run the interactive scraper
python scraper.py
```
Outputs are saved in the project root as:
- `YEAR-Graduates.csv`
- `YEAR-Outstanding-Graduates.csv`

Move or copy the CSVs you want to analyze into the `data/` folder.

## Configure Which Datasets To Analyze
The analysis script reads specific files listed in `datasets` inside `analysis.py`. Update this list to match the CSVs you have under `data/`.

Example (current default):
```python
datasets = [
    "70-Outstanding-Graduates.csv",
    "72-Graduates.csv",
    "72-Outstanding-Graduates.csv",
    "73-Graduates.csv",
    "73-Outstanding-Graduates.csv",
]
```
If you scraped `70-Graduates.csv`, add it to the list and ensure the file exists in `data/`.

## Run Analysis
```powershell
python analysis.py
```
You’ll see a menu:
1. Generate Word Cloud
2. Show Top Keywords
3. Run Clustering
4. Filter by Personal Keywords
0. Exit

Notes:
- Word cloud opens a window; close it to return to the menu.
- “Show Top Keywords” uses n-grams and stopwords based on `Configuration` in `analysis.py`.
- “Run Clustering” uses TF‑IDF + KMeans; results print sample titles with cluster labels.
- “Filter by Personal Keywords” saves matching rows to `Similar_Thesis_Title.csv`.

## Adjust Settings
See `Configuration` in `analysis.py`:
- `max_features`: limit for keyword features
- `ngrams_min`/`ngrams_max`: n‑gram range (e.g., 1–2)
- `cluster_k`: number of clusters for KMeans
- `min_df` / `max_df`: TF‑IDF term frequency thresholds