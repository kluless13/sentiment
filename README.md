## Universal Sentiment Analysis for Tickers and Events

This project provides a modular pipeline to classify sentiment as bullish, bearish, or neutral for:
- Tickers (stocks, ETFs, crypto symbols)
- General events/topics (e.g., earnings, product launches, macro news)

The initial implementation ships a robust CLI script with a baseline VADER model and a clean, extensible architecture. It avoids heavy dependencies and works offline with CSVs or Kaggle datasets; connectors for live sources can be added incrementally.

### Goals
- Classify short-form text into bullish/bearish/neutral.
- Aggregate across many documents with confidence and simple explainability.
- Work across domains (equities, crypto, general events) with pluggable sources and models.

### Architecture (modular by responsibility)
- Data collection: pluggable connectors (Kaggle/local CSV/stdin now; Reddit/News/X later).
- Preprocessing: normalize text, deduplicate, optional language filter.
- Modeling: baseline VADER; optional domain models (FinBERT, generic LLMs) via feature flags.
- Aggregation: document-level scores → weighted aggregate per time window/entity.
- Reporting: human-readable summary and JSON output.
- Backtesting (planned): link historical prices to sentiment windows for validation.

### Data Sources
- Local CSV with a text column.
- Kaggle datasets via `kagglehub` (e.g., `equinxx/stock-tweets-for-sentiment-analysis-and-prediction`).
- Planned: Reddit (Pushshift/API), News (NewsAPI/GNews), X/Twitter (snscrape/API), RSS feeds.

### Modeling
- Baseline: VADER (lexicon-based, strong on social/news tone). Domain-agnostic and zero-setup.
- Optional (later):
  - FinBERT (finance-focused transformer) for tickers/news.
  - General sentiment (DistilBERT, RoBERTa).
  - Prompted LLM for event-specific classification.

Mapping strategy (VADER compound score c in [-1, 1]):
- c > 0.05 → bullish
- c < -0.05 → bearish
- otherwise → neutral

### Aggregation & Scoring
- Document-level: compute VADER compound and class.
- Aggregate: mean score, class distribution, and final label by mean with configurable thresholds.
- Confidence: standard error and document count. Optionally weight by recency/engagement later.

### CLI Overview
The CLI orchestrates the pipeline end-to-end and prints both a concise summary and JSON.

Usage examples:
```
python sentiment.py --query "NVDA earnings" --source stdin <<<'Great results, guidance beats expectations.'

python sentiment.py --source kaggle --kaggle-dataset equinxx/stock-tweets-for-sentiment-analysis-and-prediction --limit 1000

python sentiment.py --source csv --csv-path /path/to/texts.csv --text-col text
```

Key options:
- `--source`: one of `stdin`, `csv`, `kaggle` (more later)
- `--query`: topic/entity descriptor (used for metadata, optional today)
- `--ticker`: e.g. `AAPL`, `NVDA`, `BTC` (metadata, future entity-resolver hook)
- `--limit`: cap number of docs analyzed
- `--json-out`: write detailed JSON to a file
- `--sk-model`: path to a trained scikit-learn model (joblib) to use instead of VADER

### Installation
We recommend a clean environment, then install minimal deps:
```
pip install vaderSentiment kagglehub
```
Optional for CSV convenience (not required by the script):
```
pip install pandas pyarrow
```

Note: If you previously hit NumPy 2.x ABI issues via pandas/pyarrow, prefer a clean env and avoid unnecessary compiled deps for the baseline.

### Configuration & Secrets
- Live data connectors (e.g., NewsAPI) will expect API keys via environment variables loaded by your shell or a secrets manager. This baseline does not require API keys.

### Logging & Error Handling
- Uses Python logging for structured INFO/ERROR messages.
- Graceful fallbacks: if a collector fails, errors are surfaced and exit code is non-zero.

### Testing & Validation (planned)
- Unit tests per module (collector, preprocessing, analyzer, aggregator).
- Backtest framework: compute label vs. price reaction over windows (e.g., 1h, 1d, 1w) using `yfinance`/crypto APIs.

### Roadmap
- Add Reddit/News connectors with async fetching, caching, and dedup.
- Add FinBERT and general transformer inference (with device selection and batching).
- Add entity extraction (tickers/symbols) and per-entity aggregation within a corpus.
- Implement backtesting module and simple evaluation dashboards.
- Package as a library and CLI tool, then Dockerize for deployment.

### Repository Layout (initial)
- `sentiment.py`: CLI orchestrator with modular functions for collect → preprocess → analyze → aggregate.
- `sentiment.ipynb`: experiments/scratch (optional).
- `README.md`: this plan and usage.

## Training a custom model
We provide `train.py` to train a TF-IDF + Logistic Regression model. You can derive weak labels from prices or provide a pre-labeled CSV.

Quick start (derive labels from prices):
```
python train.py \
  --tweets-csv stock_tweets.csv \
  --prices-csv stock_yfinance_data.csv \
  --threshold 0.01 \
  --output models/sentiment_pipeline.joblib \
  --limit 50000
```

Use the trained model in inference:
```
python sentiment.py --source csv --csv-path stock_tweets.csv --text-col Tweet --sk-model models/sentiment_pipeline.joblib --limit 1000
```

Notes:
- Tweets are matched to nearest previous and next closes per symbol and labeled bullish/bearish/neutral using a return threshold.
- Adjust `--threshold` (e.g., 1% or 2%) to control label sparsity and class balance.
- For pre-labeled data, ensure a `label` column exists; omit `--prices-csv`.
- To name your custom model "senti1", save to `models/senti1.joblib` instead of the default.

## Evaluate senti1 vs VADER
Compare your trained model against VADER on labeled data (either pre-labeled, or derived from prices like training):

Using derived labels (same threshold):
```
python evaluate.py \
  --tweets-csv stock_tweets.csv \
  --prices-csv stock_yfinance_data.csv \
  --threshold 0.01 \
  --model models/senti1.joblib \
  --limit 20000
```

Using a pre-labeled CSV with columns `text,label`:
```
python evaluate.py \
  --labels-csv path/to/labeled.csv \
  --model models/senti1.joblib \
  --limit 20000
```

It prints JSON containing accuracy, weighted F1, and a classification report for both VADER and your model.

### Example evaluation result
From a run on ~19,888 examples (threshold 1%):
```
{
  "model": {
    "accuracy": 0.6759,
    "f1_weighted": 0.6643,
    "report": "...\n     bearish  F1≈0.659\n     bullish  F1≈0.737\n     neutral  F1≈0.562\n ..."
  },
  "vader": {
    "accuracy": 0.3514,
    "f1_weighted": 0.3445,
    "report": "..."
  },
  "n": 19888
}
```
Interpretation: the custom model (senti1) substantially outperforms VADER; bullish recall is high, neutral is weaker. Consider time-based split, class weighting, and threshold tuning (see next steps below).

### Interpreting your results
Recent runs on your data (threshold 2%) showed:
- senti1 (logreg, class-weight balanced, time-split): accuracy ≈ 0.695, weighted F1 ≈ 0.696
  - bearish F1 ≈ 0.65, bullish F1 ≈ 0.67, neutral F1 ≈ 0.74
- senti1_svm (LinearSVC): accuracy ≈ 0.690, weighted F1 ≈ 0.685
  - tends to have very high precision on neutral but lower recall; bullish/bearish recall can be high
- VADER: accuracy ≈ 0.32, weighted F1 ≈ 0.32 (much weaker)

Takeaways:
- Both custom models strongly outperform VADER on these weak labels.
- The logreg (balanced) model gives the best overall balance; SVM trades recall/precision differently.
- Neutral is generally harder; improving labels (e.g., time window/threshold) can help.

Notes on labels: These are derived from price moves (weak supervision). They are proxy labels, not ground truth. For production, prefer aggregating sentiment across many texts and validating on downstream metrics (e.g., event reactions or backtests).

## Next steps (optional improvements)
- Time-based split to avoid leakage (`--time-split`).
- Class weighting to improve minority classes (`--class-weight balanced`).
- Try a different classifier (`--classifier svm`) and compare.

Examples:
```
python train.py \
  --tweets-csv stock_tweets.csv \
  --prices-csv stock_yfinance_data.csv \
  --threshold 0.02 \
  --classifier logreg \
  --class-weight balanced \
  --time-split \
  --output models/senti1.joblib \
  --limit 100000

python train.py \
  --tweets-csv stock_tweets.csv \
  --prices-csv stock_yfinance_data.csv \
  --classifier svm \
  --output models/senti1_svm.joblib \
  --limit 100000
```


