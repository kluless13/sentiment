import argparse
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    txt = text.replace("\r", " ").replace("\n", " ")
    return " ".join(txt.split()).strip()


def load_tweets(csv_path: str, text_col: str = "Tweet", date_col: str = "Date", stock_col: str = "Stock Name") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tweets CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in [text_col, date_col, stock_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in tweets CSV. Available: {list(df.columns)}")
    df = df[[date_col, stock_col, text_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_text)
    # Parse timezone-aware timestamps and convert to naive UTC
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_convert(None)
    df[stock_col] = df[stock_col].astype(str).str.upper()
    df = df.dropna(subset=[date_col, text_col])
    return df


def load_prices(csv_path: str, date_col: str = "Date", close_col: str = "Close", stock_col: str = "Stock Name") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prices CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in [date_col, close_col, stock_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in prices CSV. Available: {list(df.columns)}")
    df = df[[date_col, stock_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df[stock_col] = df[stock_col].astype(str).str.upper()
    df = df.dropna(subset=[date_col, close_col])
    df = df.sort_values([stock_col, date_col]).reset_index(drop=True)
    return df


def label_by_forward_return(
    tweets: pd.DataFrame,
    prices: pd.DataFrame,
    threshold: float = 0.01,
    date_col: str = "Date",
    stock_col: str = "Stock Name",
    close_col: str = "Close",
) -> pd.DataFrame:
    # Ensure dtype alignment and strict sorting required by merge_asof
    tweets = tweets.copy()
    prices = prices.copy()
    tweets[date_col] = pd.to_datetime(tweets[date_col], utc=False, errors="coerce")
    prices[date_col] = pd.to_datetime(prices[date_col], utc=False, errors="coerce")
    tweets = tweets.dropna(subset=[date_col, stock_col]).copy()
    prices = prices.dropna(subset=[date_col, stock_col, close_col]).copy()

    symbols = sorted(set(tweets[stock_col].astype(str)) & set(prices[stock_col].astype(str)))
    if not symbols:
        raise ValueError("No overlapping symbols between tweets and prices.")

    per_symbol_frames = []
    for sym in symbols:
        t_sym = tweets[tweets[stock_col].astype(str) == sym].sort_values(date_col)
        r_sym = prices[prices[stock_col].astype(str) == sym].sort_values(date_col)
        if t_sym.empty or r_sym.empty:
            continue
        prev = pd.merge_asof(
            t_sym,
            r_sym[[date_col, close_col]],
            on=date_col,
            direction="backward",
        ).rename(columns={close_col: "prev_close"})
        nxt = pd.merge_asof(
            t_sym,
            r_sym[[date_col, close_col]],
            on=date_col,
            direction="forward",
        ).rename(columns={close_col: "next_close"})
        df_sym = prev[[date_col, stock_col]].copy()
        df_sym["prev_close"] = prev["prev_close"].values
        df_sym["next_close"] = nxt["next_close"].values
        df_sym = df_sym.join(t_sym.drop(columns=[date_col, stock_col]).reset_index(drop=True))
        per_symbol_frames.append(df_sym)

    if not per_symbol_frames:
        raise ValueError("No data after per-symbol alignment.")

    df = pd.concat(per_symbol_frames, ignore_index=True)

    df = df.dropna(subset=["prev_close", "next_close"]).reset_index(drop=True)
    ret = (df["next_close"].astype(float) / df["prev_close"].astype(float)) - 1.0
    df["forward_return"] = ret
    cond_pos = df["forward_return"] > threshold
    cond_neg = df["forward_return"] < -threshold
    df["label"] = np.where(cond_pos, "bullish", np.where(cond_neg, "bearish", "neutral"))
    return df


def _ensure_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=False, errors="coerce")


def label_by_horizon(
    tweets: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: pd.Timedelta,
    benchmark: Optional[pd.DataFrame] = None,
    threshold: float = 0.01,
    threshold_mode: str = "absolute",
    q_low: float = 0.3,
    q_high: float = 0.7,
    date_col: str = "Date",
    stock_col: str = "Stock Name",
    close_col: str = "Close",
) -> pd.DataFrame:
    tweets = tweets.copy()
    prices = prices.copy()
    tweets[date_col] = _ensure_dt(tweets[date_col])
    prices[date_col] = _ensure_dt(prices[date_col])
    tweets = tweets.dropna(subset=[date_col, stock_col])
    prices = prices.dropna(subset=[date_col, stock_col, close_col])

    bench = None
    if benchmark is not None:
        bench = benchmark.copy()
        bench[date_col] = _ensure_dt(bench[date_col])
        bench = bench.dropna(subset=[date_col, close_col]).sort_values(date_col).reset_index(drop=True)

    symbols = sorted(set(tweets[stock_col].astype(str)) & set(prices[stock_col].astype(str)))
    frames = []
    for sym in symbols:
        t_sym = tweets[tweets[stock_col].astype(str) == sym].sort_values(date_col)
        p_sym = prices[prices[stock_col].astype(str) == sym].sort_values(date_col)
        if t_sym.empty or p_sym.empty:
            continue
        # Price at tweet time (backward) and at t+h (backward)
        base = pd.merge_asof(
            t_sym[[date_col, stock_col]],
            p_sym[[date_col, close_col]],
            on=date_col,
            direction="backward",
        ).rename(columns={close_col: "p0"})
        future_times = t_sym[[date_col]].copy()
        future_times[date_col] = future_times[date_col] + horizon
        fut = pd.merge_asof(
            future_times.sort_values(date_col),
            p_sym[[date_col, close_col]],
            on=date_col,
            direction="backward",
        ).rename(columns={close_col: "p1"})
        df_sym = base[[date_col, stock_col]].copy()
        df_sym["p0"] = base["p0"].values
        df_sym["p1"] = fut["p1"].values
        df_sym = df_sym.join(t_sym.drop(columns=[date_col, stock_col]).reset_index(drop=True))

        if bench is not None:
            b0 = pd.merge_asof(
                t_sym[[date_col]].sort_values(date_col),
                bench[[date_col, close_col]],
                on=date_col,
                direction="backward",
            ).rename(columns={close_col: "b0"})
            ft = t_sym[[date_col]].copy()
            ft[date_col] = ft[date_col] + horizon
            b1 = pd.merge_asof(
                ft.sort_values(date_col),
                bench[[date_col, close_col]],
                on=date_col,
                direction="backward",
            ).rename(columns={close_col: "b1"})
            df_sym["b0"] = b0["b0"].values
            df_sym["b1"] = b1["b1"].values

        frames.append(df_sym)

    if not frames:
        raise ValueError("No data after horizon alignment.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["p0", "p1"]).reset_index(drop=True)
    ret = (df["p1"].astype(float) / df["p0"].astype(float)) - 1.0
    if bench is not None:
        mask = df[["b0", "b1"]].notna().all(axis=1)
        bench_ret = (df.loc[mask, "b1"].astype(float) / df.loc[mask, "b0"].astype(float)) - 1.0
        ret.loc[mask] = ret.loc[mask] - bench_ret.values

    if threshold_mode == "quantile":
        lo = float(np.quantile(ret.dropna(), q_low))
        hi = float(np.quantile(ret.dropna(), q_high))
        df["forward_return"] = ret
        df["label"] = np.where(ret > hi, "bullish", np.where(ret < lo, "bearish", "neutral"))
    else:
        df["forward_return"] = ret
        df["label"] = np.where(ret > threshold, "bullish", np.where(ret < -threshold, "bearish", "neutral"))
    return df


def build_pipeline(
    classifier: str = "logreg",
    class_weight: Optional[str] = None,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    if classifier == "svm":
        clf = LinearSVC()
    else:
        clf = LogisticRegression(
            max_iter=300,
            solver="liblinear",
            multi_class="auto",
            class_weight=class_weight,
            n_jobs=None,
        )
    return Pipeline(steps=[("tfidf", vectorizer), ("clf", clf)])


def train_model(
    tweets_csv: str,
    prices_csv: Optional[str],
    labels_csv: Optional[str],
    labels_text_col: Optional[str],
    labels_label_col: Optional[str],
    output_path: str,
    threshold: float,
    horizon_minutes: Optional[int],
    benchmark_csv: Optional[str],
    threshold_mode: str,
    q_low: float,
    q_high: float,
    test_size: float,
    random_state: int,
    limit: Optional[int],
    classifier: str,
    class_weight: Optional[str],
    time_split: bool,
) -> None:
    if labels_csv:
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
        df = pd.read_csv(labels_csv)
        text_col = labels_text_col or "text"
        label_col = labels_label_col or "label"
        for col in [text_col, label_col]:
            if col not in df.columns:
                raise ValueError(f"Labels CSV missing column '{col}'. Columns: {list(df.columns)}")
        labeled = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).copy()
        labeled["text"] = labeled["text"].astype(str).map(normalize_text)
    else:
        tweets = load_tweets(tweets_csv)
        if limit is not None and limit > 0:
            tweets = tweets.iloc[:limit].copy()
        if prices_csv:
            prices = load_prices(prices_csv)
            if horizon_minutes is not None and horizon_minutes > 0:
                horizon = pd.Timedelta(minutes=int(horizon_minutes))
                bench = load_prices(benchmark_csv) if benchmark_csv else None
                labeled = label_by_horizon(
                    tweets,
                    prices,
                    horizon=horizon,
                    benchmark=bench,
                    threshold=threshold,
                    threshold_mode=threshold_mode,
                    q_low=q_low,
                    q_high=q_high,
                )
            else:
                labeled = label_by_forward_return(tweets, prices, threshold=threshold)
        else:
            # Expect a pre-labeled CSV with 'label' column in tweets
            if "label" not in tweets.columns:
                raise ValueError("When prices_csv is not provided, tweets CSV must include a 'label' column or use --labels-csv.")
            labeled = tweets.rename(columns={"Tweet": "text"}).copy()
            labeled["text"] = labeled["text"].astype(str).map(normalize_text)

    # If using time-based split, sort by Date when available to preserve temporal order
    if time_split and "Date" in labeled.columns:
        labeled = labeled.sort_values("Date").reset_index(drop=True)

    # Keep only text and label for modeling
    if "text" not in labeled.columns or "label" not in labeled.columns:
        raise ValueError("Prepared dataset must contain 'text' and 'label' columns.")
    labeled = labeled[["text", "label"]].dropna()
    # Drop empty strings
    labeled = labeled[labeled["text"].str.len() > 0]

    if time_split and len(labeled) > 1:
        # Use the current order (sorted by Date if present) for a simple time split
        X = labeled["text"].values
        y = labeled["label"].values
        cut = int((1 - test_size) * len(labeled))
        X_train, X_test = X[:cut], X[cut:]
        y_train, y_test = y[:cut], y[cut:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            labeled["text"].values,
            labeled["label"].values,
            test_size=test_size,
            random_state=random_state,
            stratify=labeled["label"].values,
        )

    pipeline = build_pipeline(classifier=classifier, class_weight=class_weight)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, digits=4)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": list(sorted(set(y_train))), "classifier": classifier, "class_weight": class_weight}, output_path)

    logging.info("Saved model to %s", output_path)
    print(json.dumps({"f1_weighted": f1, "report": report}, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a TF-IDF + LogisticRegression sentiment model")
    p.add_argument("--tweets-csv", default=None, help="Path to tweets CSV (must have Tweet, Date, Stock Name) when deriving labels")
    p.add_argument("--prices-csv", default=None, help="Path to prices CSV to derive labels (optional)")
    p.add_argument("--labels-csv", default=None, help="Path to pre-labeled CSV for supervised training")
    p.add_argument("--labels-text-col", default=None, help="Text column name in labels CSV (default: text)")
    p.add_argument("--labels-label-col", default=None, help="Label column name in labels CSV (default: label)")
    p.add_argument("--output", default="models/sentiment_pipeline.joblib", help="Output path for model")
    p.add_argument("--threshold", type=float, default=0.01, help="Return threshold for labels (e.g., 0.01 = 1%)")
    p.add_argument("--horizon-minutes", type=int, default=None, help="Use intraday horizon minutes for labeling (e.g., 30, 60, 240)")
    p.add_argument("--benchmark-csv", default=None, help="Benchmark prices CSV (same schema) to compute abnormal returns")
    p.add_argument("--threshold-mode", choices=["absolute", "quantile"], default="absolute", help="Use absolute threshold or quantile split for labels")
    p.add_argument("--q-low", type=float, default=0.3, help="Low quantile for quantile threshold mode")
    p.add_argument("--q-high", type=float, default=0.7, help="High quantile for quantile threshold mode")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--limit", type=int, default=None, help="Limit number of tweets for quick training")
    p.add_argument("--classifier", choices=["logreg", "svm"], default="logreg", help="Classifier choice")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="none", help="Class weighting for logistic regression")
    p.add_argument("--time-split", action="store_true", help="Use a simple time-based split instead of random split")
    p.add_argument("-v", action="count", default=0, help="Increase verbosity (-v, -vv)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.v)
    try:
        train_model(
            tweets_csv=args.tweets_csv,
            prices_csv=args.prices_csv,
            labels_csv=args.labels_csv,
            labels_text_col=args.labels_text_col,
            labels_label_col=args.labels_label_col,
            output_path=args.output,
            threshold=args.threshold,
            horizon_minutes=args.horizon_minutes,
            benchmark_csv=args.benchmark_csv,
            threshold_mode=args.threshold_mode,
            q_low=args.q_low,
            q_high=args.q_high,
            test_size=args.test_size,
            random_state=args.random_state,
            limit=args.limit,
            classifier=args.classifier,
            class_weight=(None if args.class_weight == "none" else "balanced"),
            time_split=args.time_split,
        )
        return 0
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


