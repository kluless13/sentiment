import argparse
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
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
    return " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()


def load_labeled(csv_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in labeled CSV. Available: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text"}).copy()
    df["text"] = df["text"].astype(str).map(normalize_text)
    df = df.dropna(subset=["text", label_col])
    df = df[df["text"].str.len() > 0]
    return df


def label_from_prices(
    tweets_csv: str,
    prices_csv: str,
    threshold: float,
    horizon_minutes: int = None,
    threshold_mode: str = "absolute",
    q_low: float = 0.3,
    q_high: float = 0.7,
    benchmark_csv: str = None,
    tweet_text_col: str = "Tweet",
    tweet_date_col: str = "Date",
    tweet_symbol_col: str = "Stock Name",
    price_date_col: str = "Date",
    price_close_col: str = "Close",
    price_symbol_col: str = "Stock Name",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    # Load tweets
    t = pd.read_csv(tweets_csv)
    for c in [tweet_text_col, tweet_date_col, tweet_symbol_col]:
        if c not in t.columns:
            raise ValueError(f"Tweets CSV missing column '{c}'. Columns: {list(t.columns)}")
    t = t[[tweet_date_col, tweet_symbol_col, tweet_text_col]].copy()
    t[tweet_date_col] = pd.to_datetime(t[tweet_date_col], utc=True, errors="coerce").dt.tz_convert(None)
    t[tweet_symbol_col] = t[tweet_symbol_col].astype(str).str.upper()
    t[tweet_text_col] = t[tweet_text_col].astype(str).map(normalize_text)
    t = t.dropna(subset=[tweet_date_col, tweet_text_col])
    if limit is not None:
        t = t.iloc[:limit].copy()

    # Load prices
    p = pd.read_csv(prices_csv)
    for c in [price_date_col, price_close_col, price_symbol_col]:
        if c not in p.columns:
            raise ValueError(f"Prices CSV missing column '{c}'. Columns: {list(p.columns)}")
    p = p[[price_date_col, price_symbol_col, price_close_col]].copy()
    p[price_date_col] = pd.to_datetime(p[price_date_col], utc=False, errors="coerce")
    p[price_symbol_col] = p[price_symbol_col].astype(str).str.upper()
    p = p.dropna(subset=[price_date_col, price_close_col])

    # Per-symbol asof merge
    symbols = sorted(set(t[tweet_symbol_col]) & set(p[price_symbol_col]))
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        t_sym = t[t[tweet_symbol_col] == sym].sort_values(tweet_date_col)
        p_sym = p[p[price_symbol_col] == sym].sort_values(price_date_col)
        if t_sym.empty or p_sym.empty:
            continue
        if horizon_minutes is None:
            prev = pd.merge_asof(
                t_sym,
                p_sym[[price_date_col, price_close_col]].rename(columns={price_date_col: tweet_date_col}),
                on=tweet_date_col,
                direction="backward",
            ).rename(columns={price_close_col: "prev_close"})
            nxt = pd.merge_asof(
                t_sym,
                p_sym[[price_date_col, price_close_col]].rename(columns={price_date_col: tweet_date_col}),
                on=tweet_date_col,
                direction="forward",
            ).rename(columns={price_close_col: "next_close"})
            df_sym = prev[[tweet_date_col, tweet_symbol_col, tweet_text_col]].copy()
            df_sym["prev_close"] = prev["prev_close"].values
            df_sym["next_close"] = nxt["next_close"].values
        else:
            # Intraday horizon
            base = pd.merge_asof(
                t_sym[[tweet_date_col, tweet_symbol_col]],
                p_sym[[price_date_col, price_close_col]].rename(columns={price_date_col: tweet_date_col}),
                on=tweet_date_col,
                direction="backward",
            ).rename(columns={price_close_col: "p0"})
            fut_times = t_sym[[tweet_date_col]].copy()
            fut_times[tweet_date_col] = pd.to_datetime(fut_times[tweet_date_col]) + pd.to_timedelta(horizon_minutes, unit="m")
            fut = pd.merge_asof(
                fut_times.sort_values(tweet_date_col),
                p_sym[[price_date_col, price_close_col]].rename(columns={price_date_col: tweet_date_col}),
                on=tweet_date_col,
                direction="backward",
            ).rename(columns={price_close_col: "p1"})
            df_sym = base[[tweet_date_col, tweet_symbol_col]].copy()
            df_sym["p0"] = base["p0"].values
            df_sym["p1"] = fut["p1"].values
            df_sym = df_sym.join(t_sym[[tweet_text_col]].reset_index(drop=True))
        frames.append(df_sym)
    if not frames:
        raise ValueError("No aligned data for labeling.")
    df = pd.concat(frames, ignore_index=True)
    if horizon_minutes is None:
        df = df.dropna(subset=["prev_close", "next_close"])  # drop entries lacking adjacent price
        ret = (df["next_close"].astype(float) / df["prev_close"].astype(float)) - 1.0
    else:
        df = df.dropna(subset=["p0", "p1"])  # drop entries lacking intraday prices
        ret = (df["p1"].astype(float) / df["p0"].astype(float)) - 1.0

    # Abnormal returns option
    if benchmark_csv and horizon_minutes is not None:
        b = pd.read_csv(benchmark_csv)
        if "Date" not in b.columns or "Close" not in b.columns:
            raise ValueError("Benchmark CSV must have Date, Close")
        b["Date"] = pd.to_datetime(b["Date"], utc=False, errors="coerce")
        b = b.sort_values("Date").reset_index(drop=True)
        # Compute benchmark base and future
        base_b = pd.merge_asof(
            t[[tweet_date_col]].sort_values(tweet_date_col),
            b[["Date", "Close"]].rename(columns={"Date": tweet_date_col}),
            on=tweet_date_col,
            direction="backward",
        ).rename(columns={"Close": "b0"})
        fut_b_times = t[[tweet_date_col]].copy()
        fut_b_times[tweet_date_col] = pd.to_datetime(fut_b_times[tweet_date_col]) + pd.to_timedelta(horizon_minutes, unit="m")
        fut_b = pd.merge_asof(
            fut_b_times.sort_values(tweet_date_col),
            b[["Date", "Close"]].rename(columns={"Date": tweet_date_col}),
            on=tweet_date_col,
            direction="backward",
        ).rename(columns={"Close": "b1"})
        mask = base_b["b0"].notna() & fut_b["b1"].notna()
        bench_ret = (fut_b.loc[mask, "b1"].astype(float) / base_b.loc[mask, "b0"].astype(float)) - 1.0
        ret.loc[mask] = ret.loc[mask] - bench_ret.values

    if threshold_mode == "quantile" and len(ret.dropna()) > 0:
        lo = float(np.quantile(ret.dropna(), q_low))
        hi = float(np.quantile(ret.dropna(), q_high))
        df["label"] = np.where(ret > hi, "bullish", np.where(ret < lo, "bearish", "neutral"))
    else:
        df["label"] = np.where(ret > threshold, "bullish", np.where(ret < -threshold, "bearish", "neutral"))
    df = df.rename(columns={tweet_text_col: "text"})
    return df[["text", "label"]]


def vader_predict(texts: List[str]) -> List[str]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except Exception as exc:
        raise RuntimeError("vaderSentiment is required: pip install vaderSentiment") from exc
    analyzer = SentimentIntensityAnalyzer()
    labels: List[str] = []
    for t in texts:
        c = float(analyzer.polarity_scores(t).get("compound", 0.0))
        labels.append("bullish" if c > 0.05 else ("bearish" if c < -0.05 else "neutral"))
    return labels


def model_predict(model_path: str, texts: List[str]) -> List[str]:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    return [str(x) for x in pipeline.predict(texts)]


def evaluate(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, str]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, digits=4)
    return acc, f1, report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate senti1 (scikit) vs VADER")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels-csv", help="CSV with columns: text,label")
    src.add_argument("--tweets-csv", help="Tweets CSV to derive labels from prices")
    p.add_argument("--prices-csv", help="Prices CSV for labeling (required if --tweets-csv used)")
    p.add_argument("--threshold", type=float, default=0.01, help="Return threshold for labeling when deriving")
    p.add_argument("--horizon-minutes", type=int, default=None, help="Intraday horizon (minutes) for labeling")
    p.add_argument("--threshold-mode", choices=["absolute", "quantile"], default="absolute")
    p.add_argument("--q-low", type=float, default=0.3)
    p.add_argument("--q-high", type=float, default=0.7)
    p.add_argument("--benchmark-csv", default=None, help="Benchmark prices CSV (Date, Close) for abnormal returns")
    p.add_argument("--model", required=True, help="Path to trained scikit model (e.g., models/senti1.joblib)")
    p.add_argument("--skip-vader", action="store_true", help="Skip VADER evaluation if not installed or for speed")
    p.add_argument("--limit", type=int, default=None, help="Limit examples for quick evaluation")
    p.add_argument("-v", action="count", default=0, help="Increase verbosity")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.v)
    try:
        if args.labels_csv:
            df = load_labeled(args.labels_csv)
        else:
            if not args.prices_csv:
                raise ValueError("--prices-csv is required when using --tweets-csv")
            df = label_from_prices(
                tweets_csv=args.tweets_csv,
                prices_csv=args.prices_csv,
                threshold=args.threshold,
                horizon_minutes=args.horizon_minutes,
                threshold_mode=args.threshold_mode,
                q_low=args.q_low,
                q_high=args.q_high,
                benchmark_csv=args.benchmark_csv,
                limit=args.limit,
            )

        if args.limit is not None:
            df = df.iloc[: args.limit].copy()

        texts = df["text"].astype(str).tolist()
        y_true = df["label"].astype(str).tolist()

        res_vader = None
        if not args.skip_vader:
            try:
                y_vader = vader_predict(texts)
                acc_v, f1_v, rep_v = evaluate(y_true, y_vader)
                res_vader = {"accuracy": acc_v, "f1_weighted": f1_v, "report": rep_v}
            except Exception as exc:
                logging.warning("Skipping VADER evaluation: %s", exc)

        y_model = model_predict(args.model, texts)
        acc_m, f1_m, rep_m = evaluate(y_true, y_model)

        output = {
            "model": {"accuracy": acc_m, "f1_weighted": f1_m, "report": rep_m},
            "n": len(texts),
        }
        if res_vader is not None:
            output["vader"] = res_vader
        print(json.dumps(output, ensure_ascii=False))
        return 0
    except Exception as exc:
        logging.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


