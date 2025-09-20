import argparse
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
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


def load_corpus(csv_path: str, text_col: str, date_col: str, stock_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in [text_col, date_col, stock_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in corpus CSV.")
    df = df[[date_col, stock_col, text_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_text)
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_convert(None)
    df[stock_col] = df[stock_col].astype(str).str.upper()
    return df.dropna(subset=[date_col, text_col])


def load_prices(csv_path: str, date_col: str = "Date", close_col: str = "Close", stock_col: str = "Stock Name") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in [date_col, close_col, stock_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in prices CSV.")
    df = df[[date_col, stock_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df[stock_col] = df[stock_col].astype(str).str.upper()
    return df.dropna(subset=[date_col, close_col]).sort_values([stock_col, date_col]).reset_index(drop=True)


def predict_texts(model_path: str, texts: pd.Series) -> pd.DataFrame:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    labels = pipeline.predict(texts.tolist())
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts.tolist())
    df = pd.DataFrame({"label": labels})
    if proba is not None:
        # map bullish/bearish/neutral to pseudo score
        classes = list(getattr(pipeline, "classes_", []))
        if not classes and hasattr(pipeline, "named_steps"):
            last = list(pipeline.named_steps.values())[-1]
            classes = list(getattr(last, "classes_", []))
        if classes:
            idx = {c: i for i, c in enumerate(classes)}
            score = []
            for row in proba:
                bull = float(row[idx.get("bullish", 0)]) if "bullish" in idx else 0.0
                bear = float(row[idx.get("bearish", 0)]) if "bearish" in idx else 0.0
                score.append(bull - bear)
            df["score"] = score
    return df


def aggregate_daily(corpus: pd.DataFrame, preds: pd.DataFrame, date_col: str, stock_col: str) -> pd.DataFrame:
    data = corpus[[date_col, stock_col]].copy()
    data["date"] = pd.to_datetime(data[date_col]).dt.date
    agg = pd.concat([data[["date", stock_col]].reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    # Simple numeric score per label if no probabilities
    if "score" not in agg.columns:
        mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        agg["score"] = agg["label"].map(mapping).astype(float)
    grouped = agg.groupby([stock_col, "date"]).agg(
        mean_score=("score", "mean"),
        n_docs=("score", "size"),
        bullish=(lambda x: (agg.loc[x.index, "label"] == "bullish").sum()),
        bearish=(lambda x: (agg.loc[x.index, "label"] == "bearish").sum()),
        neutral=(lambda x: (agg.loc[x.index, "label"] == "neutral").sum()),
    ).reset_index()
    return grouped


def daily_forward_return(prices: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    prices = prices.sort_values(["Stock Name", "Date"]).reset_index(drop=True)
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["next_close"] = df["Close"].shift(-horizon_days)
        df["ret_fwd"] = (df["next_close"] / df["Close"]) - 1.0
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        return df
    out = prices.groupby("Stock Name", group_keys=False).apply(compute)
    return out[["Stock Name", "date", "ret_fwd"]].dropna()


def run_backtest(
    model_path: str,
    corpus_csv: str,
    prices_csv: str,
    text_col: str = "Tweet",
    date_col: str = "Date",
    stock_col: str = "Stock Name",
    horizon_days: int = 1,
) -> dict:
    corpus = load_corpus(corpus_csv, text_col=text_col, date_col=date_col, stock_col=stock_col)
    preds = predict_texts(model_path, corpus[text_col])
    daily = aggregate_daily(corpus, preds, date_col=date_col, stock_col=stock_col)
    prices = load_prices(prices_csv)
    rets = daily_forward_return(prices, horizon_days=horizon_days)
    merged = pd.merge(daily, rets, left_on=[stock_col, "date"], right_on=["Stock Name", "date"], how="inner")
    corr = float(np.corrcoef(merged["mean_score"], merged["ret_fwd"])[0, 1]) if len(merged) > 2 else np.nan
    hit = float((np.sign(merged["mean_score"]) == np.sign(merged["ret_fwd"]).replace(0, np.nan)).mean())
    return {
        "n_days": int(len(merged)),
        "ic": corr,
        "hit_rate": hit,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple daily backtest for sentiment -> forward returns")
    p.add_argument("--model", required=True, help="Path to trained model joblib")
    p.add_argument("--corpus-csv", required=True, help="CSV with Date, Stock Name, and text column")
    p.add_argument("--prices-csv", required=True, help="Daily prices CSV")
    p.add_argument("--text-col", default="Tweet")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--stock-col", default="Stock Name")
    p.add_argument("--horizon-days", type=int, default=1)
    p.add_argument("-v", action="count", default=0)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.v)
    try:
        res = run_backtest(
            model_path=args.model,
            corpus_csv=args.corpus_csv,
            prices_csv=args.prices_csv,
            text_col=args.text_col,
            date_col=args.date_col,
            stock_col=args.stock_col,
            horizon_days=args.horizon_days,
        )
        print(json.dumps(res, ensure_ascii=False))
        return 0
    except Exception as exc:
        logging.exception("Backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


