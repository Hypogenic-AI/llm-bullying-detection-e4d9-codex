#!/usr/bin/env python3
"""Run iterative rejection vs one-shot anti-detector experiment."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from scipy.stats import ttest_rel, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Config:
    seed: int = 42
    n_fresh_items: int = 20
    n_external_items: int = 20
    iterative_rounds: int = 4
    temperature: float = 0.7
    max_tokens_answer: int = 220
    model_candidates: Tuple[str, ...] = ("gpt-4.1-mini", "gpt-4.1")
    stylometric_max_features: int = 60000
    word_detector_max_features: int = 50000
    detector_train_size: int = 6000
    results_dir: str = "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(config: Config) -> None:
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir, "plots").mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def load_hc3(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = clean_text(row.get("question", ""))
            human_answers = [clean_text(x) for x in row.get("human_answers", []) if clean_text(x)]
            ai_answers = [clean_text(x) for x in row.get("chatgpt_answers", []) if clean_text(x)]
            if not q or not human_answers:
                continue
            rows.append(
                {
                    "question": q,
                    "human_text": human_answers[0],
                    "dataset_ai_text": ai_answers[0] if ai_answers else "",
                    "source": row.get("source", "unknown"),
                }
            )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    return df


def load_external_generated(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["generated"] == True].copy()  # noqa: E712
    df["text"] = df["text"].fillna("").map(clean_text)
    df = df[df["text"].str.len() > 300]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df[["text", "model"]]


def build_stylometric_detector(config: Config) -> Dict[str, object]:
    pile_paths = sorted(Path("datasets/ai_text_detection_pile/data").glob("*.parquet"))
    if not pile_paths:
        raise FileNotFoundError("No ai_text_detection_pile parquet files found")

    frames = []
    for p in pile_paths:
        frames.append(pd.read_parquet(p, columns=["source", "text"]))
    pile = pd.concat(frames, ignore_index=True)
    pile["text"] = pile["text"].fillna("").map(clean_text)
    pile = pile[pile["text"].str.len() > 120]
    pile["label"] = (pile["source"] != "human").astype(int)

    human = pile[pile["label"] == 0].sample(
        n=min(config.detector_train_size // 2, (pile["label"] == 0).sum()), random_state=config.seed
    )
    ai = pile[pile["label"] == 1].sample(
        n=min(config.detector_train_size // 2, (pile["label"] == 1).sum()), random_state=config.seed
    )
    data = pd.concat([human, ai], ignore_index=True).sample(frac=1.0, random_state=config.seed)

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=config.seed, stratify=data["label"]
    )
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=config.stylometric_max_features)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(x_train_vec, y_train)
    test_probs = clf.predict_proba(x_test_vec)[:, 1]
    auc = float(roc_auc_score(y_test, test_probs))

    return {"vectorizer": vectorizer, "clf": clf, "test_auc": auc, "train_n": len(x_train), "test_n": len(x_test)}


def build_word_detector(config: Config) -> Dict[str, object]:
    pile_paths = sorted(Path("datasets/ai_text_detection_pile/data").glob("*.parquet"))
    if not pile_paths:
        raise FileNotFoundError("No ai_text_detection_pile parquet files found")

    frames = []
    for p in pile_paths:
        frames.append(pd.read_parquet(p, columns=["source", "text"]))
    pile = pd.concat(frames, ignore_index=True)
    pile["text"] = pile["text"].fillna("").map(clean_text)
    pile = pile[pile["text"].str.len() > 120]
    pile["label"] = (pile["source"] != "human").astype(int)

    human = pile[pile["label"] == 0].sample(
        n=min(config.detector_train_size // 2, (pile["label"] == 0).sum()), random_state=config.seed + 1
    )
    ai = pile[pile["label"] == 1].sample(
        n=min(config.detector_train_size // 2, (pile["label"] == 1).sum()), random_state=config.seed + 1
    )
    data = pd.concat([human, ai], ignore_index=True).sample(frac=1.0, random_state=config.seed + 1)

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=config.seed + 1, stratify=data["label"]
    )
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=config.word_detector_max_features)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(max_iter=2500, n_jobs=1)
    clf.fit(x_train_vec, y_train)
    test_probs = clf.predict_proba(x_test_vec)[:, 1]
    auc = float(roc_auc_score(y_test, test_probs))

    return {"vectorizer": vectorizer, "clf": clf, "test_auc": auc, "train_n": len(x_train), "test_n": len(x_test)}


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def call_openai_chat(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = completion.choices[0].message.content or ""
    return clean_text(text)


def resolve_model(client: OpenAI, candidates: Tuple[str, ...], temperature: float, max_tokens: int) -> str:
    probe = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Reply with OK."},
    ]
    last_error = None
    for m in candidates:
        try:
            _ = call_openai_chat(client, m, probe, temperature=0.0, max_tokens=8)
            return m
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            continue
    raise RuntimeError(f"No candidate model available. Last error: {last_error}")


def generate_baseline_answer(client: OpenAI, model: str, question: str, config: Config) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Write natural, specific prose with varied sentence lengths. "
                "Avoid bullet lists unless required."
            ),
        },
        {
            "role": "user",
            "content": (
                "Answer the following question in one coherent paragraph (120-180 words), "
                "with concrete detail and plain language:\n\n"
                f"{question}"
            ),
        },
    ]
    return call_openai_chat(client, model, messages, temperature=config.temperature, max_tokens=config.max_tokens_answer)


def one_shot_rewrite(client: OpenAI, model: str, text: str, config: Config) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an expert editor. Keep meaning intact while improving naturalness.",
        },
        {
            "role": "user",
            "content": (
                "Rewrite this so it sounds less AI-generated and more like a real person wrote it. "
                "Preserve meaning, key facts, and approximate length. Return only the revised text.\n\n"
                f"TEXT:\n{text}"
            ),
        },
    ]
    return call_openai_chat(client, model, messages, temperature=config.temperature, max_tokens=config.max_tokens_answer)


def iterative_rewrite_rounds(client: OpenAI, model: str, text: str, config: Config) -> List[str]:
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are an expert editor. Keep meaning intact while improving naturalness.",
        },
        {
            "role": "user",
            "content": (
                "Rewrite this text so it reads like natural human writing. "
                "Preserve core meaning and length. Return only rewritten text.\n\n"
                f"TEXT:\n{text}"
            ),
        },
    ]

    outputs: List[str] = []
    for r in range(1, config.iterative_rounds + 1):
        rewritten = call_openai_chat(client, model, messages, temperature=config.temperature, max_tokens=config.max_tokens_answer)
        outputs.append(rewritten)
        messages.append({"role": "assistant", "content": rewritten})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Rejected: this is still too AI-sounding. Rewrite it again to sound more human. "
                    "Keep the same facts and overall meaning. Return only revised text."
                ),
            }
        )
    return outputs


def stylometric_scores(texts: List[str], detector: Dict[str, object]) -> List[float]:
    vectorizer = detector["vectorizer"]
    clf = detector["clf"]
    x = vectorizer.transform(texts)
    probs = clf.predict_proba(x)[:, 1]
    return [float(p) for p in probs]


def word_detector_scores(texts: List[str], detector: Dict[str, object]) -> List[float]:
    vectorizer = detector["vectorizer"]
    clf = detector["clf"]
    x = vectorizer.transform(texts)
    probs = clf.predict_proba(x)[:, 1]
    return [float(p) for p in probs]


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(a)
    idx = np.arange(n)
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        diffs.append(float(np.mean(a[samp] - b[samp])))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    n = len(a) * len(b)
    return float((gt - lt) / n)


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adjusted = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = min(prev, ranked[i] * n / rank)
        adjusted[i] = adj
        prev = adj
    out = np.empty(n)
    out[order] = adjusted
    return [float(x) for x in out]


def compute_quality_features(text: str) -> Dict[str, float]:
    words = re.findall(r"[A-Za-z']+", text.lower())
    uniq = len(set(words))
    total = len(words)
    ttr = uniq / total if total else 0.0
    return {"word_count": float(total), "type_token_ratio": float(ttr)}


def summarize_data_quality(hc3_df: pd.DataFrame, ext_df: pd.DataFrame) -> Dict[str, object]:
    return {
        "hc3_rows": int(len(hc3_df)),
        "hc3_missing_question": int((hc3_df["question"].str.len() == 0).sum()),
        "hc3_missing_human_text": int((hc3_df["human_text"].str.len() == 0).sum()),
        "hc3_duplicate_questions": int(hc3_df.duplicated(subset=["question"]).sum()),
        "external_rows": int(len(ext_df)),
        "external_duplicate_text": int(ext_df.duplicated(subset=["text"]).sum()),
        "external_short_filtered_out_threshold": 300,
    }


def run(config: Config) -> None:
    set_seed(config.seed)
    ensure_dirs(config)

    start_time = time.time()

    hc3 = load_hc3(Path("datasets/hc3_raw/all.jsonl"))
    external = load_external_generated(Path("datasets/ghostbuster_reuter/data/train-00000-of-00001.parquet"))
    quality = summarize_data_quality(hc3, external)

    hc3_sample = hc3.sample(n=min(config.n_fresh_items, len(hc3)), random_state=config.seed).reset_index(drop=True)
    ext_sample = external.sample(n=min(config.n_external_items, len(external)), random_state=config.seed).reset_index(drop=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this experiment")
    client = OpenAI(api_key=api_key)

    chosen_model = resolve_model(client, config.model_candidates, config.temperature, config.max_tokens_answer)

    # Generate under fresh condition.
    fresh_records: List[Dict[str, object]] = []
    for idx, row in hc3_sample.iterrows():
        print(f"[fresh] item {idx + 1}/{len(hc3_sample)}", flush=True)
        q = row["question"]
        human_text = row["human_text"]
        base = generate_baseline_answer(client, chosen_model, q, config)
        one = one_shot_rewrite(client, chosen_model, base, config)
        rounds = iterative_rewrite_rounds(client, chosen_model, base, config)

        fresh_records.append(
            {
                "item_id": f"fresh_{idx}",
                "source_type": "fresh_model_generation",
                "question": q,
                "human_reference": human_text,
                "baseline": base,
                "one_shot": one,
                **{f"iter_round_{i+1}": rounds[i] for i in range(len(rounds))},
                "iterative_final": rounds[-1],
            }
        )

    # Rewrite external generated texts (not generated in this session).
    external_records: List[Dict[str, object]] = []
    for idx, row in ext_sample.iterrows():
        print(f"[external] item {idx + 1}/{len(ext_sample)}", flush=True)
        original = row["text"]
        one = one_shot_rewrite(client, chosen_model, original, config)
        rounds = iterative_rewrite_rounds(client, chosen_model, original, config)
        external_records.append(
            {
                "item_id": f"external_{idx}",
                "source_type": "external_ai_text",
                "origin_model": row.get("model", "unknown"),
                "original_external": original,
                "one_shot": one,
                **{f"iter_round_{i+1}": rounds[i] for i in range(len(rounds))},
                "iterative_final": rounds[-1],
            }
        )

    fresh_df = pd.DataFrame(fresh_records)
    ext_df = pd.DataFrame(external_records)

    # Prepare detector suite.
    stylometric = build_stylometric_detector(config)
    word_detector = build_word_detector(config)

    # Flatten all texts for scoring.
    flat_rows: List[Dict[str, object]] = []
    fresh_conditions = ["baseline", "one_shot"] + [f"iter_round_{i+1}" for i in range(config.iterative_rounds)] + ["iterative_final"]
    for _, row in fresh_df.iterrows():
        for c in fresh_conditions:
            flat_rows.append(
                {
                    "item_id": row["item_id"],
                    "source_type": row["source_type"],
                    "condition": c,
                    "text": row[c],
                }
            )
        flat_rows.append(
            {
                "item_id": row["item_id"],
                "source_type": row["source_type"],
                "condition": "human_reference",
                "text": row["human_reference"],
            }
        )

    ext_conditions = ["original_external", "one_shot"] + [f"iter_round_{i+1}" for i in range(config.iterative_rounds)] + ["iterative_final"]
    for _, row in ext_df.iterrows():
        for c in ext_conditions:
            flat_rows.append(
                {
                    "item_id": row["item_id"],
                    "source_type": row["source_type"],
                    "condition": c,
                    "text": row[c],
                }
            )

    scored_df = pd.DataFrame(flat_rows)
    scored_df["text"] = scored_df["text"].map(clean_text)

    word_prob = word_detector_scores(scored_df["text"].tolist(), word_detector)
    stylo_prob = stylometric_scores(scored_df["text"].tolist(), stylometric)
    scored_df["word_ai_prob"] = word_prob
    scored_df["stylometric_ai_prob"] = stylo_prob

    quality_feats = scored_df["text"].map(compute_quality_features)
    scored_df["word_count"] = [q["word_count"] for q in quality_feats]
    scored_df["type_token_ratio"] = [q["type_token_ratio"] for q in quality_feats]

    # Main paired tests.
    tests = []
    for detector_col in ["word_ai_prob", "stylometric_ai_prob"]:
        for src, one_cond, iter_cond in [
            ("fresh_model_generation", "one_shot", "iterative_final"),
            ("external_ai_text", "one_shot", "iterative_final"),
        ]:
            sub = scored_df[scored_df["source_type"] == src]
            one = sub[sub["condition"] == one_cond].sort_values("item_id")[detector_col].to_numpy()
            itr = sub[sub["condition"] == iter_cond].sort_values("item_id")[detector_col].to_numpy()
            if len(one) == 0 or len(itr) == 0:
                continue
            t_res = ttest_rel(itr, one)
            try:
                w_res = wilcoxon(itr, one, zero_method="wilcox", correction=False)
                w_p = float(w_res.pvalue)
            except Exception:
                w_p = math.nan
            ci_low, ci_high = bootstrap_ci_mean_diff(itr, one, n_boot=10000, seed=config.seed)
            tests.append(
                {
                    "detector": detector_col,
                    "source_type": src,
                    "n": int(len(one)),
                    "mean_iterative": float(np.mean(itr)),
                    "mean_one_shot": float(np.mean(one)),
                    "mean_diff_iter_minus_one": float(np.mean(itr - one)),
                    "ttest_p": float(t_res.pvalue),
                    "wilcoxon_p": w_p,
                    "cohens_d_paired": cohens_d_paired(itr, one),
                    "cliffs_delta": cliffs_delta(itr, one),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        tests_df["ttest_p_fdr_bh"] = benjamini_hochberg(tests_df["ttest_p"].fillna(1.0).tolist())

    # Pass-rate summaries.
    pass_thresholds = {"word_ai_prob": 0.5, "stylometric_ai_prob": 0.5}
    pass_rows = []
    for detector_col, thr in pass_thresholds.items():
        tmp = scored_df.copy()
        tmp["pass"] = (tmp[detector_col] < thr).astype(int)
        for (src, cond), grp in tmp.groupby(["source_type", "condition"]):
            pass_rows.append(
                {
                    "detector": detector_col,
                    "source_type": src,
                    "condition": cond,
                    "threshold": thr,
                    "pass_rate": float(grp["pass"].mean()),
                    "n": int(len(grp)),
                }
            )
    pass_df = pd.DataFrame(pass_rows)

    # AUC vs human references for fresh condition.
    auc_rows = []
    fresh_sub = scored_df[scored_df["source_type"] == "fresh_model_generation"].copy()
    for detector_col in ["word_ai_prob", "stylometric_ai_prob"]:
        for cond in ["baseline", "one_shot", "iterative_final"] + [f"iter_round_{i+1}" for i in range(config.iterative_rounds)]:
            ai_grp = fresh_sub[fresh_sub["condition"] == cond].sort_values("item_id")
            hm_grp = fresh_sub[fresh_sub["condition"] == "human_reference"].sort_values("item_id")
            if len(ai_grp) == 0 or len(hm_grp) == 0:
                continue
            m = min(len(ai_grp), len(hm_grp))
            y_true = np.array([1] * m + [0] * m)
            y_score = np.concatenate([ai_grp[detector_col].to_numpy()[:m], hm_grp[detector_col].to_numpy()[:m]])
            auc = float(roc_auc_score(y_true, y_score))
            auc_rows.append({"detector": detector_col, "condition": cond, "auc_vs_human": auc, "n_ai": m, "n_human": m})
    auc_df = pd.DataFrame(auc_rows)

    # Round trajectory summaries.
    traj_rows = []
    for src in ["fresh_model_generation", "external_ai_text"]:
        for detector_col in ["word_ai_prob", "stylometric_ai_prob"]:
            for r in range(1, config.iterative_rounds + 1):
                cond = f"iter_round_{r}"
                grp = scored_df[(scored_df["source_type"] == src) & (scored_df["condition"] == cond)]
                if grp.empty:
                    continue
                traj_rows.append(
                    {
                        "source_type": src,
                        "detector": detector_col,
                        "round": r,
                        "mean_ai_prob": float(grp[detector_col].mean()),
                        "std_ai_prob": float(grp[detector_col].std(ddof=1)),
                    }
                )
    traj_df = pd.DataFrame(traj_rows)

    # Build compact summary table.
    summary_rows = []
    key_conditions = {
        "fresh_model_generation": ["baseline", "one_shot", "iterative_final", "human_reference"],
        "external_ai_text": ["original_external", "one_shot", "iterative_final"],
    }
    for src, conds in key_conditions.items():
        for cond in conds:
            grp = scored_df[(scored_df["source_type"] == src) & (scored_df["condition"] == cond)]
            if grp.empty:
                continue
            summary_rows.append(
                {
                    "source_type": src,
                    "condition": cond,
                    "n": int(len(grp)),
                    "word_mean": float(grp["word_ai_prob"].mean()),
                    "word_std": float(grp["word_ai_prob"].std(ddof=1)),
                    "stylometric_mean": float(grp["stylometric_ai_prob"].mean()),
                    "stylometric_std": float(grp["stylometric_ai_prob"].std(ddof=1)),
                    "mean_word_count": float(grp["word_count"].mean()),
                    "mean_ttr": float(grp["type_token_ratio"].mean()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    # Plotting.
    sns.set_theme(style="whitegrid")

    # Plot 1: main condition comparison by detector.
    p1 = summary_df[summary_df["condition"].isin(["baseline", "one_shot", "iterative_final", "original_external", "human_reference"])]
    if not p1.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=p1, x="condition", y="word_mean", hue="source_type", ax=axes[0])
        axes[0].set_title("Mean AI Probability (Word Detector)")
        axes[0].set_ylabel("AI probability")
        axes[0].tick_params(axis="x", rotation=35)

        sns.barplot(data=p1, x="condition", y="stylometric_mean", hue="source_type", ax=axes[1])
        axes[1].set_title("Mean AI Probability (Stylometric Detector)")
        axes[1].set_ylabel("AI probability")
        axes[1].tick_params(axis="x", rotation=35)

        plt.tight_layout()
        fig.savefig(Path(config.results_dir, "plots", "condition_comparison.png"), dpi=220)
        plt.close(fig)

    # Plot 2: iterative trajectory.
    if not traj_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax_i, detector_col in enumerate(["word_ai_prob", "stylometric_ai_prob"]):
            sub = traj_df[traj_df["detector"] == detector_col]
            sns.lineplot(data=sub, x="round", y="mean_ai_prob", hue="source_type", marker="o", ax=axes[ax_i])
            axes[ax_i].set_title(f"Round Trajectory: {detector_col}")
            axes[ax_i].set_ylabel("Mean AI probability")
            axes[ax_i].set_xlabel("Iterative rejection round")
        plt.tight_layout()
        fig.savefig(Path(config.results_dir, "plots", "iterative_trajectory.png"), dpi=220)
        plt.close(fig)

    # Plot 3: AUC by condition.
    if not auc_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=auc_df, x="condition", y="auc_vs_human", hue="detector", ax=ax)
        ax.set_title("Detector AUC vs Human References (Fresh condition)")
        ax.set_ylabel("AUC")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        fig.savefig(Path(config.results_dir, "plots", "auc_vs_human.png"), dpi=220)
        plt.close(fig)

    # Plot 4: pass rates.
    if not pass_df.empty:
        p2 = pass_df[pass_df["condition"].isin(["baseline", "one_shot", "iterative_final", "original_external", "human_reference"])]
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=p2, x="condition", y="pass_rate", hue="detector", ax=ax)
        ax.set_title("Pass Rate at AI-probability Threshold 0.5")
        ax.set_ylabel("Pass rate")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        fig.savefig(Path(config.results_dir, "plots", "pass_rates.png"), dpi=220)
        plt.close(fig)

    # Save outputs.
    scored_df.to_csv(Path(config.results_dir, "scored_texts.csv"), index=False)
    summary_df.to_csv(Path(config.results_dir, "summary_table.csv"), index=False)
    tests_df.to_csv(Path(config.results_dir, "stat_tests.csv"), index=False)
    pass_df.to_csv(Path(config.results_dir, "pass_rates.csv"), index=False)
    auc_df.to_csv(Path(config.results_dir, "auc_table.csv"), index=False)
    traj_df.to_csv(Path(config.results_dir, "trajectory_table.csv"), index=False)

    with Path(config.results_dir, "fresh_records.json").open("w", encoding="utf-8") as f:
        json.dump(fresh_records, f, ensure_ascii=False, indent=2)
    with Path(config.results_dir, "external_records.json").open("w", encoding="utf-8") as f:
        json.dump(external_records, f, ensure_ascii=False, indent=2)

    duration_sec = time.time() - start_time

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "seaborn": sns.__version__,
        "model_used": chosen_model,
        "detectors": {
            "stylometric": {
                "train_n": stylometric["train_n"],
                "test_n": stylometric["test_n"],
                "test_auc": stylometric["test_auc"],
            },
            "word": {
                "train_n": word_detector["train_n"],
                "test_n": word_detector["test_n"],
                "test_auc": word_detector["test_auc"],
            },
        },
        "gpu_detection_nvidia_smi": os.popen(
            "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo NO_GPU"
        )
        .read()
        .strip(),
        "duration_sec": duration_sec,
        "data_quality": quality,
    }

    with Path(config.results_dir, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Run complete")
    print(f"Model: {chosen_model}")
    print(f"Duration sec: {duration_sec:.1f}")
    print(f"Stylometric detector test AUC: {stylometric['test_auc']:.4f}")
    if not tests_df.empty:
        print("\nPrimary paired tests:")
        print(tests_df[["detector", "source_type", "n", "mean_diff_iter_minus_one", "ttest_p", "ttest_p_fdr_bh", "cohens_d_paired"]])


if __name__ == "__main__":
    run(Config())
