import os
import gc
import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from text_cleaner import normalize_text, normalize_address
from blocking import run_blocking
from features import get_features


def f05_score(y_true: set, y_pred: set) -> float:
    if not y_true and not y_pred:
        return 1.0
    if not y_true or not y_pred:
        return 0.0

    tp = len(y_true & y_pred)
    p = tp / len(y_pred)
    r = tp / len(y_true)

    denom = 0.25 * p + r
    return (1.25 * p * r) / denom if denom > 0 else 0.0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_name'] = df['business_name'].fillna('').apply(normalize_text)
    df['clean_addr'] = df['business_address'].fillna('').apply(normalize_address)
    return df


def train_model(train_dir: str, top_k: int = 15):
    print("Loading training data...")
    tag = 'sample' if 'sample' in train_dir else 'train'
    s1 = preprocess(pd.read_csv(f"{train_dir}/{tag}_source1.tsv", sep='\t'))
    s2 = preprocess(pd.read_csv(f"{train_dir}/{tag}_source2.tsv", sep='\t'))
    s3 = preprocess(pd.read_csv(f"{train_dir}/{tag}_source3.tsv", sep='\t'))
    gt = pd.read_csv(f"{train_dir}/{tag}_ground_truth.tsv", sep='\t')

    ground_truth = {}
    for _, row in gt.iterrows():
        val = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        ground_truth[row['source1_entity_id']] = set(val.split(',')) if val.strip() else set()

    pool = pd.concat([s2, s3], ignore_index=True)
    s1_dict = {eid: (name, addr) for eid, name, addr in zip(s1['entity_id'], s1['clean_name'], s1['clean_addr'])}
    pool_dict = {eid: (name, addr) for eid, name, addr in zip(pool['entity_id'], pool['clean_name'], pool['clean_addr'])}

    candidates = run_blocking(s1, pool, top_k=top_k)

    X, y, pairs = [], [], []
    for s1_id, cands in candidates.items():
        if not cands:
            continue
        n1, a1 = s1_dict[s1_id]
        true_ids = ground_truth.get(s1_id, set())

        for cid in cands:
            if cid not in pool_dict:
                continue
            n2, a2 = pool_dict[cid]
            X.append(get_features({'clean_name': n1, 'clean_addr': a1}, {'clean_name': n2, 'clean_addr': a2}))
            y.append(1 if cid in true_ids else 0)
            pairs.append((s1_id, cid))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Training on {len(y):,} pairs (positives: {int(y.sum()):,})")

    model = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.07,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    pred_map = {}
    for (s1_id, cid), prob in zip(pairs, probs):
        pred_map.setdefault(s1_id, []).append((cid, prob))

    best_t, best_score = 0.60, -1.0
    for t in np.arange(0.50, 0.85, 0.05):
        scores = [
            f05_score(ground_truth.get(eid, set()), {cid for cid, p in pred_map.get(eid, []) if p >= t})
            for eid in s1['entity_id']
        ]
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_t = t

    print(f"Trained: best threshold = {best_t:.2f}, sample F0.5 = {best_score:.4f}")
    del s1, s2, s3, pool, X, y, probs, pred_map, ground_truth
    gc.collect()
    return model, best_t


def predict_test(model, threshold: float, test_dir: str, output_dir: str, top_k: int = 15):
    os.makedirs(output_dir, exist_ok=True)
    cand_out = os.path.join(output_dir, 'candidate_pairs.tsv')
    match_out = os.path.join(output_dir, 'matching_results.tsv')

    with open(cand_out, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(match_out, 'w', encoding='utf-8') as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    print("Loading test data...")
    test_s1 = pd.read_csv(f"{test_dir}/test_source1.tsv", sep='\t')
    test_s2 = pd.read_csv(f"{test_dir}/test_source2.tsv", sep='\t')
    test_s3 = pd.read_csv(f"{test_dir}/test_source3.tsv", sep='\t')
    test_pool = pd.concat([test_s2, test_s3], ignore_index=True)
    del test_s2, test_s3
    gc.collect()

    test_s1['country'] = test_s1['country'].fillna('UNKNOWN')
    test_pool['country'] = test_pool['country'].fillna('UNKNOWN')

    countries = list(test_s1['country'].unique())
    print(f"Countries to process: {countries}")

    for country in countries:
        print(f"\n--- Processing {country} ---")
        s1_c = preprocess(test_s1[test_s1['country'] == country])
        pool_c = preprocess(test_pool[test_pool['country'] == country])

        if s1_c.empty:
            continue

        s1_tuples = {eid: (n, a) for eid, n, a in zip(s1_c['entity_id'], s1_c['clean_name'], s1_c['clean_addr'])}
        pool_tuples = {eid: (n, a) for eid, n, a in zip(pool_c['entity_id'], pool_c['clean_name'], pool_c['clean_addr'])}

        candidates = run_blocking(s1_c, pool_c, top_k=top_k)

        cand_lines = []
        all_pairs = []
        pair_meta = []

        for eid in s1_c['entity_id']:
            cands = candidates.get(eid, [])
            cand_lines.append(f"{eid}\t{','.join(cands)}")

            if not cands:
                continue

            n1, a1 = s1_tuples[eid]
            for cid in cands:
                if cid in pool_tuples:
                    n2, a2 = pool_tuples[cid]
                    all_pairs.append(get_features({'clean_name': n1, 'clean_addr': a1}, {'clean_name': n2, 'clean_addr': a2}))
                    pair_meta.append((eid, cid))

        print(f"  Scoring {len(all_pairs):,} candidate pairs...")
        matches_by_s1 = {eid: [] for eid in s1_c['entity_id']}

        if all_pairs:
            feats_array = np.array(all_pairs, dtype=np.float32)
            probs = model.predict_proba(feats_array)[:, 1]

            # Single-Owner rule: pick best S1 for each pool entity to destroy false positives!
            best_claim = {}
            for (eid, cid), prob in zip(pair_meta, probs):
                if prob >= threshold:
                    if cid not in best_claim or prob > best_claim[cid][1]:
                        best_claim[cid] = (eid, prob)

            for cid, (eid, _) in best_claim.items():
                matches_by_s1[eid].append(cid)

        match_lines = [f"{eid}\t{','.join(matches_by_s1[eid])}" for eid in s1_c['entity_id']]

        # Append to files
        with open(cand_out, 'a', encoding='utf-8') as f_cand:
            f_cand.write("\n".join(cand_lines) + "\n")
        with open(match_out, 'a', encoding='utf-8') as f_match:
            f_match.write("\n".join(match_lines) + "\n")

        print(f"  Finished {country} ({len(s1_c):,} entities written)")
        del s1_c, pool_c, s1_tuples, pool_tuples, candidates, cand_lines, match_lines, all_pairs, pair_meta, matches_by_s1
        gc.collect()

    print(f"\nAll countries finished! Outputs ready in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='dataset/sample')
    parser.add_argument('--test-dir', default=None)
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--top-k', type=int, default=15)
    args = parser.parse_args()

    model, threshold = train_model(args.train_dir, top_k=args.top_k)

    if args.test_dir:
        predict_test(model, threshold, args.test_dir, args.output_dir, top_k=args.top_k)