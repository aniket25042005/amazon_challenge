import re
from array import array
from collections import Counter, defaultdict
import pandas as pd
from text_cleaner import normalize_text, normalize_address, extract_numbers

STOP_TOKENS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'pvt', 'limited', 
    'co', 'enterprises', 'services', 'solutions', 'industries', 'group', 'the', 'and'
}

ADDR_STOP_WORDS = {
    'road', 'street', 'avenue', 'lane', 'drive', 'floor', 'block', 'near', 'opposite', 'unit', 'suite'
}


def get_blocking_tokens(name: str, addr: str) -> set:
    """Extracts dual-channel tokens: both from Name AND from Address locality."""
    tokens = set()

    # 1. Distinctive name tokens
    words = re.findall(r'[a-z0-9]{3,}', name)
    for w in words:
        if w not in STOP_TOKENS:
            tokens.add(w)
            if len(w) >= 4:
                tokens.add(w[:3])

    # 2. Distinctive address tokens (house numbers + city/street anchors)
    tokens.update(extract_numbers(addr))
    if addr:
        addr_words = re.findall(r'[a-z]{4,}', addr)
        sig_addr = [w for w in addr_words if w not in ADDR_STOP_WORDS]
        # Include top 3 distinctive locality words (e.g. ticonderoga, mylapore, kansas)
        tokens.update(sig_addr[:3])

    return tokens


def run_blocking(
    df_s1: pd.DataFrame, 
    df_pool: pd.DataFrame, 
    top_k: int = 15, 
    max_posting_size: int = 3500
) -> dict:
    candidates = {eid: [] for eid in df_s1['entity_id']}
    countries = set(df_s1['country'].dropna().unique()) | set(df_pool['country'].dropna().unique())

    for country in countries:
        s1_subset = df_s1[df_s1['country'] == country]
        pool_subset = df_pool[df_pool['country'] == country]

        if s1_subset.empty or pool_subset.empty:
            continue

        print(f"\n[Dual-Channel Index: {country}] S1={len(s1_subset):,}, Pool={len(pool_subset):,}")

        pool_ids = pool_subset['entity_id'].values
        pool_names = pool_subset['clean_name'].values
        pool_addrs = pool_subset['clean_addr'].values

        # Build Inverted Index using C arrays (< 250 MB)
        index = defaultdict(lambda: array('I'))
        for idx in range(len(pool_subset)):
            toks = get_blocking_tokens(pool_names[idx], pool_addrs[idx])
            for t in toks:
                index[t].append(idx)

        print(f"  Index built: {len(index):,} distinct tokens.")

        s1_ids = s1_subset['entity_id'].values
        s1_names = s1_subset['clean_name'].values
        s1_addrs = s1_subset['clean_addr'].values
        total = len(s1_subset)

        print(f"  Querying {total:,} records...")

        for i in range(total):
            q_toks = get_blocking_tokens(s1_names[i], s1_addrs[i])
            hit_counts = Counter()

            for t in q_toks:
                postings = index.get(t)
                if postings and len(postings) <= max_posting_size:
                    hit_counts.update(postings)

            if hit_counts:
                top_hits = [pool_ids[idx] for idx, _ in hit_counts.most_common(top_k)]
                candidates[s1_ids[i]] = top_hits
            else:
                candidates[s1_ids[i]] = []

            if (i + 1) % 150000 == 0:
                print(f"  Progress {country}: {i + 1:,} / {total:,} completed")

        del index
        print(f"  Finished {country} blocking!")

    return candidates