from rapidfuzz import fuzz, distance
from text_cleaner import extract_numbers


def get_features(r1: dict, r2: dict) -> list:
    n1, n2 = r1['clean_name'], r2['clean_name']
    a1, a2 = r1['clean_addr'], r2['clean_addr']

    # Name similarity metrics
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = distance.JaroWinkler.similarity(n1, n2)
    exact_name = 1.0 if (n1 and n2 and n1 == n2) else 0.0

    # Address similarity metrics (handles missing/nan addresses cleanly)
    has_addr = bool(a1 and a2)
    addr_missing = 1.0 if (not a1 or not a2) else 0.0
    addr_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0 if has_addr else 0.0
    addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0 if has_addr else 0.0

    # Number / PIN / house number anchors
    nums1 = extract_numbers(a1)
    nums2 = extract_numbers(a2)
    shared_nums = nums1 & nums2
    union_nums = nums1 | nums2
    has_shared_num = 1.0 if shared_nums else 0.0
    has_conflict_num = 1.0 if (nums1 and nums2 and not shared_nums) else 0.0
    num_jaccard = len(shared_nums) / len(union_nums) if union_nums else 0.0

    # Explicit Generator Rules:
    # Rule 1: High name match when address was dropped (nan)
    rule_name_no_addr = 1.0 if (name_token_set >= 0.85 and addr_missing) else 0.0
    # Rule 2: High address match with exact building number (DBA trade names like Drexkor)
    rule_addr_dba = 1.0 if (addr_token_set >= 0.82 and has_shared_num) else 0.0
    # Rule 3: High combined confidence
    rule_combined = 1.0 if (name_token_set >= 0.70 and addr_token_set >= 0.70) else 0.0

    return [
        name_ratio,
        name_token_sort,
        name_token_set,
        name_jw,
        exact_name,
        addr_token_sort,
        addr_token_set,
        addr_missing,
        has_shared_num,
        has_conflict_num,
        num_jaccard,
        rule_name_no_addr,
        rule_addr_dba,
        rule_combined,
        abs(len(n1) - len(n2)) / max(1, len(n1), len(n2)),
        abs(len(a1) - len(a2)) / max(1, len(a1), len(a2)) if has_addr else 0.0
    ]