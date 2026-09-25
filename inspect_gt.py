import pandas as pd

print("Loading ground truth...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t').set_index('entity_id')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t').set_index('entity_id')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t').set_index('entity_id')
pool = pd.concat([s2, s3])

for _, row in gt.head(3).iterrows():
    eid = row['source1_entity_id']
    m = str(row['matched_entity_ids'])
    print(f"\n=== S1: {eid} | {s1.loc[eid, 'business_name']} | {s1.loc[eid, 'business_address']} ===")
    for cid in m.split(','):
        cid = cid.strip()
        if cid in pool.index:
            print(f"   MATCH ({cid}): {pool.loc[cid, 'business_name']} | {pool.loc[cid, 'business_address']}")