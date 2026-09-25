import os
import pandas as pd

print("Creating high-density sample with guaranteed ground truth matches...")
os.makedirs('dataset/sample', exist_ok=True)

# 1. Take 5,000 S1 records
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=5000)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

s1_ids = set(s1['entity_id'])
gt_sample = gt[gt['source1_entity_id'].isin(s1_ids)]

# Collect all matching S2 and S3 IDs needed
target_pool_ids = set()
for ids in gt_sample['matched_entity_ids'].dropna():
    target_pool_ids.update([x.strip() for x in str(ids).split(',') if x.strip()])

print(f"Targeting {len(s1)} S1 entities with {len(target_pool_ids)} true matching S2/S3 records...")

# 2. Filter S2 and S3 in chunks to grab the matching records + some random distractors
def extract_relevant(source_file, target_ids, max_distractors=15000):
    chunks = []
    distractors = []
    found_targets = 0

    for chunk in pd.read_csv(source_file, sep='\t', chunksize=100000):
        matched = chunk[chunk['entity_id'].isin(target_ids)]
        if not matched.empty:
            chunks.append(matched)
            found_targets += len(matched)
        if len(distractors) < max_distractors:
            distractors.append(chunk.head(1000))

    out_df = pd.concat(chunks + distractors, ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"Loaded {len(out_df)} records from {source_file} (matched {found_targets} targets)")
    return out_df

s2_sample = extract_relevant('dataset/train/train_source2.tsv', target_pool_ids)
s3_sample = extract_relevant('dataset/train/train_source3.tsv', target_pool_ids)

# Save
s1.to_csv('dataset/sample/sample_source1.tsv', sep='\t', index=False)
s2_sample.to_csv('dataset/sample/sample_source2.tsv', sep='\t', index=False)
s3_sample.to_csv('dataset/sample/sample_source3.tsv', sep='\t', index=False)
gt_sample.to_csv('dataset/sample/sample_ground_truth.tsv', sep='\t', index=False)

print("Done! High-density sample ready.")