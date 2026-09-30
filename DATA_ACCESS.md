# Getting the official data

1. Open the [competition data page](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).
2. Sign in to Kaggle. Review and accept any applicable access terms yourself.
3. Download the Parquet files below from the official distribution. CSV copies are not needed by this pipeline.
4. Extract them under `data/raw`, preserving the directory names shown below. Keep Kaggle credentials outside this repository.

Required for the first training run:

```text
data/raw/parquet_files/train/train_base.parquet
data/raw/parquet_files/train/train_static_0_0.parquet
data/raw/parquet_files/train/train_static_0_1.parquet
data/raw/parquet_files/train/train_applprev_1_0.parquet
data/raw/parquet_files/train/train_applprev_1_1.parquet
```

For unlabeled scoring, obtain `test_base.parquet`, all `test_static_0_*.parquet`, and all `test_applprev_1_*.parquet` under `data/raw/parquet_files/test/`.

If using Kaggle's official CLI, list the competition's files before downloading to confirm exact names for your release. The [official API/CLI documentation](https://github.com/Kaggle/kaggle-api) explains authentication and competition downloads. No credentials or automatic agreement acceptance are built into this project.

Do not mix releases, CSV and Parquet copies, or repeated shards. The pipeline checks every shard for required columns and rejects duplicate keys. It records source-file sizes and hashes so a run can be matched to its inputs. A hash establishes file identity, not licensing rights or target quality.

The starter config uses the base, static, and previous-application tables. It is not necessary to load every competition table just to establish a baseline. For a smaller feature output, use `--sample-fraction 0.1`; sampling is deterministic by case ID and is applied across all weeks. The source validation and aggregation still scan the selected source tables.

Data and generated artifacts are excluded from Git. The source dataset keeps its own terms; the code license does not grant rights to the dataset.
