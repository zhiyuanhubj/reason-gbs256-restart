# Dataset paths on Isambard

All paths below are on the shared Lustre filesystem. `/projects/b6db/...` and
`/lus/lfs1aip2/projects/b6db/...` resolve to the same storage; use the shorter
`/projects/b6db/...` form in job configuration.

## Ready-to-train entry points

| Dataset | Format | Train | Validation | Verified tokens |
|---|---|---|---|---:|
| Cyberpunk 2077 guided | ms-swift JSONL plus extracted images | `/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/cyberpunk_2077/train.jsonl` | `/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/cyberpunk_2077/val.jsonl` | 8,010,389,471 |
| Marvel's Spider-Man 2 guided | ms-swift JSONL plus extracted images | `/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/marvel_spider_2/train.jsonl` | `/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/marvel_spider_2/val.jsonl` | 7,568,554,291 |
| Cyberpunk 2077 + Spider-Man 2 union | ms-swift JSONL plus extracted images | `/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_merged/train.jsonl` | `/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_merged/val.jsonl` | 15,578,943,762 |
| LLaVA-OneVision 1.5 | ms-swift Qwen 3.5 Parquet | `/projects/b6db/opensima/vlm_data/LLaVA-OneVision-1.5-Instruct-Data-qwen-format` | None; do not configure a LLaVA validation path | Not locally measured |
| Genshin BC SL135 | ms-swift JSONL plus extracted images | `/projects/b6db/opensima/data/genshin_bc_sl135/_merged/train.jsonl` | `/projects/b6db/opensima/data/genshin_bc_sl135/_merged/val.jsonl` | Not recorded here |

The Genshin paths are the completed outputs of `06_prepare_bc_dataset.py` then
`07_merge_shuffle_genshin.py`: 89,439 train records and 64 validation records.
Its existing data configuration is `data_configs/yuanshen_bc_sl135.yaml`.

## Training configurations

```text
/projects/b6db/opensima/code/training-best-cp1-full-vittp/data_configs/2077_llava_onevision_128K.yaml
/projects/b6db/opensima/code/training-best-cp1-full-vittp/data_configs/spider_llava_onevision_128K.yaml
/projects/b6db/opensima/code/training-best-cp1-full-vittp/data_configs/llava_onevision.yaml
/projects/b6db/opensima/code/training-best-cp1-full-vittp/data_configs/yuanshen_bc_sl135.yaml
```

The first two configurations mix one game JSONL dataset with the LLaVA
Parquet corpus. They use the corresponding game's validation split only;
LLaVA itself has no validation split.

## Token accounting

The guided-game token totals come from `total_token_count` in:

```text
/projects/b6db/opensima/code/training-best-cp1-full-vittp/hf-bc-parquet-repos-with-secrets-0830.csv
```

```text
Cyberpunk 2077 guided       8,010,389,471
Marvel's Spider-Man 2       7,568,554,291
Combined                   15,578,943,762
```

Use the CSV's `total_token_count` directly. It is not the simple sum of its
`text_token_count` and `vision_token_count` columns; the difference matches the
number of image placeholders. No complete Qwen processor token-count result is
currently stored locally for LLaVA-OneVision, so comments such as `20.142B`
must not be treated as verified measurements.

## Cyberpunk 2077

Extracted source Parquet (518 shards):

```text
/projects/b6db/opensima/sft_data/guided_cyberpunk_2077_bc_01/赛博朋克2077/bc_parquet
```

Converted per-game dataset:

```text
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/cyberpunk_2077/train.jsonl
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/cyberpunk_2077/images/
```

Leakage-free split used by the 2077 + LLaVA configuration (64,194 train and 33
validation records):

```text
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/cyberpunk_2077/train.jsonl
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/cyberpunk_2077/val.jsonl
```

The 47 downloaded `.7z` archives were verified, extracted, and deleted. Their
small `.sha256` records and extraction markers remain.

## Marvel's Spider-Man 2

Extracted source Parquet (403 shards):

```text
/projects/b6db/opensima/sft_data/guided_marvel_spider_bc_01/漫威蜘蛛侠2/bc_parquet
```

Converted per-game dataset:

```text
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/marvel_spider_2/train.jsonl
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/marvel_spider_2/images/
```

Leakage-free split used by the Spider-Man 2 + LLaVA configuration (60,711
train and 31 validation records):

```text
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/marvel_spider_2/train.jsonl
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/marvel_spider_2/val.jsonl
```

The 64 downloaded `.7z` archives were verified, extracted, and deleted. Their
small `.sha256` records and extraction markers remain.

The shuffled union of Cyberpunk 2077 and Spider-Man 2 contains 124,905 train
records and 64 validation records:

```text
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_merged/train.jsonl
/projects/b6db/opensima/sft_data/processed_bc_2077_spider/_merged/val.jsonl
```

## LLaVA-OneVision 1.5

The downloaded repository is already in trainable Parquet form (8,732 Parquet
shards and 21,847,820 rows):

```text
/projects/b6db/opensima/vlm_data/LLaVA-OneVision-1.5-Instruct-Data-qwen-format
```

Do not run this dataset through `06_prepare_bc_dataset.py` or
`07_merge_shuffle_genshin.py`. Its schema contains the ms-swift `messages`,
`images`, and optional grounding `objects` columns, and the Parquet loader reads
them directly. There is no local LLaVA validation dataset.

## Genshin BC SL135

Extracted source Parquet (560 shards, 89,503 trajectories):

```text
/projects/b6db/opensima/yuanshen_bc_formatted_sl135
```

Intermediate output from `06_prepare_bc_dataset.py`:

```text
/projects/b6db/opensima/data/genshin_bc_sl135/yuanshen_bc_sl135/train.jsonl
/projects/b6db/opensima/data/genshin_bc_sl135/yuanshen_bc_sl135/images/
```

Final shuffled output from `07_merge_shuffle_genshin.py`:

```text
/projects/b6db/opensima/data/genshin_bc_sl135/_merged/train.jsonl
/projects/b6db/opensima/data/genshin_bc_sl135/_merged/val.jsonl
```

## Quick checks

```bash
wc -l /projects/b6db/opensima/sft_data/processed_bc_2077_spider/_merged/{train,val}.jsonl
wc -l /projects/b6db/opensima/sft_data/processed_bc_2077_spider/_by_game/{cyberpunk_2077,marvel_spider_2}/{train,val}.jsonl
wc -l /projects/b6db/opensima/data/genshin_bc_sl135/_merged/{train,val}.jsonl
find /projects/b6db/opensima/vlm_data/LLaVA-OneVision-1.5-Instruct-Data-qwen-format \
  -type f -name '*.parquet' | wc -l
```
