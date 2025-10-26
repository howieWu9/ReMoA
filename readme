# ReMoA README (ICDE Submission Format)

## 1. Project Overview
The ReMoA project provides a Mixture-of-Experts (MoE) reinforcement learning framework for retrieval-augmented question generation. The core script `moe_duqgen_klock_rl.py` consumes document embeddings, structural priors, and policy gradients to generate and filter candidate questions. This document serves as an operational guide tailored to ICDE supplementary materials, covering dependency setup, data preparation, training/inference workflows, and key parameter summaries.

## 2. Environment and Dependencies
1. Use **Python 3.10+** on a GPU node with CUDA support.
2. Create an isolated Conda environment:
   ```bash
   conda create -n remoa python=3.10 -y
   conda activate remoa
   ```
3. Install project dependencies (replace with the actual `requirements.txt` path while keeping it abstract):
   ```bash
   pip install -r <PROJECT_ROOT>/requirements.txt
   ```
4. If Hugging Face models are required, configure offline cache directories before the first run:
   ```bash
   export TRANSFORMERS_CACHE=<PROJECT_ROOT>/cache
   export HF_DATASETS_CACHE=<PROJECT_ROOT>/cache
   ```

## 3. Data Preparation
1. Place the raw corpus (JSONL) and its vector representations (PyTorch `.pt`) in `<PROJECT_ROOT>/data/<DATASET_NAME>/`.
2. Example directory layout:
   ```text
   <PROJECT_ROOT>/data/<DATASET_NAME>/
   ├── documents.jsonl        # Raw document collection
   ├── embeddings.pt          # Document embedding cache
   └── moe/                   # Output directory for generated questions
   ```
3. To rebuild embeddings, use helper scripts under `data_preparation/`.

## 4. SLURM Job Example
On a SLURM-enabled cluster, submit the following `sbatch` script. The configuration assumes a single A100 (40GB) GPU; adjust according to available resources.
```bash
#!/bin/bash
#SBATCH --job-name=remoa-moe
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module purge
module load cuda/<CUDA_VERSION>
conda activate remoa

cd <PROJECT_ROOT>

python -X utf8 moe_duqgen_klock_rl.py \
  --collection_jsonl "<PROJECT_ROOT>/data/<DATASET_NAME>/documents.jsonl" \
  --collection_emb_pt "<PROJECT_ROOT>/data/<DATASET_NAME>/embeddings.pt" \
  --out_dir "<PROJECT_ROOT>/data/<DATASET_NAME>/moe" \
  --cache_dir "<PROJECT_ROOT>/cache" \
  --local_only \
  --auto_k \
  --target_num_q 1000 \
  --use_candidate_best --n_cands 10 \
  --use_dedup --dedup_threshold 0.90 \
  --use_zscore --z_temp 1.3 \
  --eta 0.30 --use_floor --w_floor 0.05 --use_entropy --entropy 0.10 \
  --hard_match --lambda_hard 0.80 --hard_warmup_ratio 0.30 \
  --topk 500 --sim_threshold 0.50 --r1_tau 42 \
  --temp 0.7 --top_p 0.95 \
  --use_advantage --baseline_beta 0.05
```

### 4.1 Option A (Recommended)
- **Auto-K** leverages structural priors while keeping the total number of questions fixed by `--target_num_q` (1000).
- Recommended when the candidate pool is large and cluster sizes should be inferred automatically.

### 4.2 Option B: Align Cluster Count
To force exactly one question per cluster, replace `--auto_k` with `--align_to_k`:
```bash
python -X utf8 moe_duqgen_klock_rl.py \
  --collection_jsonl "<PROJECT_ROOT>/data/<DATASET_NAME>/documents.jsonl" \
  --collection_emb_pt "<PROJECT_ROOT>/data/<DATASET_NAME>/embeddings.pt" \
  --out_dir "<PROJECT_ROOT>/data/<DATASET_NAME>/moe" \
  --cache_dir "<PROJECT_ROOT>/cache" \
  --local_only \
  --target_num_q 1000 \
  --align_to_k \
  --use_candidate_best --n_cands 10 \
  --use_dedup --dedup_threshold 0.90 \
  --use_zscore --z_temp 1.3 \
  --eta 0.30 --use_floor --w_floor 0.05 --use_entropy --entropy 0.10 \
  --hard_match --lambda_hard 0.80 --hard_warmup_ratio 0.30 \
  --topk 500 --sim_threshold 0.50 --r1_tau 42 \
  --temp 0.7 --top_p 0.95 \
  --use_advantage --baseline_beta 0.05
```

## 5. Key Parameter Reference
| Parameter | Description |
| --- | --- |
| `--collection_jsonl` | Path to the document collection in JSONL format. |
| `--collection_emb_pt` | Cached document embeddings to avoid redundant encoding. |
| `--out_dir` | Output directory for generated questions and logs. |
| `--cache_dir` | Location for models and intermediate artifacts, enabling offline use. |
| `--local_only` | Forces reliance on local model caches and disables online downloads. |
| `--auto_k` | Allocates per-cluster capacity from structural priors while honoring `--target_num_q`. |
| `--align_to_k` | Aligns the number of generated questions with the cluster count (typically with `--target_num_q` set to the total number of clusters). |
| `--target_num_q` | Desired total number of generated questions. |
| `--use_candidate_best / --n_cands` | Retain the top `n` candidates per document and pick the best-performing one. |
| `--use_dedup / --dedup_threshold` | Remove near-duplicate questions using a semantic similarity threshold. |
| `--use_zscore / --z_temp` | Normalize rewards to reduce gradient variance. |
| `--eta` | Global reward scaling factor. |
| `--use_floor / --w_floor` | Provide a lower bound for low-reward samples to avoid vanishing gradients. |
| `--use_entropy / --entropy` | Add policy entropy regularization to encourage exploration. |
| `--hard_match / --lambda_hard / --hard_warmup_ratio` | Control the weight and warm-up schedule of the hard matching loss. |
| `--topk / --sim_threshold / --r1_tau` | Configure retrieval depth and similarity thresholds for scoring. |
| `--temp / --top_p` | Sampling temperature and nucleus filtering for generation. |
| `--use_advantage / --baseline_beta` | Enable an advantage baseline to stabilize policy gradients. |

## 6. Outputs and Evaluation
1. Upon completion, `<OUT_DIR>` contains:
   - `questions.jsonl`: Generated questions with metadata.
   - `metrics.json`: Training logs and reward curves.
   - `checkpoints/`: Policy checkpoints.
2. Use scripts such as `smart_eval_monot5_only.sh` to run automated evaluations on the generated questions.

## 7. Troubleshooting
- **Insufficient GPU memory**: Reduce `--n_cands` or `--topk`, and adjust batch sizes within the script.
- **Missing caches**: Ensure `--cache_dir` and related environment variables point to the same location, pre-downloading models if necessary.
- **Duplicate questions**: Increase `--dedup_threshold` or raise the entropy weight to encourage diversity.

## 8. Citation
If you build upon this work, please cite the relevant paper and include this operational guide in the appendix.
