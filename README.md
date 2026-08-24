# Unified Metric Learning for Personalized Fashion Recommendations with Fast User Adaptation

Official PyTorch implementation of:

> Z. Lu, Y. Hu, C. Yu, Y. Chen, and B. Zeng, "Unified metric learning for personalized fashion recommendations with fast user adaptation," *IEEE Transactions on Multimedia*, pp. 1–14, 2026, doi: [10.1109/TMM.2026.3694502](https://doi.org/10.1109/TMM.2026.3694502).

## Citation

If you find this code useful for your research, please consider citing:

```bibtex
@article{Lu2026UnifiedMetric,
  title     = {Unified Metric Learning for Personalized Fashion Recommendations with Fast User Adaptation},
  author    = {Lu, Zhi and Hu, Yang and Yu, Cong and Chen, Yan and Zeng, Bing},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2026},
  pages     = {1--14},
  doi       = {10.1109/TMM.2026.3694502}
}
```

## Method

LPAE learns a unified metric space for personalized fashion recommendation. Given an outfit $o=\{x_1,\ldots,x_n\}$, a shared backbone encoder first computes item-level embeddings $\{\boldsymbol{h}_1^{(b)},\ldots,\boldsymbol{h}_n^{(b)}\}$. A set transformer then captures high-order interactions among items and produces contextualized embeddings $\{\boldsymbol{h}_1^{(c)},\ldots,\boldsymbol{h}_n^{(c)}\}$, which are aggregated into an outfit-level representation $\boldsymbol{z}$ via mean pooling.

Each candidate embedding $\boldsymbol{x}$ is scored with a von Mises–Fisher style energy that interpolates between a global prototype $\boldsymbol{\omega}$ and a personalized shift $\boldsymbol{\mu}$. The normalized probability over a candidate pool $\Omega$ is:

$$
p(\boldsymbol{x} ; \boldsymbol{\omega}, \boldsymbol{\mu}, \kappa)=\frac{\exp \bigl(\kappa (\lambda \boldsymbol{\omega}+\boldsymbol{\mu})^{\top} \boldsymbol{x}\bigr)}{\sum_{\boldsymbol{x} \in \Omega} \exp \bigl(\kappa (\lambda \boldsymbol{\omega}+\boldsymbol{\mu})^{\top} \boldsymbol{x}\bigr)},
$$

where $\kappa > 0$ controls the sharpness of the distribution and $\lambda>0$ balances global versus personalized preferences. The same scoring function instantiates all supported tasks:

- **Outfit Compatibility** — evaluates the holistic coherence of an outfit: $f(o;\boldsymbol{\omega},\boldsymbol{\mu},\kappa) = p(\boldsymbol{z};\boldsymbol{\omega},\boldsymbol{\mu},\kappa)$
- **Outfit Completion** — ranks candidate items for a missing slot: $f(x_i;o_{-i};\boldsymbol{\omega},\boldsymbol{\mu},\kappa) = p(\boldsymbol{h}^{(c)}_{i};\boldsymbol{\omega},\boldsymbol{\mu},\kappa)$
- **Outlier Detection** — pinpoints the least compatible item within an outfit by selecting the minimum-scoring candidate
- **Fast User Adaptation** — adapts the model to a brand-new user from only a handful of outfits ($\texttt{n\_outfits}$), without gradient-based retraining (`closed_form`, `shrink`, `omega_only`) or with lightweight fine-tuning (`finetune`)

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

```bash
git clone https://github.com/lzcn/outfit-recommendation.git
cd outfit-recommendation

conda create -n lpae python=3.10 -y
conda activate lpae

# install PyTorch matching your CUDA version from https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Data Preparation

Datasets are preprocessed with our companion toolkit [lzcn/outfit-datasets](https://github.com/lzcn/outfit-datasets). Please follow its per-dataset instructions to download and process the raw data, then place (or symlink) the processed outputs under `data/` as below:

```text
data/
├── polyvore-u/                  # PolyVore-U (used by polyvore-519 / 630 / 53 / 32 configs)
│   ├── features/
│   │   ├── images/              # raw-image LMDB (data.mdb)
│   │   └── resnet34/            # ResNet-34 feature LMDB (data.mdb)
│   └── original/
│       ├── tuples_519/          # per-user-split tuples: train_pos/, train_neg/, valid_*/, test_*/...
│       ├── tuples_630/
│       ├── tuples_53/
│       └── tuples_32/
├── iqon-3000/                   # IQON3000 (used by iqon-550 / iqon-58 configs)
│   ├── features/...
│   └── original/tuples_550/...
└── maryland-polyvore/
    ├── features/...
    └── original/...
```

Available dataset configurations:

| Config                 | Dataset          | Users | Max items | Description                        |
| ---------------------- | ---------------- | ----- | --------- | ---------------------------------- |
| `data=polyvore-519`    | PolyVore-U       | 519   | 4         | Personalized split                 |
| `data=polyvore-630`    | PolyVore-U       | 630   | 4         | Personalized split                 |
| `data=polyvore-53`     | PolyVore-U       | 53    | 3         | New-user evaluation split          |
| `data=polyvore-32`     | PolyVore-U       | 32    | 4         | New-user evaluation split          |
| `data=iqon-550`        | IQON3000         | 550   | 8         | Personalized split                 |
| `data=iqon-58`         | IQON3000         | 58    | 8         | New-user evaluation split          |
| `data=maryland_polyvore` | Maryland-PolyVore | –  | 8         | Non-personalized compatibility split |
| `data=polyvore-outfits`  | PolyVore (UIUC)   | –  | 19        | Non-disjoint FITB split            |
| `data=polyvore-outfits-d` | PolyVore (UIUC)  | –  | 16        | Disjoint FITB split                |

All processed data (features + tuples) can be downloaded from our Hugging Face dataset repository [lzcn/outfit-datasets](https://huggingface.co/datasets/lzcn/outfit-datasets) and placed under `data/`.

If your data lives elsewhere, override the root on the command line:

```bash
python main.py ... data.data_root=/absolute/path/to/polyvore-u
```

## Quick Start

All experiments are driven by Hydra through `main.py`. Experiment presets live in [`configs/experiment`](configs/experiment).

### Outfit Compatibility

Train:

```bash
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_compatibility \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_o=true \
    model.use_g_lambda=true \
    model.kappa=10.0 \
    hydra.run.dir=outputs/compatibility/${dataset}/resnet34-nn-num-points-8-dense-o-g-lambda \
    task.device=0
done
```

Evaluate (AUC / AUC-User):

```bash
experiment_name="resnet34-nn-num-points-8-dense-o-g-lambda"
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_compatibility \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_o=true \
    model.use_g_lambda=true \
    model.kappa=6.0 \
    task.num_runs=10 \
    task/eval="[compatibility]" \
    task.load_pretrained=outputs/compatibility/${dataset}/${experiment_name}/checkpoints/compatibility_best.pt \
    hydra.run.dir=outputs/compatibility/${dataset}/${experiment_name}/evaluation \
    task.device=0
done
```

### Outfit Completion (FITB)

Train from scratch or from a compatibility checkpoint:

```bash
python main.py \
  +experiment=lpae_resnet34_completion \
  data=polyvore-519 \
  model.num_points=0 \
  model.use_dense_h=true \
  model.use_dense_o=true \
  model.use_g_anchors=false \
  hydra.run.dir=outputs/completion/resnet34-nn-use-dense-h-use-dense-o \
  task.device=0
```

Evaluate:

```bash
base_dir="outputs/completion"
exp_name="resnet34-nn-num-points-8-use-z-score-use-dense-h-use-dense-o-no-g-anchors"
ckpt_name="completion_best.pt"
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_completion \
    data=${dataset} \
    data.test.dataset.data_param.num_answers=4 \
    model.num_points=8 \
    model.use_g_anchors=false \
    model.use_dense_o=true \
    model.use_dense_h=true \
    task.stage=test \
    task.load_pretrained=${base_dir}/${dataset}/${exp_name}/checkpoints/${ckpt_name} \
    hydra.run.dir=${base_dir}/${dataset}/${exp_name}/evaluate \
    task.device=0
done
```

### Group FITB

Train from scratch:

```bash
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_group_fitb \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_d=true \
    model.use_g_lambda=true \
    model.kappa=10.0 \
    hydra.run.dir=outputs/group_fitb/${dataset}/resnet34-nn-num-points-8-dense-d-g-lambda \
    task.device=0
done
```

Fine-tune from a compatibility checkpoint by adding:

```bash
model=lpae/model_m model.use_dense_h=true +model.fine_tune_h=true \
task.load_pretrained=outputs/compatibility/${dataset}/resnet34-nn-num-points-8-dense-o-g-lambda/checkpoints/compatibility_best.pt
```

Evaluate:

```bash
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_group_fitb \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_o=false \
    task.stage=test \
    task.load_pretrained=outputs/group_fitb/${dataset}/resnet34-nn-num-points-8/checkpoints/group_fitb_best.pt \
    hydra.run.dir=outputs/group_fitb/${dataset}/resnet34-nn-num-points-8/evaluation \
    task.device=0
done
```

### Outlier Detection

Train:

```bash
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_outlier \
    data=${dataset} \
    model.num_points=8 \
    model.use_g_lambda=true \
    model.use_dense_d=true \
    hydra.run.dir=outputs/outlier/${dataset}/resnet34-nn-num-points-8-dense-d-g-lambda \
    task.device=0
done
```

Evaluate:

```bash
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_outlier \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_d=true \
    model.use_g_lambda=true \
    task.stage=test \
    task.load_pretrained=outputs/outlier/${dataset}/resnet34-nn-num-points-8-dense-d-g-lambda/checkpoints/outlier_best.pt \
    hydra.run.dir=outputs/outlier/${dataset}/resnet34-nn-num-points-8-dense-d-g-lambda/evaluation \
    task.device=0
done
```

### Fast Adaptation to New Users

Adapt a pretrained model to unseen users using only a few support outfits (`+task.n_outfits`). Switch the strategy with `task.new_user` (`closed_form`, `shrink`, `omega_only`, `adhoc`, `finetune`):

```bash
python main.py \
  +experiment=lpae_resnet34_new_user \
  data=polyvore-32 \
  model.num_points=8 \
  model.use_dense_d=false \
  model.use_dense_o=true \
  model.use_g_lambda=true \
  model.kappa=10 \
  task.new_user=closed_form \
  +task.n_outfits=10 \
  task/eval='[compatibility]' \
  task.load_pretrained=outputs/compatibility/polyvore-519/resnet34-nn-num-points-8-dense-o-g-lambda/checkpoints/compatibility_best.pt \
  hydra.run.dir=outputs/compatibility/polyvore-32/eval/new-user-10-closed_form \
  task.device=0
```

Group-FITB variant:

```bash
python main.py \
  +experiment=lpae_resnet34_new_user \
  data=polyvore-53 \
  model.num_points=8 \
  model.use_dense_d=false \
  model.use_dense_o=false \
  model.use_g_lambda=true \
  model.kappa=10.0 \
  +task.n_outfits=5 \
  task/eval='[group_fitb]' \
  task.load_pretrained=outputs/group_fitb/polyvore-630/resnet34-nn-num-points-8-g-lambda-pretrained/checkpoints/group_fitb_best.pt \
  hydra.run.dir=outputs/group_fitb/polyvore-53/eval/new-user-5 \
  task.device=0
```

### Baselines

BiLSTM and CSNet baselines are included:

```bash
python main.py +experiment=lstm_resnet34 data=polyvore-519 task.device=0
python main.py +experiment=csnet_resnet34 data=polyvore-519 task.device=0
```

## Project Structure

```text
outfit-recommendation/
├── main.py                  # entry point (Hydra CLI)
├── configs/                 # Hydra configuration groups
│   ├── default.yaml         # base config
│   ├── data/                # datasets, readers & tuple-generation strategies
│   ├── experiment/          # ready-to-run experiment presets
│   ├── model/               # backbones & models
│   ├── optimizer|scheduler/
│   └── task/                # trainers / evaluators wiring
├── datasets/                # data loading, tuple generation, metrics
├── models/                  # LPAE variants + BiLSTM / CSNet baselines
├── tasks/                   # trainers & evaluators (registry based)
│   ├── trainers/            # compatibility, completion, outlier, new-user, ...
│   └── evaluators/
└── utils/                   # logging, checkpointing, IO helpers
```

## License

This project is released under the [MIT License](LICENSE).
Dataset preprocessing relies on [lzcn/outfit-datasets](https://github.com/lzcn/outfit-datasets); please follow the licenses of PolyVore, IQON3000, and Maryland-PolyVore for the underlying data.
