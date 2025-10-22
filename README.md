# Outfit Recommendation

## Quick Start

### Compatibility Task

#### Train Compatibility Model

```bash
num_points=8
exp_name="resnet34-nn-num-points-${num_points}-dense-o-g-lambda-kappa-6.0"
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_compatibility \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_o=true \
    model.use_g_lambda=true \
    model.kappa=6.0 \
    hydra.run.dir=outputs/compatibility/${dataset}/${exp_name} \
    task.device=7
done
```

#### Evaluate Compatibility Model

```bash
base_dir="outputs/compatibility"
exp_name="resnet34-nn"
ckpt_name="compatibility_best.pt"
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_compatibility \
    data=$dataset \
    model.num_points=0 \
    model.use_dense_o=false \
    hydra.run.dir=${base_dir}/${dataset}/${exp_name}/evaluate \
    task=base \
    task.num_runs=10 \
    "task/eval=[compatibility]" \
    task.load_pretrained=${base_dir}/${dataset}/${exp_name}/checkpoints/${ckpt_name} \
    task.device=2
done
```

> Note: `task=base` uses the default testing mode without extra configurations.
> We can also use `task.stage=test` to enable testing mode.

### Completion Task

#### Train Completion Model

```bash
python main.py -m \
  +experiment=lpae_resnet34_completion \
  data=polyvore-519,polyvore-630,iqon-550 \
  model.num_points=0 \
  model.use_dense_h=true \
  model.use_dense_o=true \
  model.use_g_anchors=false \
  hydra.run.dir=outputs/completion/resnet34-nn-use-dense-h-use-dense-o \
  task.device=4
```

```bash
%  polyvore-630 iqon-550
for data in polyvore-519; do
  python main.py \
    +experiment=lpae_resnet34_completion \
    data=$data \
    model.num_points=8 \
    model.use_dense_h=true \
    model.use_dense_o=true \
    model.use_g_lambda=true \
    model.use_g_anchors=false \
    hydra.run.dir=outputs/completion/${data}/resnet34-nn-num-points-8-use-dense-h-use-dense-o-no-g-anchors-pretrained-lambda-g \
    task.load_pretrained=outputs/compatibility/${data}/resnet34-nn-num-points-8-lambda-g/checkpoints/compatibility_best.pt
done
```

```bash
python main.py \
  +experiment=lpae_resnet34_completion \
  data=polyvore-519 \
  model.num_points=4 \
  model.use_z_score=true \
  model.use_dense_h=true \
  model.use_dense_o=true \
  model.use_g_anchors=false \
  hydra.run.dir=outputs/completion/polyvore-519/resnet34-nn-num-points-4-use-z-score-use-dense-h-use-dense-o-no-g-anchors
```

Previous Paper on

```bash
python main.py\
    +experiment=lpae_resnet34_completion \
    data=polyvore-519 \
    data.test.dataset.data_param.num_answers=4 \
    model.num_points=0 \
    model.use_g_anchors=false \
    model.use_dense_o=true \
    model.use_dense_h=true \
    task.stage=test \
    task.load_pretrained=summaries/polyvore-519/polyvore-519-lpae-c-resnet34-nn-learnable-CE-kappa-10-only-u-anchors-1-completion-dense-h-num-answers-32/checkpoints/net_best.pt
```

    # task.load_pretrained=checkpoints/lpaenet/completion-polyvore-519-resnet34-nn-num-points-4-use-dense-h.pt

python main.py -m \
 +experiment=lpae_resnet34_completion \
 data=polyvore-519 \
 data.test.dataset.data_param.num_answers=4,10 \
 model.num_points=0 \
 model.use_g_anchors=false \
 model.use_dense_o=true \
 model.use_dense_h=true \
 model.use_z_score=true \
 task=completion \
 task.stage=test \
 task.load_pretrained=summaries/polyvore-519/polyvore-519-lpae-c-resnet34-nn-learnable-CE-kappa-10-only-u-anchors-1-completion-dense-h-num-answers-32/checkpoints/net_best.pt

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
    hydra.run.dir=${base_dir}/${dataset}/${exp_name}/evaluate
done
```

python main.py \
 +experiment=lpae_resnet34_completion \
 data=polyvore-519 \
 data.test.dataset.data_param.num_answers=4 \
 model.num_points=0 \
 model.use_g_anchors=true \
 model.use_dense_o=true \
 model.use_dense_d=true \
 model.use_dense_h=true \
 model.use_g_lambda=true \
 model.use_z_score=true \
 task=completion \
 task.stage=test \
 task.load_pretrained=summaries/polyvore-519/polyvore-519-lpae-g-resnet34-nn-learnable-CE-kappa-10-num-anchors-2-num-neg-32-lr-0.01-lambda-multi-task/checkpoints/net_latest.pt

### Group FITB task

#### Train Group FITB Model

```bash
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_group_fitb \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_d=true \
    model.use_g_lambda=true \
    model.type_encoding="none" \
    model.kappa=10.0 \
    hydra.run.dir=outputs/group_fitb/${dataset}/resnet34-nn-num-points-8-dense-d-g-lambda-type_encoding-none \
    task.load_pretrained=outputs/compatibility/${dataset}/resnet34-nn-num-points-8-dense-o-g-lambda/checkpoints/compatibility_best.pt \
    task.device=1
done
python main.py -m \
  +experiment=lpae_resnet34_group_fitb \
  data=polyvore-519,polyvore-630,iqon-550 \
  model.num_points=8 \
  model.use_dense_o=true \
  hydra.sweep.dir=outputs/group_fitb/resnet34-nn-num-points-8
```

python main.py \
 +experiment=lpae_resnet34_group_fitb \
 data=iqon-550 \
 model.use_dense_o=false \
 model.use_g_lambda=true \
 hydra.run.dir=outputs/group_fitb/iqon-550/resnet34-nn-num-points-8-g-lambda-pretrained \
 task.load_pretrained=outputs/compatibility/iqon-550/resnet34-nn-num-points-8-g-lambda/checkpoints/compatibility_best.pt

#### Evaluate Group FITB Model

```bash
python main.py \
  +experiment=lpae_resnet34_group_fitb \
  data=iqon-550 \
  model.num_points=8 \
  model.use_dense_o=false \
  task.stage=test \
  task.load_pretrained=outputs/group_fitb/iqon-550/resnet34-nn-num-points-8-pretrained/checkpoints/group_fitb_best.pt
```

#### Evalute New User

```zsh
python main.py \
    +experiment=lpae_resnet34_new_user \
    data=polyvore-53 \
    model=lpae/model_g \
    model.num_points=8 \
    model.use_dense_d=false \
    model.use_dense_o=false \
    model.use_g_lambda=true \
    model.kappa=10.0 \
    task/eval='[group_fitb]' \
    +task.n_outfits=5 \
    hydra.run.dir=outputs/group_fitb/polyvore-630/resnet34-nn-num-points-8-g-lambda-pretrained/eval/evaluate-new-user-5 \
    task.load_pretrained=outputs/group_fitb/polyvore-630/resnet34-nn-num-points-8-g-lambda-pretrained/checkpoints/group_fitb_best.pt
```

### Outlier Task

#### Train Outlier Model

```bash
python main.py \
    +experiment=lpae_resnet34_outlier \
    data=polyvore-519 \
    model.num_points=8 \
    model.use_g_lambda=true \
    model.use_dense_d=false \
    hydra.run.dir=outputs/outlier/polyvore-519/resnet34-nn-num-points-8-g-lambda-pretrained \
    task.load_pretrained=outputs/compatibility/polyvore-519/resnet34-nn-num-points-8-g-lambda/checkpoints/compatibility_best.pt
```

python main.py\
 +experiment=lpae_resnet34_outlier \
 data=iqon-550 \
 model.num_points=0 \
 model.use_dense_d=true \
 hydra.run.dir=outputs/outlier/iqon-550/resnet34-nn-num-points-8-use-dense-d-old-pretrained \
 task.load_pretrained=summaries/iqon-550/iqon-550-lpae-g-resnet34-nn-learnable-CE-kappa-10-num-anchors-2-num-neg-32-lr-0.01-lambda/checkpoints/best_model.pt \
 task.device=7

##### Evaluate outlier detection performance

```bash
base_dir="outputs/outlier"
exp_name="resnet34-nn-num-points-8-use-dense-d-pretrained"
ckpt_name="outlier_best.pt"
for dataset in polyvore-519 polyvore-630 iqon-550; do
  python main.py \
    +experiment=lpae_resnet34_outlier \
    data=$dataset \
    model.num_points=8 \
    model.use_dense_d=true \
    hydra.run.dir=${base_dir}/${dataset}/${exp_name}/evaluate \
    task=base \
    task.num_runs=10 \
    "task/eval=[outlier]" \
    task.load_pretrained=${base_dir}/${dataset}/${exp_name}/checkpoints/${ckpt_name} \
    task.device=2
done
```

```bash
python main.py \
    +experiment=lpae_resnet34_outlier \
    data=polyvore-519 \
    model.num_points=8 \
    model.use_dense_d=false \
    model.use_dense_o=true \
    task.stage=test \
    task.load_pretrained=summaries/polyvore-519/polyvore-519-lpae-d-resnet34-nn-learnable-CE-kappa-10-use-all-anchors-2-outfit-outlier-no-z-score-num-points-8/checkpoints/net_best.pt
```

#### New User Task

```zsh
python main.py \
    +experiment=lpae_resnet34_new_user \
    data=iqon-58 \
    model=lpae/model_d \
    model.num_points=8 \
    model.use_dense_d=false \
    model.use_dense_o=false \
    model.use_g_lambda=false \
    model.kappa=6 \
    task/eval='[outlier]' \
    +task.n_outfits=10 \
    task.new_user=closed_form \
    hydra.run.dir=outputs/outlier/iqon-550/resnet34-nn-num-points-8-kappa-6.0-pretrained/eval/evaluate-new-user-10 \
    task.load_pretrained=outputs/outlier/iqon-550/resnet34-nn-num-points-8-kappa-6.0-pretrained/checkpoints/outlier_best.pt
```

python main.py \
 +experiment=lpae_resnet34_outlier \
 data=iqon-550 \
 model.num_points=8 \
 model.use_g_lambda=true \
 model.use_dense_d=false \
 hydra.run.dir=outputs/outlier/resnet34-nn-num-points-8-g-lambda-pretrained \
 task.load_pretrained=outputs/compatibility/iqon-550/resnet34-nn-num-points-8-lambda-g/checkpoints/compatibility_best.pt
