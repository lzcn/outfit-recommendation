# Unified Metric Learning for Personalized Fashion Recommendations with Fast User Adaptation

PyTorch implementation of the IEEE TMM 2026 paper.

Z. Lu, Y. Hu, C. Yu, Y. Chen, and B. Zeng, "Unified metric learning for personalized fashion recommendations with fast user adaptation," IEEE Transactions on Multimedia, pp. 1-14, 2026. doi: 10.1109/TMM.2026.3694502

## Citation

```bibtex
@article{UnifiedMetricLearningLu26,
  title = {Unified Metric Learning for Personalized Fashion Recommendations with Fast User Adaptation},
  author = {Lu, Zhi and Hu, Yang and Yu, Cong and Chen, Yan and Zeng, Bing},
  year = 2026,
  journal = {IEEE Transactions on Multimedia},
  pages = {1--14},
  doi = {10.1109/TMM.2026.3694502}
}
```

## Install

Requires Python >= 3.10. Everything runs from this repository alone; no other local packages or repositories are needed.

```bash
conda create -n lpae python=3.10 -y
conda activate lpae
pip install torch torchvision   # see https://pytorch.org for the CUDA build you need
pip install -r requirements.txt
```

## Data

Place the processed datasets (polyvore-u, iqon-3000, maryland-polyvore, polyvore-outfits) under `data/` in this repository. If the data lives elsewhere, pass `data.data_root=/path/to/<dataset>` on the command line.

## Run

Full example on polyvore-519: train outfit compatibility, then evaluate.

Train:

```bash
python main.py \
  +experiment=lpae_resnet34_compatibility \
  data=polyvore-519 \
  model.num_points=8 model.use_dense_o=true model.use_g_lambda=true model.kappa=10.0 \
  hydra.run.dir=outputs/compatibility/polyvore-519 \
  task.device=0
```

Evaluate:

```bash
python main.py \
  +experiment=lpae_resnet34_compatibility \
  data=polyvore-519 \
  model.num_points=8 model.use_dense_o=true model.use_g_lambda=true model.kappa=6.0 \
  task.num_runs=10 task/eval="[compatibility]" \
  task.load_pretrained=outputs/compatibility/polyvore-519/checkpoints/compatibility_best.pt \
  hydra.run.dir=outputs/compatibility/polyvore-519/evaluation \
  task.device=0
```

Other tasks (outfit completion, group FITB, outlier detection, fast adaptation to new users) work the same way; see `configs/experiment/`.

## License

MIT.
