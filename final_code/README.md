# Online Learning Final Code
This folder contains the training, evaluation, and analysis code for the cat-vs-dog online learning study (CIFAR-10 ➜ Oxford-IIIT Pet ➜ STL-10). The scripts share the same dataset roots (`./data` and `./data/oxford-iiit-pet`) and assume CUDA is available but can fall back to CPU.

## File Map
- **File 1 — `online_training_pipeline.py`**: Used for the first stage. It provides the `train_base_and_online()` core and the `run_online_recipes_and_summarize()` function for training configurations on the “online” learning variation created earlier. This code accepts configuration dictionaries (`phaseA/B/C`) and returns a dataset (pandas `DataFrame`) that pairs each configuration with the base/online/third-dataset accuracy results.
- **File 2 — `attacks_second_check_third.py`**: Reuses the model produced by File 1, runs adversarial attacks on the second dataset (Oxford-IIIT Pet) using `torchattacks`, and then checks clean accuracy on the third dataset (STL-10) without attacks. Produces a CSV/`DataFrame` with per-attack drops plus helper tips.
- `online_recipe_builders.py`: Utility builders for composing/changing “online” recipes (mixup/cutmix/PGD-AT/etc.). Use it when you need systematic sweeps before feeding configs to File 1.
- `online_experiments.py`: A more verbose, notebook-friendly version of the pipeline with extra instrumentation (custom augment knobs, spectral norm, temp-scaling, etc.). Handy for debugging or ablation studies outside the main scripted flow.
- `pgd_at_training_with_torchattacks.py`: Standalone PGD-adversarial-training script for CIFAR-10 cat/dog plus automated robustness benchmarking on multiple attacks. Useful for establishing a stronger Phase A backbone before enabling online adaptation.
- `config_grid.ipynb`: Notebook used to prototype/search recipe grids (invokes the builders and pipelines above). Keep large sweeps here so the Python scripts stay lean.

## Recommended Run Sequence
1. **Compose recipes**  
   - Quick start: call `build_online_recipes()` inside File 1.  
   - For richer sweeps, open `online_recipe_builders.py` (or `config_grid.ipynb`) to generate a `dict[str, dict]` of recipes and persist them (JSON/YAML).
2. **Train and log accuracies (File 1)**  
   ```bash
   cd rai_research/final_code
   python online_training_pipeline.py
   ```  
   This trains Phase A (CIFAR-10), performs Phase B online adaptation on Oxford-IIIT Pet, evaluates Phase C on STL-10, and writes `online_results/*.recipe.json` plus a summary `DataFrame`. To plug in custom recipes, import the module elsewhere and call `run_online_recipes_and_summarize(custom_recipes)`.
3. **Adversarial evaluation (File 2)**  
   ```bash
   python attacks_second_check_third.py \
     --eps 0.03137 --steps 10  # defaults: 8/255 and 10 steps
   ```  
   The script imports File 1 to train/load each config, crafts RAW loaders for the second dataset, runs FGSM/PGD/MIFGSM/DIFGSM/TIFGSM/APGD, reports drops, and stores `attack_results/attacks_summary.csv`.
4. **(Optional) Advanced experiments**  
   - Use `online_experiments.py` when you need tighter control over augmentation knobs or to integrate with notebooks.  
   - Run `pgd_at_training_with_torchattacks.py` to pretrain robust models or to benchmark custom recipes with additional attack suites.

## How to Run Everything Cleanly
- **Environment**: `pip install torch torchvision lightning torchmetrics pandas torchattacks`. Add `jupyter` if you plan to execute the notebooks.
- **Data**: All scripts auto-download CIFAR-10, STL-10, and Oxford-IIIT Pet into `./data`. Verify you have enough disk space and, if running offline, pre-populate those folders.
- **Hardware**: CUDA is preferred (multiple scripts auto-enable `torch.cuda.is_available()`), but CPU mode works for correctness checks albeit slower.
- **Repro**: Every script seeds Python/NumPy/PyTorch; set `PYTHONHASHSEED` if you wrap them in other launchers.

With configs defined via the builders, running File 1 followed by File 2 gives you the end-to-end pipeline the project is centered on; the remaining files provide convenience layers for sweeping, debugging, or stress-testing the models.
