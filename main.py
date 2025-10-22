#!/usr/bin/env python
"""
Main entry point for outfit recommendation tasks.

This script handles both training and evaluation of outfit compatibility models
using a clean registry-based architecture.
"""

import hydra
from omegaconf import DictConfig

import utils
from tasks import get_evaluator, get_trainer

logger = utils.get_logger("main")


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run training and/or evaluation based on configuration.

    Args:
        cfg: Hydra configuration object.
    """
    utils.print_config(cfg, save_cfg=True)
    task = cfg.task
    name = task.name
    model = None

    # Training stage
    if task.stage == "train":
        logger.info(f"Starting training for task: {task.name}")
        trainer = get_trainer(task.name, cfg=cfg)
        trainer.train()
        model = trainer.get_best_model()
        logger.info("Training completed successfully")

    # Evaluation stage
    eval_list = task.get("eval")
    if eval_list is None:  # fallback to default using data.test
        logger.info(f"Starting evaluation for task: {task.name}")
        evaluator = get_evaluator(task.name, cfg=cfg)
        metrics = evaluator.evaluate(model=model)
        # Log results
        logger.info("Evaluation completed successfully")
        title = f"{name.capitalize()} Evaluation (Final)"
        utils.logger.log_metrics_table(title=title, metrics=metrics, logger=logger)
    else:  # using eval list to eval multiple tasks
        for name, value in eval_list.items():
            logger.info(f"Starting evaluation for task: {name}")
            evaluator = get_evaluator(name, cfg=cfg)
            metrics = evaluator.evaluate(value.data, model=model)
            # Log results
            logger.info("Evaluation completed successfully")
            title = f"{name.capitalize()} Evaluation (Final)"
            utils.logger.log_metrics_table(title=title, metrics=metrics, logger=logger)


if __name__ == "__main__":
    main()
