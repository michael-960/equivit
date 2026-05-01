import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import lightning as L



@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    # 1. Build the Science (The LightningModule we just designed)
    # Hydra will recursively build the nn.Module, criterion, and factories first!
    module = instantiate(cfg.module)
    
    # 2. Build the Data (LightningDataModule)
    # datamodule = instantiate(cfg.datamodule)
    
    # 3. Build the Specialists (Callbacks & Loggers)
    # We use .values() to iterate over the dictionary of callbacks in the YAML
    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", {}).values()]
    
    # Instantiate the logger if defined, otherwise let Lightning use its default
    logger = instantiate(cfg.logger) if "logger" in cfg else True
    
    # 4. Build the Engineer (The Trainer)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    print(module.model)


    print("done building, starting training loop...")

    # return here to test the building process without starting training
    return 
    
    # 5. Start the Engine!
    # By passing ckpt_path, we enable the seamless pausing/resuming we discussed
    trainer.fit(
        model=module, 
        datamodule=datamodule,
        ckpt_path=cfg.get("ckpt_path")
    )
    
    # Optional: Automatically run the test set after training completes
    # trainer.test(model=module, datamodule=datamodule)

if __name__ == "__main__":
    main()