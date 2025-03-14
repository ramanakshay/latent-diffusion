from data.data import Data
from model.model import DiffusionModel
from algorithm.train import Trainer

from accelerate import Accelerator
import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

def setup(config):
    # (Optional) Initial setup (can return process state (for ddp), amp context, etc.)
    # random seed
    config = config.accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.grad_accum,
    )
    torch.manual_seed(42 + accelerator.process_index)

    return accelerator

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##
    accelerator = setup(config)
    
    ## DATA ##
    data = Data(config, accelerator)
    dataset = data.dataset
    print('Data Loaded.')

    ## MODEL ##
    model = DiffusionModel(config, accelerator)
    print('Model Created.')

    # ## ALGORITHM ##
    print('Running Algorithm.')
    alg = Trainer(data, model, config, accelerator)
    alg.run()
    print('Done!')

if __name__ == "__main__":
    main()
