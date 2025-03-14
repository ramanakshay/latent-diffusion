from data.data import Data
from model.model import DiffusionModel
import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

def setup():
    # (Optional) Initial setup (can return process state (for ddp), amp context, etc.)
    # random seed
    torch.manual_seed(42)

def cleanup():
    # (Optional) Cleanup code (ex. destroy_process_group)
    pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##
    setup()
    
    ## DATA ##
    data = Data(config)
    dataset = data.dataset
    print('Data Loaded.')

    ## MODEL ##
    model = DiffusionModel(config)
    print('Model Created.')

    sample_image = dataset[0]["images"].unsqueeze(0)
    print("Input shape:", sample_image.shape)

    noise = torch.randn(sample_image.shape)
    timesteps = torch.LongTensor([50])
    noisy_image = model.noise_scheduler.add_noise(sample_image, noise, timesteps)
    print("Output shape:", noisy_image.shape)

    # ## ALGORITHM ##
    # print('Running Algorithm.')
    # alg = Trainer(data, model, config)
    # alg.run()
    # print('Done!')

    ## CLEANUP ##
    cleanup()

if __name__ == "__main__":
    main()
