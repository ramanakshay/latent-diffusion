from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import torch

class DiffusionModel:
    def __init__(self, config, accelerator):
        self.config = config.model
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.unet = UNet2DModel(
            sample_size=self.config.image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        if self.config.to_compile:
            self.unet = torch.compile(self.unet)
            if accelerator.is_main_process:
                print('Model Compiled.')

        self.unet = accelerator.prepare(self.unet)

    def add_noise(self, clean_images, noise, timesteps):
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        return noisy_images

    def predict_noise(self, noisy_images, timesteps, return_dict=False):
        return self.unet(noisy_images, timesteps, class_labels=None, return_dict=return_dict)[0]
