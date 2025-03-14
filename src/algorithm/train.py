import torch
from torch import nn, optim
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch.nn.functional as F

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(0), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    # test_dir = os.path.join(config.output_dir, "samples")
    # os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{epoch:04d}.png")

class Trainer:
    def __init__(self, data, model, config, accelerator):
        self.model = model
        self.data = data
        self.config = config.algorithm

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.unet.parameters(), lr=self.config.learning_rate)
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.data.dataloader) * self.config.num_epochs),
        )

        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator

    def run(self):
        global_step = 0
        # Now you train the model
        dataloader = self.data.dataloader
        accelerator = self.accelerator
        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                clean_images = batch["images"]
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.model.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                    dtype=torch.int64
                )
                # forward diffusion
                noisy_images = self.model.add_noise(clean_images, noise, timesteps)
                with accelerator.accumulate(self.model.unet):
                    # Predict the noise residual
                    noise_pred = self.model.predict_noise(noisy_images, timesteps, return_dict=False)
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.model.unet.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": self.scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                # accelerator.log(logs, step=global_step)
                global_step += 1
            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(self.model.unet), scheduler=self.model.noise_scheduler)
                if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                    evaluate(self.config, epoch, pipeline)
                # if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                #     pipeline.save_pretrained(config.output_dir)


