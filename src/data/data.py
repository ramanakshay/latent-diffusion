from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset


class Data:
    def __init__(self, config, accelerator):
        self.config = config.data
        self.dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

        preprocess = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        self.dataset.set_transform(transform)

        self.dataloader = DataLoader(self.dataset, self.config.batch_size, shuffle=True)

        self.dataloader = accelerator.prepare(self.dataloader)

