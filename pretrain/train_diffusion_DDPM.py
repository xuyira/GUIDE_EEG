from data_preprocess import eegdata_preprocess
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import torch

import torch.nn.functional as F
from diffusers import  UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.data import DataLoader

from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path

from accelerate import notebook_launcher

output_dir = './model_checkpoints/ckpt-DDPM/' #模型保存地址
pic_dir = './model_pic/DDPM/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pic_dir, exist_ok=True)

@dataclass
class TrainingConfig:
    # 图像尺寸
    image_size = (64, 64)
    # 训练批次大小
    train_batch_size = 64
    # 评估批次大小
    eval_batch_size = 64
    # 训练轮数
    num_epochs = 50
    # 梯度累积步数（累计几次梯度更新一次参数）
    gradient_accumulation_steps = 1
    # 学习率
    learning_rate = 1e-4
    # 学习率衰减
    lr_warmup_steps = 500

    save_image_epochs = 10 #    todo
    save_model_epochs = 30

    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = output_dir
    # 是否上传模型到HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "hibiscus/test_model"  # the name of the repository to create on the HF Hub
    hub_private_repo = None
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42

config = TrainingConfig()

train_data, train_label, test_data, test_label = eegdata_preprocess(
    testsubj=1,        # 测试受试者编号 (1-9)
    dataset='bci2a',   # 数据集
    delay=15,          # 时间延迟参数
    embedding=64,      # 嵌入维度
    device='cuda'      # 计算设备
)

"""
plt.figure()
plt.imshow(train_data[0, 0, :, :], aspect='auto')
plt.colorbar()
plt.show()
"""

train_dataloader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=22,  # the number of input channels, 3 for RGB images
    out_channels=22,  # the number of output channels
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

"""
sample_channel = 0
sample_image = train_data[0].unsqueeze(0)
print("输入图像维度：", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)
"""

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def evaluate(config, epoch, pipeline):
    # 从随机噪声中生成一些图像（这是反向扩散过程）。
    # 默认的管道输出类型是 `List[torch.Tensor]`
    # 取sample_channel来展示
    # print("生成图像......")
    sample_channel = 0
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed),  # 使用单独的 torch 生成器来避免回绕主训练循环的随机状态
        output_type="np.array",
    ).images

    print("生成图像维度：", images.shape)
    # (batch, height, width, channel)

    # 将生成的eval_batch_size个图像拼接成一张大图
    pig, ax = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(2):
        for j in range(10):
            ax[i, j].imshow(images[i * 10 + j, :, :, sample_channel], aspect='auto')
            ax[i, j].axis("off")
            ax[i, j].set_title(f"Image {i * 10 + j}")

    plt.savefig(f"{pic_dir}/{epoch:04d}.png", dpi=400)
    plt.close()


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
