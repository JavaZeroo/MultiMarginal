import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import shutil

from tqdm.auto import tqdm
from pathlib import Path
from utils.unet import UNetModel
from rich.progress import track
import hydra
from omegaconf import DictConfig
from utils.mydataset import imgDataset, gen_2d_data
from utils.utils import save_gif_frame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def inference(model, test_ts, test_source_sample, test_num_samples, sigma, reverse=False):
    model.eval()
    model.to(device)
    pred_bridge = torch.zeros(len(test_ts), test_num_samples, test_source_sample.shape[1], test_source_sample.shape[2], test_source_sample.shape[3])
    pred_drift = torch.zeros(len(test_ts)-1, test_num_samples, test_source_sample.shape[1], test_source_sample.shape[2], test_source_sample.shape[3])
    pred_bridge[0, :] = test_source_sample
    with torch.no_grad():
        for i in tqdm(range(len(test_ts) - 1)):
            dt = abs(test_ts[i+1] - test_ts[i])
            direction = torch.ones_like(test_ts[i:i+1]) if reverse else torch.zeros_like(test_ts[i:i+1])
            dydt = model(pred_bridge[i].to(device), test_ts[i:i+1].to(device), direction.to(device), None).to('cpu')
            diffusion = sigma * torch.sqrt(dt) * torch.randn(test_num_samples, test_source_sample.shape[1], test_source_sample.shape[2], test_source_sample.shape[3])
            pred_drift[i, :] = dydt
            pred_bridge[i+1, :] = pred_bridge[i, :] + dydt * dt
            pred_bridge[i+1, :] += diffusion
    return pred_bridge, pred_drift

@hydra.main(config_path="conf", config_name="infer")
def main(cfg: DictConfig):
    print(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    
    image_size = cfg.dataset.image_size
    image_channels = cfg.dataset.image_channels

    experiment_name = "gaussian2mnist"
    log_dir = Path('experiments') / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    data1_path = Path(cfg.dataset.data_path)
    vaild_font_names = [file.stem for file in data1_path.glob("*.png")]
    data1_dataset = imgDataset(data1_path, vaild_font_names, image_size, False)
    print("Length of dataset: ", len(data1_dataset))

    test_samples = cfg.num_samples
    test_P2_samples = torch.stack([data1_dataset[-i] for i in range(test_samples)], dim=0)
    test_P1_samples = torch.randn_like(test_P2_samples)
    test_ts, _, _ = gen_2d_data(test_P1_samples, test_P2_samples, epsilon=cfg.EPSILON, T=1, sigma=cfg.SIGMA)
    print(test_ts.shape)

    model = UNetModel(
        in_channels=image_channels,
        model_channels=cfg.model.model_channels,
        out_channels=image_channels,
        num_res_blocks=cfg.model.num_res_blocks,
        attention_resolutions=tuple(image_size // int(res) for res in cfg.model.attention_resolutions.split(",")),
        dropout=cfg.model.dropout,
        channel_mult=tuple(cfg.model.channel_mult),
        num_classes=None,
        use_checkpoint=cfg.model.use_checkpoint,
        num_heads=cfg.model.num_heads,
        num_heads_upsample=cfg.model.num_heads_upsample,
        use_scale_shift_norm=cfg.model.use_scale_shift_norm
    )

    checkpoint_path = Path(cfg.checkpoint_path)
    model_path = checkpoint_path / f"model_0.pt"
    print(f'Loading model from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    pred_bridge, _ = inference(model, test_ts, test_P1_samples, test_samples, cfg.SIGMA)

    save_gif_frame(pred_bridge, log_dir, name="pred_infer.gif")

if __name__ == "__main__":
    main()
