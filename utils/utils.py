import torch
import imageio
import shutil

from pathlib import Path
from rich.progress import track
import matplotlib.pyplot as plt


# 生成二维Brownian bridge
def gen_bridge_2d(x, y, ts, T, sigma):
    sigma = sigma
    x_shape = x.shape
    bridge = torch.zeros((ts.shape[0], *x_shape))
    drift = torch.zeros((ts.shape[0], *x_shape))
    bridge[0] = x
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]
        dydt = (y - bridge[i]) / (T - ts[i])
        drift[i, :] = dydt
        diffusion = sigma * torch.sqrt(dt) * torch.randn_like(dydt)
        bridge[i+1] = bridge[i] + dydt * dt
        bridge[i+1, :] += diffusion
    return bridge, drift

def gen_2d_data(source_dist, target_dist, sigma, epsilon, T=1):
    ts = torch.arange(0, T+epsilon, epsilon)
    # source_dist = torch.Tensor(source_dist)
    # target_dist = torch.Tensor(target_dist)
    assert source_dist.shape == target_dist.shape
    bridge, drift = gen_bridge_2d(source_dist, target_dist, ts, T=T, sigma=sigma)
    return ts, bridge, drift


def draw_comapre_split(test_pred_bridges):
    n_sub_interval = len(test_pred_bridges)+1
    fig, axs = plt.subplots(1, n_sub_interval, figsize=(5*n_sub_interval, 5))
    def plot_test_pred_bridges(sub_axs, data):
        for i in range(n_sub_interval):
            now = data[i][0, :] if i != n_sub_interval-1 else data[i-1][-1, :]
            combined_image = torch.cat([torch.cat([now[j] for j in range(k, k+5)], dim=2) for k in range(0, 25, 5)], dim=1)
            combined_image = (combined_image-combined_image.min())/(combined_image.max()-combined_image.min())
            
            sub_axs[i].imshow(combined_image.permute(1,2,0).numpy(), cmap='gray')
            
    plot_test_pred_bridges(axs, test_pred_bridges)

    # set tight layout
    fig.tight_layout()
    
    # fig
    fig.show()
    
    return fig

def save_gif_frame(bridge, save_path=None, name='brownian_bridge.gif'):
    assert save_path is not None, "save_path cannot be None"
    save_path = Path(save_path)
    if len(bridge) > 200:
        # downsample to 100
        downsample_rate = len(bridge) // 100
        bridge = bridge[::downsample_rate]

    temp_dir = save_path / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    frame = 0
    
    for i in track(range(bridge.shape[0]), description="Processing image"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        now = bridge[i, :]
        combined_image = torch.cat([torch.cat([now[j] for j in range(k, k+5)], dim=2) for k in range(0, 25, 5)], dim=1)
        ax.imshow(combined_image.permute(1,2,0).numpy(), cmap='gray')
        fig.savefig(save_path / 'temp' / f'{frame:03d}.png', dpi=100)
        frame += 1
        fig.show()
        plt.close('all')
    frames = []
    for i in range(bridge.shape[0]):
        frame_image = imageio.imread(save_path / 'temp' / f'{i:03d}.png')
        frames.append(frame_image)
    imageio.mimsave(save_path / name, frames, duration=0.2)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)