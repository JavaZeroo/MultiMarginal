import gc
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn
from tqdm.auto import tqdm
from pathlib import Path
from utils.unet import UNetModel
import hydra
from omegaconf import DictConfig
from utils.mydataset import imgDataset, get_dl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train(model, train_dl, optimizer, scheduler, loss_fn, epoch_iterator):
    losses = 0
    for training_data in train_dl:
        x = training_data['x'].to(device)
        y = training_data['y'].to(device)
        t = training_data['t'].to(device)
        direction = training_data['direction'].to(device)
        if 'status' in training_data:
            status = training_data['status'].to(device)
        else:
            status = None
        pred = model(x, t, direction, status)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses = loss.item()
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[-1]['lr']
        epoch_iterator.set_description("Training (lr: %2.5f)  (loss=%2.5f)" % (cur_lr, losses))
    return losses

@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    print(cfg)
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    image_size = cfg.dataset.image_size
    image_channels = cfg.dataset.image_channels

    ############ Training setting ############
    checkpoint_path = None if cfg.checkpoint_path is None else Path(cfg.checkpoint_path)
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    refresh_rate = cfg.refresh_rate
    log_dir = Path('.')
    ##########################################

    # Dataset
    data1_path = Path(cfg.dataset.data_path)
    vaild_font_names = [file.stem for file in data1_path.glob("*.png")]

    data1_dataset = imgDataset(data1_path, vaild_font_names, image_size, False, cfg.train_nums)
    print("Length of dataset: ", len(data1_dataset))

    train_tgt_imgs_1 = torch.stack([data1_dataset[i] for i in range(cfg.train_nums)], dim=0)
    gauss_samples = torch.randn_like(train_tgt_imgs_1)

    dists = [gauss_samples, train_tgt_imgs_1]
    train_pair_list = [(0, 1)]

    model_list = []

    for index, pair in enumerate(train_pair_list):
        print("Training Pair: ", pair)
        num_channels = cfg.model.model_channels
        num_res_blocks = cfg.model.num_res_blocks
        num_heads = cfg.model.num_heads
        num_heads_upsample = cfg.model.num_heads_upsample
        attention_resolutions = cfg.model.attention_resolutions
        dropout = cfg.model.dropout
        use_checkpoint = cfg.model.use_checkpoint
        use_scale_shift_norm = cfg.model.use_scale_shift_norm

        channel_mult = tuple(cfg.model.channel_mult)

        attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]
        kwargs = {
            "in_channels": image_channels,
            "model_channels": num_channels,
            "out_channels": image_channels,
            "num_res_blocks": num_res_blocks,
            "attention_resolutions": tuple(attention_ds),
            "dropout": dropout,
            "channel_mult": channel_mult,
            "num_classes": None,
            "use_checkpoint": use_checkpoint,
            "num_heads": num_heads,
            "num_heads_upsample": num_heads_upsample,
            "use_scale_shift_norm": use_scale_shift_norm,
        }
        print(kwargs)
        model = UNetModel(**kwargs)

        src_id, tgt_id = pair
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
        loss_fn = nn.MSELoss()
        loss_list = []

        if checkpoint_path is not None:
            laod_checkpoint_from = checkpoint_path / f"model_{index}.pt"
            print(f'Load Checkpoint from {laod_checkpoint_from}')
            model.load_state_dict(torch.load(laod_checkpoint_from))
            
        epoch_iterator = tqdm(range(epochs), desc="Training (lr: X)  (loss= X)", dynamic_ncols=True)
        model.train()
        
        model = model.cuda()
        for e in epoch_iterator:
            if e == 0 or (refresh_rate != 0 and e%refresh_rate==0):
                train_dl = get_dl(src_id, tgt_id, dists, cfg.SIGMA, batch_size, cfg.EPSILON)
            now_loss = train(model ,train_dl, optimizer, scheduler, loss_fn, epoch_iterator)
            loss_list.append(now_loss)
            cur_lr = optimizer.param_groups[-1]['lr']
            epoch_iterator.set_description("Training (lr: %2.5f)  (loss=%2.5f)" % (cur_lr, now_loss))
        plt.figure()
        plt.plot(loss_list)
        plt.savefig(log_dir / f'loss_{src_id}_{tgt_id}.png')
        plt.gca().cla()
        epoch_iterator.close()
        torch.save(model.state_dict(), log_dir / f"model_{index}.pt")
        del train_dl
        gc.collect()
        torch.cuda.empty_cache()
    model_list.append(model)
    plt.show()

if __name__ == "__main__":
    main()