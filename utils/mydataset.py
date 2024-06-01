import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import v2 as transforms
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from utils import gen_2d_data
import gc

class BasicDataset(Dataset):
    def __init__(self, ts, bridge, drift, direction, status=None):
        # scaled_tensor = normalized_tensor * 2 - 1
        train_nums = bridge.shape[1]
        self.times = ts[:len(ts)-1].repeat(train_nums,)
        self.positions = torch.cat(torch.split(bridge[:-1, :], 1, dim=1), dim=0)[:, 0]
        self.scores = torch.cat(torch.split(drift[:-1, :], 1, dim=1), dim=0)[:,0]
        self.direction = torch.Tensor([direction])
        self.status = status
        
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        ret = {
            'x': self.positions[index], 
            'y': self.scores[index],
            't': self.times[index],
            'direction': self.direction,
            }

        if self.status is not None:
            ret['status'] = self.status[index]
        return ret



class imgDataset(Dataset):
    def __init__(self, dir, img_names, image_size, sketch, max_load=0):
        """
        TODO sketch need to be rewrite
        """
        assert max_load >= 0 and image_size > 0
        self.dir = Path(dir)
        train_transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            ])
        self.data = []
        img_names = img_names if max_load == 0 else img_names[:max_load]
        for img_name in tqdm(img_names):
            if sketch:
                img_name = f"{img_name}_1.png"
            else:
                img_name = f"{img_name}.png"
            img_path = self.dir / img_name
            img = train_transform(Image.open(img_path).convert('RGB'))
            # convert to gray
            # img = torch.mean(img, dim=0, keepdim=True)
            
            if torch.any(torch.isnan(img)).item():
                print(img_name)
            if img.max() - img.min() == 0:
                print(img_name)
                
            img = 5 * img
            
            self.data.append(img)
            
        self.data = torch.stack(self.data)
        print(self.data.min(), self.data.max())
            
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    

def get_dl(src_id, tgt_id, dists, sigma, batch_size, epsilon):
    src_dist, tgt_dist = torch.Tensor(dists[src_id]),torch.Tensor(dists[tgt_id])
    gc.collect()
    
    print("Generate Forward Data")
    ts, bridge_f, drift_f = gen_2d_data(src_dist, tgt_dist, sigma, epsilon=epsilon, T=1)
    print("Generate Backward Data")
    ts, bridge_b, drift_b = gen_2d_data(tgt_dist, src_dist, sigma, epsilon=epsilon, T=1)

    print(ts.shape, bridge_f.shape, drift_f.shape)
    dataset1 = BasicDataset(ts, bridge_f, drift_f, 0)
    dataset2 = BasicDataset(ts, bridge_b, drift_b, 1)
    combined_dataset = ConcatDataset([dataset1, dataset2])

    train_dl = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    return train_dl