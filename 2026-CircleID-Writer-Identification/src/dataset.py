# --- Dataset and Augmentation Logic ---
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CircleDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Construct the full image path
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['image_path'])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Test mode doesn't have labels
        if self.is_test:
            return image, self.df.iloc[idx]['image_id']
        
        return image, self.df.iloc[idx]['label']

# Data Augmentation: Crucial for "Simple" shapes like circles
def get_transforms(img_size, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(360), # Circles can be drawn at any angle
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Account for different lighting/scanners
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
