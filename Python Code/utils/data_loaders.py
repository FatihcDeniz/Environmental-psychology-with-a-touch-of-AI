from torch.utils.data import Dataset, DataLoader
import torch
import os 
from PIL import Image
import pandas as pd

class load_images(Dataset):
    # Dataloader for extracting embeddings
    """
    Torch Dataset that loads images from one folder or a folder tree.
    """
    def __init__(self, img_dir, transform, multiple_folder = False):
        """
        Initialize the object and store configuration required by later calls.
        """
        self.img_dir = img_dir                        # Store the path to the image directory
        self.transform = transform                    # Store the transformation function
        
        self.images = []
        
        if multiple_folder:
            for folder in os.listdir(self.img_dir):
                image_folder  = os.path.join(img_dir, folder)
                for image in os.listdir(image_folder):
                    if image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        self.images.append(os.path.join(image_folder, image))
        else:
            for image in os.listdir(self.img_dir):
                if image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    self.images.append(os.path.join(self.img_dir, image))

    def __len__(self):
        """
        Return the number of discovered images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        # Construct the full image file path using the image name from the DataFrame
        """
        Load, transform, and return one image with its path.
        """
        img_name = self.images[idx]

        # Load the image and ensure it is in RGB format
        image = Image.open(img_name).convert('RGB')

        # Apply transformations to the image if specified
        if self.transform:
            image = self.transform(image)

        # Return the processed image, label as a float tensor, and image name
        return image, img_name

class load_csv(Dataset):
    def __init__(self, mean_data, target_variable, multiple_folder):
        self.data_frame = mean_data                    # Store the DataFrame containing embeddings and metadata
        self.target_variable = target_variable          # Store the target label column name

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Select all columns except non-feature/meta columns to extract the embedding
        image_embed = self.data_frame[
            self.data_frame.columns.difference([
                "image", "valence", "Unnamed: 0", "room_type", 
                "approach", "tense", "energetic"
            ])
        ].loc[idx]

        # Extract the label for the given index
        label = self.data_frame[self.target_variable].loc[idx] #iloc

        # Convert the embedding and label to float tensors and return
        return torch.tensor(image_embed, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def load_dataset(dataloader, multiple_folder, mean_data, target_variable, train_indices, test_indices, batch_size):
    # Initialize the dataset using the custom dataloader class and provided parameters
    dataset = dataloader(mean_data, target_variable, multiple_folder)
    
    # Create training and testing subsets using the extracted indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Wrap the subsets in DataLoaders with specified batch size and appropriate options
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Return the training and testing DataLoaders along with the full dataset object
    return train_dataloader, test_dataloader, dataset
