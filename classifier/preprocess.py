import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch
from tqdm import tqdm
from PIL import Image

RESCALE_SIZE = 100


class PhoneDataset(Dataset):
    """
        Dataset class. __getItem__ returns pair image-label.
        Image - torch.Tensor of shape [3, RESCALE_SIZE, RESCALE_SIZE].
        Labels - {0, 1}. 1 Corresponds to image of a phone.
    """
    def __init__(self, df, mode = 'test'):
        super().__init__()
        self.files = df['Image_File']
        self.labels = df['Label']  # !! Brand
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        label = self.labels[index]
        x = self.load_sample(self.files[index])
        x = self.prepare_sample(x)
        x = np.array(x / 255, dtype='float32')

        # RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]
        if self.mode != "train":
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=45),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        try:
            x = transform(x)
        except RuntimeError:
            x = np.array([x, x, x])
            x = x.reshape(-1, -1, 3)
            x = transform(x)
        return x, label

    @staticmethod
    def load_sample(file):
        image = Image.open(file)
        image.load()
        return image

    @staticmethod
    def prepare_sample(image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


def correct_filename(filename: str):
    """
    Gets filename, leaves only ID of a photo.
    Applied to phone-images dataset only due to encoding errors in file names.
    :param filename: string of path to file from mobile_data_img.csv.
    :return: string with corrected filename
    """
    extra = re.search("_(.+?)\.", filename.split('/')[-1]).group(1)
    return filename.replace(extra, '')


def correct_names_in_df(data):
    """
    Corrects filenames in mobile_data_img.csv.
    :param data: pd.DataFrame from mobile_data_img.csv
    :return: corrected df.
    """
    print("Correct filenames...")
    data["Image_File"] = data["Image_File"].map(correct_filename)
    return data


def image_is_correct(file_path):
    """
        Some images are grayscale. They are few and thus are filtered
    """
    img = PhoneDataset.load_sample(file_path)
    img = PhoneDataset.prepare_sample(img)
    img = np.array(img / 255, dtype='float32')
    if img.shape != (RESCALE_SIZE, RESCALE_SIZE, 3):
        return False
    else:
        return True


def prepare_final_df(data, junk_df_path):
    """
    # Concat phone and "junk" photo into one dataset for binary classification.
    :param data:
    :param junk_df_path: "../data/junk_photo"
    :return: imbalanced dataset containing both phone- and junk- images.
    """
    print("Concatenate dataframes...")
    binary_task_df = pd.DataFrame(data["Image_File"])
    binary_task_df["Label"] = 1

    file_dict = {"Image_File": [], "Label": []}
    error_counter = 0
    for dirpath, dirs, files in os.walk(junk_df_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(dirpath, filename)
                if image_is_correct(file_path):
                    file_dict["Image_File"].append(file_path)
                    file_dict["Label"].append(0)
                else:
                    error_counter += 1
    junk_df = pd.DataFrame(file_dict)
    binary_task_df = pd.concat([binary_task_df, junk_df], axis=0, ignore_index=True)
    print("Number of incorrect images in junk files:", error_counter)
    return binary_task_df


def get_weighted_sampler(imbalanced_binary_df):
    """
    Dataset is highly imbalanced.
    Thus, we need weighted sampler to solve this problem on dataloader step.
    :param imbalanced_binary_df: pd.Dataframe, consists of rows: image_filename - label {0, 1}.
    :return: a WeightedRandomSampler which would be given as a param to a DataLoader.
    """
    print("Calculate dataset weights...")
    imb_binary_dataset = PhoneDataset(imbalanced_binary_df)
    v_counts = imbalanced_binary_df["Label"].value_counts()

    class_weights = {label: 1. / c for label, c in v_counts.items()}
    sample_weights = [0] * len(imb_binary_dataset)
    with tqdm(total=len(imb_binary_dataset)) as pbar_outer:
        for i, (data, label) in enumerate(imb_binary_dataset):
            class_weight = class_weights.get(label)
            sample_weights[i] = class_weight
            pbar_outer.update(1)
            if i == len(imb_binary_dataset) - 1:
                break
    return WeightedRandomSampler(sample_weights, num_samples=2 * len(imb_binary_dataset))


def count_values(image_df, sampler=None):
    """
    Count number of classes in df, with different samplers (or None)
    :param image_df: pf.DataFrame: Image_File -> Label
    :param sampler: some sampler or None
    :return: pd.DataFrame of shape [1, 2].
    """
    print("Counting class samples...")
    print(' ')
    if sampler is None:
        plot_loader = DataLoader(PhoneDataset(image_df), batch_size=100, shuffle=True)
    else:
        plot_loader = DataLoader(PhoneDataset(image_df), batch_size=100, sampler=weighted_sampler)
    counter_list = [0, 0]
    with tqdm(total=len(plot_loader)) as pbar_outer:
        for _, labels in plot_loader:
            tp = torch.sum(labels) # True Positive
            counter_list[1] += tp
            counter_list[0] += 100 - tp
            pbar_outer.update(1)
    return counter_list # pd.DataFrame(np.array(counter_list), columns=["False", "True"])


def plot_class_balance(df_1, df_2):
    """
    Plots bar charts with class balances
    """
    print("Plotting class balance before & after applying weights...")
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].barh([0, 1], df_1) #
    ax[1].barh([0, 1], df_2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    meta_df = pd.read_csv("../data/mobile_images/mobile_data_img.csv")
    IMG_URL = "../data/mobile_images/"
    JUNK_IMG_URL = "../data/junk_photo"

    meta_df.Image_File = IMG_URL + meta_df.Image_File

    meta_df = correct_names_in_df(meta_df)
    binary_df = prepare_final_df(meta_df, JUNK_IMG_URL)

    weighted_sampler = get_weighted_sampler(binary_df)
    imbalanced_counter = count_values(binary_df)
    balanced_counter = count_values(binary_df, weighted_sampler)
    plot_class_balance(imbalanced_counter, balanced_counter)

