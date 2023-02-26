from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import torch
import preprocess as prep
from preprocess import PhoneDataset
import pandas as pd


def train(train_df, val_df,
          model, optimizer, criterion,
          sampler=None,
          batch_size=32,
          epochs=10,
          device = torch.device('cpu')):

    train_dataset = PhoneDataset(train_df)
    val_dataset = PhoneDataset(val_df)
    # train loop:
    if sampler is None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            # fit loop
            model.train()
            tr_loss = 0.0
            tr_accuracy = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                tr_loss += loss

                preds = torch.argmax(outputs, 1)
                tr_accuracy += torch.sum(preds == y_batch) / batch_size

            # evaluation loop
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                with torch.set_grad_enabled(False):
                    outputs = model(x_batch)
                    val_loss += criterion(outputs, y_batch)

                    preds = torch.argmax(outputs, 1)
                    val_accuracy += torch.sum(preds == y_batch) / batch_size

            history.append((tr_loss.detach().to('cpu').numpy(),
                            tr_accuracy.detach().to('cpu').numpy(),
                            val_loss.detach().to('cpu').numpy(),
                            val_accuracy.detach().to('cpu').numpy()))
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=tr_loss,
                                           v_loss=val_loss, t_acc=tr_accuracy, v_acc=val_accuracy))

    return history


def plot_train_stats(hist):
    """
    Plots the statistic of train loops.
    Includes loss and accuracy values per train dataset and validation dataset.
    :param hist: list of tuples. Each tuple cosists of four values: train loss & accuracy, validation loss & accuracy.
    :return: None
    """
    loss, acc, val_loss, val_acc = zip(*hist)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].plot(loss, label="train_loss")
    ax[0].plot(val_loss, label="val_loss")
    ax[0].legend(loc='best')
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].plot(acc, label="train_acc")
    val_accuracy = [i.item() for i in val_acc]
    ax[1].plot(val_accuracy, label="val_acc")
    ax[1].legend(loc='best')
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("accuracy")
    plt.show()


if __name__ == "__main__":
    # Paths and prerequisites.
    meta_df = pd.read_csv("../data/4k_mobile_images/mobile_data_img.csv")
    IMG_URL = "../data/4k_mobile_images/"
    JUNK_IMG_URL = "../data/junk_photo"
    RESCALE_SIZE = 224
    meta_df.Image_File = IMG_URL + meta_df.Image_File
    if torch.cuda.is_available():
        print("Train on GPU...")
        DEVICE = torch.device('cuda')
    else:
        print("Train on CPU, CUDA not available")
        DEVICE = torch.device('cpu')

    # Preprocess data
    meta_df = prep.correct_names_in_df(meta_df)
    binary_df = prep.prepare_final_df(meta_df, JUNK_IMG_URL)
    #binary_df = binary_df.reset_index()
    train_dataframe, val_dataframe = train_test_split(binary_df, train_size=0.8, shuffle=False)
    val_dataframe = val_dataframe.reset_index()
    # Set pretrained model and its fc-layers.
    weighted_sampler = prep.get_weighted_sampler(train_dataframe)
    pretrained_model = models.resnet18(pretrained=True)
    fc_intro_features = pretrained_model.fc.in_features
    num_classes = 2
    pretrained_model.fc = nn.Sequential(nn.Linear(fc_intro_features, 256),
                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, num_classes))

    # Set parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

    # Training begins!
    pretrained_model.to(DEVICE)
    history = train(train_dataframe,
                    val_dataframe,
                    model=pretrained_model,
                    optimizer=optimizer,
                    criterion=criterion,
                    sampler=weighted_sampler,
                    device=DEVICE)
    plot_train_stats(history)

    checkpoint = {
        'model': pretrained_model,
        'state_dict': pretrained_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, './models/model_resnet18.pth')
