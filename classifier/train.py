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


def fit_epoch(model, loader, batch_size, device):
    """
        Train loop. Each epoch fit epoch updates the weights of the model. Return stats of the epoch.
        params:
            model
            loader
            batch_size
            device
        output:
            tr_loss: mean criterion over the train_set.
            tr_accuracy: share of correct answers per epoch
    """
    model.train()
    tr_loss = 0.0
    tr_accuracy = 0.0
    accuracy_norm = len(loader)
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)  # encode labels!!!!
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        tr_loss += loss.detach()
        preds = torch.argmax(outputs, 1)
        acc = torch.sum(preds == y_batch) / batch_size
        tr_accuracy += acc.detach()
    tr_loss = tr_loss / accuracy_norm
    tr_accuracy = tr_accuracy / accuracy_norm
    return tr_loss, tr_accuracy


def eval_epoch(model, loader, batch_size, device):
    """
        Eval loop. Each epoch eval_epoch only calculates stats of the epoch over validation set.
        params:
            model
            loader
            batch_size
            device
        output:
            val_loss: mean criterion over the validation_set.
            val_accuracy: share of correct answers per epoch
    """
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    accuracy_norm = len(loader)
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(x_batch)
            val_loss += criterion(outputs, y_batch) #TODO: criterion is defined outside of eval_epoch.
            preds = torch.argmax(outputs, 1)
            acc = torch.sum(preds == y_batch) / batch_size
            val_accuracy += acc.detach()
    val_loss = val_loss / accuracy_norm
    val_accuracy = val_accuracy / accuracy_norm
    return val_loss, val_accuracy


def train(train_df, val_df,
          model, optimizer, criterion,
          sampler=None,
          batch_size=32,
          epochs=10,
          device=torch.device('cpu')):
    """
        Train function builds a dataloader based on provided datasets and starts train process of the provided model.
        :param:
            train_df: dataframe for fit loops. In fit loops updating of model weights is enabled.
            val_df: dataframe for validation loops. In val loop model is not updated, i.e. it is not influenced by them.
            model: a neural network model to be trained.
            optimizer: a specific algorithm of gradient descent.
            criterion: loss function. params - predicted values and real labels. This function is minimized via optimizer.
            sampler: used in dataloader in case of imbalanced classes.
            batch_size: size of sample batches provided by dataloader. Inflicts on train speed and memory.
            epochs: number of training loops. Optimal number depends on the task, here 15-20 is likely to be enough.
            device: in case GPU is available.
        :return:
            history: list of tuples with values of loss and accuracy (% of correct answers) for train and eval loops.
        :TODO:
            -test.
    """
    train_dataset = PhoneDataset(train_df, mode='train')
    val_dataset = PhoneDataset(val_df)
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
            # Fit loop.
            tr_loss, tr_accuracy = fit_epoch(model, train_loader, batch_size, device)

            # Evaluation loop.
            val_loss, val_accuracy = eval_epoch(model, val_loader, batch_size, device)

            # Preparing epoch stats.
            history.append((tr_loss.to('cpu').numpy(),
                            tr_accuracy.to('cpu').numpy(),
                            val_loss.to('cpu').numpy(),
                            val_accuracy.to('cpu').numpy()))
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=tr_loss,
                                           v_loss=val_loss, t_acc=tr_accuracy, v_acc=val_accuracy))
    return history


def plot_train_stats(hist):
    """
    Plots the statistic of train loops.
    Includes loss and accuracy values per train dataset and validation dataset.
    :param hist: list of tuples. Each tuple consists of four values: train loss & accuracy, validation loss & accuracy.
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
    RESCALE_SIZE = 100
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

    torch.save(checkpoint, 'trained_models/model_resnet18.pth')
