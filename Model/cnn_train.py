import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import TCNNConfig, TextCNN
from input_data import load_data_label

train_list_side = None
train_list_tag = None
val_list_side = None
val_list_tag = None
test_list_side = None
test_list_tag = None

Top_k = 20

save_dir = 'model/'
save_path = os.path.join(save_dir, 'cnn_model')

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch):
    x_batch = torch.FloatTensor(x_batch)
    y_batch = torch.FloatTensor(y_batch)
    return x_batch, y_batch

def batch_iter(x, y, batch_size=32):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def evaluate(model, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_map = 0.0
    with torch.no_grad():
        for x_batch, y_batch in batch_eval:
            x_batch, y_batch = feed_data(x_batch, y_batch)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            y_pred = outputs.data.cpu().numpy()
            map_val = MAP(y_pred, y_batch)
            total_loss += loss.item() * len(x_batch)
            total_map += map_val * len(x_batch)

    return total_loss / data_len, total_map / data_len

def MAP(y_p, y_t):
    data_len = len(y_p)
    map = 0
    for yp, yt in zip(y_p, y_t):
        ids = np.argsort(-yp)[:Top_k]
        count = 1
        true_num = 1
        map_each = 0
        for id in ids:
            if yt[count] > 0:
                map_each += true_num / count
                true_num += 1
            count += 1
        map += map_each / Top_k
    return map / data_len

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = 'tensorboard/cnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0
    best_map_val = 0.0
    last_improved = 0
    require_improvement = 200
