import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNNConfig(object):
    num_classes = 7067  # aspect
    num_filters = 12
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 30000
    print_per_batch = 20
    save_per_batch = 1000
    W = 100

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        self.input_x = torch.placeholder(torch.float32, [None, 5, self.config.num_classes, 1], name='input_x')
        self.input_y = torch.placeholder(torch.float32, [None, self.config.num_classes], name='input_y')

        self.cnn()

    def cnn(self):
        num_filters = self.config.num_filters

        conv1_1 = nn.Conv2d(1, num_filters, (5, 1), name='conv1_1')
        conv1_2 = nn.Conv2d(1, num_filters, (4, 1), name='conv1_2')
        conv1_3 = nn.Conv2d(1, num_filters, (3, 1), name='conv1_3')
        conv1_4 = nn.Conv2d(1, num_filters, (2, 1), name='conv1_4')

        maxpool1_1 = F.max_pool2d(conv1_1(self.input_x), (1, 1), stride=(1, 1), padding=0, name='maxpool1_1')
        maxpool1_2 = F.max_pool2d(conv1_2(self.input_x), (2, 1), stride=(1, 1), padding=0, name='maxpool1_2')
        maxpool1_3 = F.max_pool2d(conv1_3(self.input_x), (3, 1), stride=(1, 1), padding=0, name='maxpool1_3')
        maxpool1_4 = F.max_pool2d(conv1_4(self.input_x), (4, 1), stride=(1, 1), padding=0, name='maxpool1_4')

        concat = torch.cat([maxpool1_1, maxpool1_2, maxpool1_3, maxpool1_4], 1)
        fc1 = nn.Linear(concat.size(1), 1, name='fc1')
        conv2 = nn.Conv2d(1, num_filters, (2, 1), name='conv2')
        maxpool2 = F.max_pool2d(conv2(fc1(concat)), (3, 1), stride=(1, 1), padding=0, name='maxpool1_4')

        fc = nn.Linear(maxpool2.size(1), 1, name='fc2')
        self.y_pred = fc.view(-1, self.config.num_classes)

        cross_entropy = F.binary_cross_entropy_with_logits(self.y_pred, self.input_y, pos_weight=torch.tensor(self.config.W))
        self.loss = torch.mean(cross_entropy)
        self.optim = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)