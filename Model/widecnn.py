import torch
import torch.nn as nn
import torch.nn.functional as F

class IRFeatureCNN(nn.Module):
    def __init__(self, dim_dense, report_idx2cf, report_idx2ff, report_idx2fr,
                 report_idx2sim, report_idx2tr, report_idx2cc):
        super(IRFeatureCNN, self).__init__()
        
        self.report_idx2cf = torch.cuda.FloatTensor(report_idx2cf)
        self.report_idx2sim = torch.cuda.FloatTensor(report_idx2sim)
        self.report_idx2ff = torch.cuda.FloatTensor(report_idx2ff)
        self.report_idx2fr = torch.cuda.FloatTensor(report_idx2fr)
        self.report_idx2cc = torch.cuda.FloatTensor(report_idx2cc)
        
        
        self.conv1_1=nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(5,1),
            stride=1,
            padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,1)),
        )
        self.conv1_2=nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(4,1),
            stride=1,
            padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
        )
        self.conv1_3=nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(3,1),
            stride=1,
            padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1)),
        )
        self.conv1_4=nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(2,1),
            stride=1,
            padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1)),
        )
          
        # self.fc1 = nn.Linear(concat.size(3), 1)
        # [4146,4,4,2795]
        self.fc1 = nn.Linear(2795, 1)
        
        # self.conv2 = nn.Conv2d(4, 16, (2, 1))
        # maxpool2 = F.max_pool2d(self.conv2(feature1), (3, 1), stride=(1, 1), padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=(2,1),
            stride=1,
            padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1)),
        )
        
        # (1,m,a) (1,a,n)  --->(1,m,n)
        self.fc2 = nn.Linear(16*1*2795, dim_dense)
        
        
    def forward(self, r_idx, c_idx):
        # 特征矩阵
        feature_matrix = torch.cat([self.report_idx2cf.unsqueeze(0), self.report_idx2sim.unsqueeze(0), self.report_idx2ff.unsqueeze(0), self.report_idx2fr.unsqueeze(0), self.report_idx2cc.unsqueeze(0)], dim=0)
        feature_matrix = feature_matrix.permute(1,0,2)
        feature_matrix = feature_matrix.unsqueeze(1)
        
        print("\ninput size:"+str(feature_matrix.shape))
        
        # 第一个卷积层
        conv1 = self.conv1_1(feature_matrix)
        print("conv1:"+str(conv1.shape))
        conv2 = self.conv1_2(feature_matrix)
        print("conv2:"+str(conv2.shape))
        conv3 = self.conv1_3(feature_matrix)
        print("conv3:"+str(conv3.shape))
        conv4 = self.conv1_4(feature_matrix)
        print("conv4:"+str(conv4.shape))
        # 将结果拼接得到一个新的特征矩阵
        concat = torch.cat([conv1,conv2,conv3,conv4], 2)

        # 第二个卷积层
        # fc = self.fc1(concat)
        # print("fc:"+str(fc.shape))
        maxpool2 = self.conv2(concat)
        x = maxpool2.view(-1, 16*1*2795)
        # (bz, 16, 1, 2795) ---> (batch_size, 16*1*2795)
        # 全连接层
        feature = self.fc2(x)
        # (bz, dim_dense)
        
        print("output size:"+str(feature.shape))

        return feature

        
        