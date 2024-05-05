import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as ospj


class CNN(nn.Module):
    """CNN"""
    # def __init__(self):
    #     # 定义顺序模型
    #     model = models.Sequential()
    #     model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    #     model.add(layers.MaxPooling2D((3, 3)))
    #     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #     model.add(layers.MaxPooling2D((4, 4)))
    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
    #     model.add(layers.Dense(192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
    #     model.add(layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
    #     model.add(layers.Softmax())
    #
    #     # 打印架构
    #     # model.summary()
    #     self.network = model

    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # (B, 3, 32, 32)
                out_channels=16,
                kernel_size=3,
                stride=1
            ),  # (B, 16, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)  # 默认步长跟池化窗口大小一致 -> (B, 16, 10, 10)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # (B, 16, 10, 10)
                out_channels=64,
                kernel_size=3,
                stride=1
            ),  # (B, 64, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)  # (B, 64, 2, 2)
        )

        self.flatten = nn.Flatten()  # torch.nn.Flatten(start_dim=1,end_dim=-1) 默认从第1维到-1维展平，batch为第0维
        # (B, 64*2*2)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 384),
            nn.ReLU()
        )  # (B, 384)

        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU()
        )
        # (B, 192)
        self.fc3 = nn.Linear(192, 10)
        # nn.Softmax(dim=1)
        # (B, 10)

    def forward(self, x):
        # print(x.shape)
        # x = [B, 3, 32, 32]
        x = self.conv1(x)
        # x = [B, 16, 10, 10]
        x = self.conv2(x)
        # x = [B, 64, 2, 2]
        x = self.flatten(x)
        # x = [B, 64*2*2]
        x = self.fc1(x)
        # x = [B, 384]
        x = self.fc2(x)
        # x = [B, 192]
        x = self.fc3(x)
        # x = [B, 10]
        return x
