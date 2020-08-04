import torch

from wavenet_classifier import wave_net

sample_length = 20
sample_channels = 3
num_classes = 4

kernel_size = 2
num_kernels = 3
density = 1

num_hidden_channels = 36
num_out_channels = 24
dropout = 15



net = wave_net(num_time_samples=sample_length, num_in_channels=sample_channels, num_classes=num_classes,
               num_hidden=num_hidden_channels, num_out_features=num_hidden_channels, drop=dropout,
               kernel_size=kernel_size, num_kernels=num_kernels, density=density             
               )
print('\n---\ninit- ok\n---\n')

temp = torch.randn(sample_channels, sample_length).view(-1, sample_channels, sample_length)
temp_res = net(temp)
print(temp_res)
