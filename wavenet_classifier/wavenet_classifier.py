from functools import reduce

import torch
import torch.nn as nn


class wave_net(nn.Module):
    def __init__(self,
                 num_time_samples,
                 num_in_channels=1,
                 num_hidden=128,
                 num_out_features=48,
                 num_classes=5,
                 kernel_size=2,
                 num_kernels=2,
                 density=1,
                 drop=20,
                 debug=False,
                 ):
        """
        
        :param num_time_samples: The expected length of the 1-d input n x c n L
        :param num_in_channels: Num# of input channels n x C x l
        :param num_hidden: Num# of forward channels through the dilated net
        :param num_out_features: Num# of channels used for output features, after the dilated net
        :param num_classes: Num# of sample/dataset classes, practically final output width
        :param kernel_size: Size of the 1st(see bellow) kernel used throught the dilated net
        :param num_kernels: How many blocks, with kernels all differing by +1, will be created
                            e.g kernel_size=2, num_kernels=3 will create 3 dilated blocks, with kernel sizes
                            2, 3 and 4, fed with the same input, and their outpus concatenated
        :param density: How many blocks of the -same- kernel will be created, practically a multiplier on the above number
        :param drop: Dropout used, after the dilated blocks, fed on the last two conv layers
        :param debug: Perform a mock pass and print on init


        """
        super().__init__()
        self.num_time_samples = num_time_samples
        self.num_in_channels = num_in_channels
        self.num_out_features = num_out_features
        self.num_classes = num_classes
        print(f"Configured for {self.num_classes} classes")
        self.num_kernels = num_kernels

        self.num_hidden = num_hidden
        self.kernel_size = kernel_size

        self.output_width = kernel_size + (density - 1)

        d_temp = drop / 100
        print(f"Dropout: {d_temp}")
        self.drop = nn.Dropout(p=d_temp)


        self.file_path = f"{__file__}"

        print(f"\nIn_channels = {self.num_in_channels}\nHidden = {self.num_hidden}\n")
        self.conv_init = nn.Conv1d(self.num_in_channels, self.num_hidden, 1)    # create channels the blocks are going to use

        wbs = []
        for i in range(num_kernels):
            for j in range(density):
                kernel = kernel_size + i

                wb = wave_block(num_time_samples=self.num_time_samples,
                                num_channels=num_hidden,
                                kernel_size=kernel,
                                output_width=self.output_width,
                                )

                wbs.append(wb)

        self.wbs = nn.ModuleList(wbs)

        self.activ_1 = torch.nn.ReLU()
        self.conv_1_1 = nn.Conv1d((num_hidden * num_kernels * density), (num_out_features * num_kernels), 1)
        self.activ_2 = torch.nn.ReLU()
        self.conv_1_2 = nn.Conv1d((num_out_features * num_kernels), self.num_classes, kernel_size=self.output_width)

        # -----------------------------------------------------

        if debug:
            self.debug()
            print("\n===\ninitialized\n===")

    def forward(self, x):
        x = self.conv_init(x)
        """
        Input goes through a conv layer with kernel size 1 to create the number of hidden channels
        """
        
        block_skips = []
        for block in self.wbs:
            block_out = block(x)
            block_skips.append(block_out)

        x = reduce((lambda a, b: torch.cat((a, b), 1)), block_skips)

        """
        Passing the input through each block in parallel and the outputs are concatenated
        """

        x = self.drop(self.conv_1_1(self.activ_1(x)))
        x = self.conv_1_2(self.activ_2(x))


        x = x.view(-1, self.num_classes)

        return x

    def debug(self):
        x_temp = torch.randn(self.num_in_channels, self.num_time_samples).view(-1, self.num_in_channels, self.num_time_samples)
        print(f"\nGiven sample size: {x_temp.shape}")
        x_temp = self.forward(x_temp)
        print(f"past wave blocks: {x_temp[0].shape}\n")
        del(x_temp)

# ===========================================================================================


class wave_block (nn.Module):
    def __init__(self, num_time_samples, num_channels, kernel_size, output_width):
        super(wave_block, self).__init__()

        self.num_channels = num_channels
        self.output_width = output_width
        self.kernel_size = kernel_size

        counter = 1
        while True:
            if (self.kernel_size**counter) + output_width > num_time_samples:
                break
            else:
                counter += 1
        print(f"Block kernel size: {self.kernel_size}: {counter-1} layers reading {self.kernel_size**(counter-1)} elements")

        self.num_layers = counter - 1

        """
        We create enough layers to 'consume'/concentrate the information on the first X values on the top layer
        where X is no shorter than the biggest kernel - but also as short as we can make it
        """

        hs = []
        batch_norms = []

        for i in range(self.num_layers):
            rate = self.kernel_size**i
            h = Gated_Residual_Layer(self.num_channels, self.kernel_size, self.output_width,
                                     dilation=rate)
            h.name = 'b{}-l{}'.format(self.kernel_size, i)

            hs.append(h)
            batch_norms.append(nn.BatchNorm1d(self.num_channels))

        self.hs = nn.ModuleList(hs)
        self.batch_norms = nn.ModuleList(batch_norms)

    def forward(self, x):
        x, skips = self.hs[0](x)
        x = self.batch_norms[0](x)
        for layer, batch_norm in zip(self.hs[1:], self.batch_norms[1:]):
            x, skip = layer(x)
            x = batch_norm(x)
            skips += skip
        return skips


# ===========================================================================================


class Gated_Residual_Layer(nn.Module):
    def __init__(self, channels, kernel_size, output_width, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(Gated_Residual_Layer, self).__init__()
        print(f"Gated layer - dil:{dilation}")
        self.dilation = dilation
        self.output_width = output_width
        self.gated = Gated_Conv_1d(channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias)

        self.conv_res = nn.Conv1d(in_channels=channels, out_channels=channels,
                                  kernel_size=1, stride=1, padding=0, dilation=1,
                                  groups=1, bias=bias)

        self.conv_skip = nn.Conv1d(in_channels=channels, out_channels=channels,
                                   kernel_size=1, stride=1, padding=0, dilation=1,
                                   groups=1, bias=bias)

    def forward(self, x):
        d_res = self.gated(x)

        x = x.narrow(-1, 0, d_res.shape[-1])

        residual = x + self.conv_res(d_res)
        skip = self.conv_skip(d_res.narrow(-1, 0, self.output_width))

        return residual, skip


class Gated_Conv_1d(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(Gated_Conv_1d, self).__init__()
        self.dilation = dilation
        self.channels = channels

        self.conv_dil = nn.Conv1d(in_channels=channels, out_channels=(2 * channels),
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups, bias=bias)

        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_dil(x)
        tn, sg = torch.split(x, self.channels, 1)
        return self.tan(tn) * self.sig(sg)
