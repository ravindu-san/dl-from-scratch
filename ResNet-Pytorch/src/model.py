import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_7x7 = nn.Conv2d(3,64,7,2)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res_block_1 = ResBlock(64,64,1)
        self.res_block_2 = ResBlock(64,128,2)
        self.res_block_3 = ResBlock(128,256,2)
        self.res_block_4 = ResBlock(256,512,2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten() # default start_dim=1, end_dim=-1
        self.fc = nn.Linear(512,2)
        self.sigmoid = nn.Sigmoid()

        self.sequence = nn.Sequential(self.conv2d_7x7, self.batch_norm, self.relu, self.max_pool, self.res_block_1, self.res_block_2, self.res_block_3, self.res_block_4, self.global_avg_pool, self.flatten, self.fc, self.sigmoid)

    def forward(self, input_tensor):
        return self.sequence(input_tensor)   



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size=3

         
        self.conv2D_1=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.batch_norm_1=nn.BatchNorm2d(self.out_channels)
        self.relu_1=nn.ReLU()

        self.conv2D_2=nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=1)
        self.batch_norm_2=nn.BatchNorm2d(self.out_channels)

        self.sequence_1=nn.Sequential(self.conv2D_1,self.batch_norm_1,self.relu_1, self.conv2D_2, self.batch_norm_2)
        
        self.relu_2=nn.ReLU()

        self.conv2D_1x1_skipp = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=self.stride)
        self.batch_norm_skipp = nn.BatchNorm2d(self.out_channels)

        self.sequence_skipp = nn.Sequential(self.conv2D_1x1_skipp,self.batch_norm_skipp)


    def forward(self, input_tensor):
        res_out = self.sequence_1(input_tensor)
        input_tensor = self.sequence_skipp(input_tensor)
        output = input_tensor + res_out
        output = self.relu_2(output)
        return output

        
       
         
def test_ResBlock():
    input_tensor=torch.rand((1,64,112,112))
    model=ResBlock(64,128,2)
    print(model(input_tensor).shape)



def test_ResNet():
    input_tensor=torch.rand((10,3,112,112))
    model=ResNet()
    print(model(input_tensor).shape)



# test_ResBlock()
# test_ResNet()