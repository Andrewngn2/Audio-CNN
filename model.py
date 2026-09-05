import torch
import torch.nn as nn




class ResidualBlock(nn.Module):  #our constructor for the residual block and inherits the existing nn.module class from pytorch
    def __init__(self, in_channels, out_channels, stride=1):                                    
        super().__init__() #calls the intialization method of its nn.module which is the parent class
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False) #first convulational layer of residual block which will be initalized with the instance of residual block.
        self.bn1 = nn.BatchNorm2d(out_channels)                                             #batch norm already has a shifting mechanism so bias for this layer will not needed to be learned
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
         # as we transition from one layer of residuals to another the shortcut's out channels might not be enough for the in channels of the next block.
        self.shortcut = nn.Sequential() #by default  sequence does nothing
        self.use_shortcut = stride != 1 or in_channels != out_channels #this checks if we should use the shortcut. If stride isnt 1 it means the data will be down sampled. the shorcut data needs to be transformed to match. Second case is for mentioned above
        if self.use_shortcut:  #transform the input data so its able to added to the output layers above
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride,bias=False),nn.BatchNorm2d(out_channels)) 
            # a convoluational takes in original input and creates additional feature maps and matches the stride so size of feature maps and number of feature maps matches
    def forward(self, x, fmap_dict = None, prefix= ""): #defines logic for the forward pass
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x #determines if we need to use the transformation

        out_add = out + shortcut
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add
        out = torch.relu(out_add)
        if fmap_dict is not None:
                fmap_dict[f"{prefix}.relu"] = out
        return out
    

class AudioCNN(nn.Module):
    def __init__(self, num_classes=50): 
        super().__init__()
        #inital layer (1 input channel, feature maps produced, kernel size, stride, padding, bias)  inplace=true means that is is applied to the output of the previous layer
        self.conv1 = nn.Sequential(nn.Conv2d(1,64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range (3)]) # holds a list of modules allowing which is our residual blocks and shows that it is trainable
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i == 0 else 128,128,stride=2 if i ==0 else 1) for i in range (4)]) #has to take in right amount of feature maps. stride change to downsample the size of featuremaps
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i == 0 else 256,256, stride=2 if i ==0 else 1) for i in range (6)])
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512,512,stride=2 if i ==0 else 1) for i in range (3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        if not return_feature_maps:
            x =self.conv1(x)
            for block in self.layer1:      #loops through residual blocks in modulelist
                x = block(x) 
            for block in self.layer2:     
                x = block(x) 
            for block in self.layer3:     
                x = block(x) 
            for block in self.layer4:     
                x = block(x) 
            x = self.avgpool(x)
            x = x.view(x.size(0),-1) #.view reshapes a tensor without changing data. We keep the batch size of the tensor  which is the first element of x. -1 infers the size of the second dimension such that the total number of elements stay the same
            x = self.dropout(x)
            x = self.fc(x)
            return x
        else:
            feature_maps = {}
            x =self.conv1(x)
            feature_maps["conv1"]=x


            for i,block in enumerate(self.layer1):     
                x = block(x, feature_maps, prefix = f"layer1.block{i}") 
            feature_maps["layer1"]=x

            for i, block in enumerate(self.layer2):   
                x = block(x, feature_maps, prefix = f"layer2.block{i}") 
            feature_maps["layer2"]=x

            for i, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix = f"layer3.block{i}") 
            feature_maps["layer3"]=x
            
            for i, block in enumerate(self.layer4):     
                x = block(x, feature_maps, prefix = f"layer4.block{i}")
            feature_maps["layer4"]=x 


            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            x = self.dropout(x)
            x = self.fc(x)
            return x, feature_maps
