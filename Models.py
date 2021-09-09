import re
from torch import nn, optim
from torchvision import models

#####################################################################################################

class ConvNet(nn.Module):
    def __init__(self, model_name=None, pretrained=False):
        super(ConvNet, self).__init__()

        if re.match(r"resnet", model_name, re.IGNORECASE):
            self.model = models.resnet50(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
            in_features = self.model.fc.in_features 
            self.model.fc = nn.Linear(in_features=in_features, out_features=4)
            self.model.add_module("Final Activation", nn.LogSoftmax(dim=1))

        elif re.match(r"vgg", model_name, re.IGNORECASE):
            self.model = models.vgg16_bn(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features=in_features, out_features=4)
            self.model.classifier.add_module("Final Activation", nn.LogSoftmax(dim=1))

        elif re.match(r"mobilenet", model_name, re.IGNORECASE):
            self.model = models.mobilenet_v3_small(pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features=in_features, out_features=4)
            self.model.classifier.add_module("Final Activation", nn.LogSoftmax(dim=1))

        else:
            raise ValueError("Incorrect value passed to model_name. Supported are \n\n1. resnet\n2. vgg\n3. mobilenet\n\n")

        
        
    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False
    
    def get_optimizer(self, lr=1e-3, wd=0):
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=lr, weight_decay=wd)
    
    def get_plateau_scheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)
    
    def forward(self, x):
        return self.model(x)

#####################################################################################################
