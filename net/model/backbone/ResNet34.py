import torch
import torch.nn as nn
import torchvision.models.video as video_models

class R3D_18(nn.Module):
    def __init__(self, num_landmarks, coordinate_regression=False):
        super(R3D_18, self).__init__()
        # Load pre-trained 3D ResNet-34
        resnet34_3d = video_models.r3d_18(pretrained=False)  # Using R3D-18 as a 3D backbone
        self.num_landmarks = num_landmarks
        self.backbone = nn.Sequential(*list(resnet34_3d.children())[:-2])  # Remove the fully connected layers
        
        # Fully Convolutional Network (FCN) head
        self.fcn_head = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.fcn_output = nn.Conv3d(256, num_landmarks, kernel_size=1)
        
        self.coordinate_regression = coordinate_regression
        if coordinate_regression:
            # Dense layer for landmark regression
            self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(512, 3 * num_landmarks)

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        fcn_features = self.fcn_head(features)
        fcn_output = self.fcn_output(fcn_features)
        
        if self.coordinate_regression:
            # Dense layer path
            pooled_features = self.global_avg_pool(features)
            pooled_features = torch.flatten(pooled_features, 1)
            dense_output = self.fc(pooled_features).view(-1, self.num_landmarks, 3)
            return fcn_output, dense_output
        else: 
            return fcn_output            
