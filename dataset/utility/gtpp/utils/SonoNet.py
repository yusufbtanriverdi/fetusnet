import torch
from torch import nn
from torch.nn import functional as F

class _GridAttentionBlockND_TORR(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(1,1,1), bn_layer=True, use_W=True, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu'):
        super(_GridAttentionBlockND_TORR, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_softmax',
                        'concatenation_sigmoid', 'concatenation_mean',
                        'concatenation_range_normalise', 'concatenation_mean_flow']

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = sub_sample_factor if isinstance(sub_sample_factor, tuple) else tuple([sub_sample_factor])*dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # initialise id functions
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.W = lambda x: x
        self.theta = lambda x: x
        self.psi = lambda x: x
        self.phi = lambda x: x
        self.nl1 = lambda x: x

        if use_W:
            if bn_layer:
                self.W = nn.Sequential(
                    conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                    bn(self.in_channels),
                )
            else:
                self.W = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)

        if use_theta:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)


        if use_phi:
            self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                               kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)


        if use_psi:
            self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


        if nonlinearity1:
            if nonlinearity1 == 'relu':
                self.nl1 = lambda x: F.relu(x, inplace=True)

        if 'concatenation' in mode:
            self.operation_function = self._concatenation
        else:
            raise NotImplementedError('Unknown operation function.')

        if use_psi and self.mode == 'concatenation_sigmoid':
            nn.init.constant(self.psi.bias.data, 3.0)

        if use_psi and self.mode == 'concatenation_softmax':
            nn.init.constant(self.psi.bias.data, 10.0)

        # if use_psi and self.mode == 'concatenation_mean':
        #     nn.init.constant(self.psi.bias.data, 3.0)

        # if use_psi and self.mode == 'concatenation_range_normalise':
        #     nn.init.constant(self.psi.bias.data, 3.0)

        parallel = False
        if parallel:
            if use_W: self.W = nn.DataParallel(self.W)
            if use_phi: self.phi = nn.DataParallel(self.phi)
            if use_psi: self.psi = nn.DataParallel(self.psi)
            if use_theta: self.theta = nn.DataParallel(self.theta)

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        #############################
        # compute compatibility score

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w)
        # phi   => (b, c, t, h, w) -> (b, i_c, t, h, w)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        #  nl(theta.x + phi.g + bias) -> f = (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)

        f = theta_x + phi_g
        f = self.nl1(f)

        psi_f = self.psi(f)

        ############################################
        # normalisation -- scale compatibility score
        #  psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
        if self.mode == 'concatenation_softmax':
            sigm_psi_f = F.softmax(psi_f.view(batch_size, 1, -1), dim=2)
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == 'concatenation_mean':
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            psi_f_sum = torch.sum(psi_f_flat, dim=2)#clamp(1e-6)
            psi_f_sum = psi_f_sum[:,:,None].expand_as(psi_f_flat)

            sigm_psi_f = psi_f_flat / psi_f_sum
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == 'concatenation_mean_flow':
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            ss = psi_f_flat.shape
            psi_f_min = psi_f_flat.min(dim=2)[0].view(ss[0],ss[1],1)
            psi_f_flat = psi_f_flat - psi_f_min
            psi_f_sum = torch.sum(psi_f_flat, dim=2).view(ss[0],ss[1],1).expand_as(psi_f_flat)

            sigm_psi_f = psi_f_flat / psi_f_sum
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == 'concatenation_range_normalise':
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            ss = psi_f_flat.shape
            psi_f_max = torch.max(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)
            psi_f_min = torch.min(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)

            sigm_psi_f = (psi_f_flat - psi_f_min) / (psi_f_max - psi_f_min).expand_as(psi_f_flat)
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])

        elif self.mode == 'concatenation_sigmoid':
            sigm_psi_f = F.sigmoid(psi_f)
        else:
            raise NotImplementedError

        # sigm_psi_f is attention map! upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D_TORR(_GridAttentionBlockND_TORR):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(1,1), bn_layer=True,
                 use_W=True, use_phi=True, use_theta=True, use_psi=True,
                 nonlinearity1='relu'):
        super(GridAttentionBlock2D_TORR, self).__init__(in_channels,
                                               inter_channels=inter_channels,
                                               gating_channels=gating_channels,
                                               dimension=2, mode=mode,
                                               sub_sample_factor=sub_sample_factor,
                                               bn_layer=bn_layer,
                                               use_W=use_W,
                                               use_phi=use_phi,
                                               use_theta=use_theta,
                                               use_psi=use_psi,
                                               nonlinearity1=nonlinearity1)

class SonoNet(nn.Module):

    """Network for translation and rotation

    Args:
      x: an input tensor with the dimensions (B, C, H, W).
      config: network details
      input_plane: number of input planes (1 or 3)
      num_output_t: Dimensions of translation output (3 neurons)
      num_output_r: Dimensions of rotation output (3 or 4 neurons)
      dropout_p: is a scalar, for the probability of dropout.

    Returns:
      tran: translation classification output [B, num_output_t].
      rot: translation regression output [B, num_output_r].
      
      [tran rot]"""
      
    net_dict = {
        'SN16': [16, 32, 64, 128, 128],
        'SN32': [32, 64, 128, 256, 256],
        'SN64': [64, 128, 256, 512, 512]
    }
      
    def __init__(self,config,input_dim, num_output_t, num_output_r, dropout_p = 0.25, nonlocal_mode='concatenation_mean_flow'):
        
        self.limit_tran = config.limit_translation
        self.filters_n = self.net_dict[config.net]
        self.output_dim_t = num_output_t 
        self.output_dim_r = num_output_r
        self.output_dim = num_output_t + num_output_r
        self.latent_feat = config.latent
        self.latent_feat2 = config.latent2
        self.last_bn = config.last_bn
        super().__init__()
        
        ####################
        # Feature Extraction
    
        
        ## FEATURES EXTRACTION FROM ULTRASOUND IMAGES
        
        # First convolution block 
        self.blook1 = nn.Sequential(nn.Conv2d(1, self.filters_n[0], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[0]), nn.ReLU(), 
                                    nn.Conv2d( self.filters_n[0], self.filters_n[0], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[0]), nn.ReLU())
        self.max_pool1 = nn.MaxPool2d( kernel_size =2)
        
        # Second convolution block
        self.blook2 = nn.Sequential(nn.Conv2d(self.filters_n[0], self.filters_n[1], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[1]), nn.ReLU(), 
                                    nn.Conv2d( self.filters_n[1], self.filters_n[1], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[1]), nn.ReLU())
        self.max_pool2 = nn.MaxPool2d( kernel_size =2)
        
        # Third convolution block
        self.blook3 = nn.Sequential(nn.Conv2d(self.filters_n[1], self.filters_n[2], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[2]), nn.ReLU(),
                                    nn.Conv2d( self.filters_n[2], self.filters_n[2], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[2]), nn.ReLU(), 
                                    nn.Conv2d( self.filters_n[2], self.filters_n[2], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[2]), nn.ReLU())
        self.max_pool3 = nn.MaxPool2d(kernel_size =2)
        
        # Fourth convolution block
        self.blook4 = nn.Sequential(nn.Conv2d(self.filters_n[2], self.filters_n[3], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[3]), nn.ReLU(),
                            nn.Conv2d( self.filters_n[3], self.filters_n[3], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[3]), nn.ReLU(), 
                            nn.Conv2d( self.filters_n[3], self.filters_n[3], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[3]), nn.ReLU())
        self.max_pool4 = nn.MaxPool2d(kernel_size =2)
        
        # Fifth convolution block
        self.blook5 = nn.Sequential(nn.Conv2d(self.filters_n[3], self.filters_n[4], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[4]), nn.ReLU(),
                            nn.Conv2d( self.filters_n[4], self.filters_n[4], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[4]), nn.ReLU(), 
                            nn.Conv2d( self.filters_n[4], self.filters_n[4], (3,3),stride = (1,1),padding = 1), nn.BatchNorm2d(self.filters_n[4]), nn.ReLU())
        self.max_pool5 = nn.MaxPool2d( kernel_size =2)    
        
        
        ################
        # Attention Maps
        self.attention_filter_sizes = [self.filters_n[2], self.filters_n[3]]
        self.compatibility_score1 = GridAttentionBlock2D_TORR(in_channels=self.filters_n[2], gating_channels=self.filters_n[4],
                                                     inter_channels=self.filters_n[4], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        self.compatibility_score2 = GridAttentionBlock2D_TORR(in_channels=self.filters_n[3], gating_channels=self.filters_n[4],
                                                     inter_channels=self.filters_n[4], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')        
        #########################
        # Aggreagation Strategies
                     
        self.aggregation_features1 = nn.Linear(self.filters_n[2]+self.filters_n[3]+self.filters_n[3], self.latent_feat)
        self.aggregation_features2 = nn.Linear(self.filters_n[2]+self.filters_n[3]+self.filters_n[3], self.latent_feat)
        self.aggregation_features3 = nn.Linear(self.filters_n[2]+self.filters_n[3]+self.filters_n[3], self.latent_feat)
        self.aggr_bn1 = nn.BatchNorm1d(self.latent_feat)
        self.aggr_bn2 = nn.BatchNorm1d(self.latent_feat)
        self.aggr_bn3 = nn.BatchNorm1d(self.latent_feat)
        
        self.reduce_dim = nn.Linear(self.latent_feat*3, self.latent_feat)
        self.red_bn = nn.BatchNorm1d(self.latent_feat)
        '''self.reduce_dim2 = nn.Linear(self.latent_feat, self.latent_feat2)
        self.red_bn2 = nn.BatchNorm1d(self.latent_feat2)
        self.predict_transform =  nn.Linear(self.latent_feat2, self.output_dim)'''
        self.predict_transform =  nn.Linear(self.latent_feat, self.output_dim)
        self.pred_bn = nn.BatchNorm1d(self.output_dim)
        #Last layer actibation fuction output in range [-1, 1]
        self.act2 = F.tanh
        
        self.act1 = F.relu
        
        #########################
        # Dropout     
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.dropout4 = nn.Dropout(p=dropout_p)
        self.dropout5 = nn.Dropout(p=dropout_p)
        
    def forward(self, images):
    
        batch_size = images.shape[0]
        
        #################### 
        # Feature Extraction Image 1
        conv1 = self.blook1(images[:,0,:,:].unsqueeze(1))
        maxpool1 = self.max_pool1(conv1)

        conv2 = self.blook2(maxpool1)
        maxpool2 = self.max_pool2(conv2)

        conv3 = self.blook3(maxpool2)
        maxpool3 = self.max_pool3(conv3)

        conv4 = self.blook4(maxpool3)
        maxpool4 = self.max_pool4(conv4)

        conv5 = self.blook5(maxpool4)
        
        pooled = F.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)
        
        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(conv3, conv5)
        g_conv2, att2 = self.compatibility_score2(conv4, conv5)

        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)
        
        # Aggreate
        feature_1 = self.act1(self.aggr_bn1(self.aggregation_features1(torch.cat((g1,g2,pooled), dim=1))))
        feature_1 = self.dropout1(feature_1)
                 
        #################### 
        # Feature Extraction Image 2
        conv1 = self.blook1(images[:,1,:,:].unsqueeze(1))
        maxpool1 = self.max_pool1(conv1)

        conv2 = self.blook2(maxpool1)
        maxpool2 = self.max_pool2(conv2)

        conv3 = self.blook3(maxpool2)
        maxpool3 = self.max_pool3(conv3)

        conv4 = self.blook4(maxpool3)
        maxpool4 = self.max_pool4(conv4)

        conv5 = self.blook5(maxpool4)
        
        pooled = F.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)
        
        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(conv3, conv5)
        g_conv2, att2 = self.compatibility_score2(conv4, conv5)

        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)
        
        # Aggreate
        feature_2 = self.act1(self.aggr_bn2(self.aggregation_features1(torch.cat((g1,g2,pooled), dim=1))))       
        feature_2 = self.dropout2(feature_2) 
                
        #################### 
        # Feature Extraction Image 3
        conv1 = self.blook1(images[:,2,:,:].unsqueeze(1))
        maxpool1 = self.max_pool1(conv1)

        conv2 = self.blook2(maxpool1)
        maxpool2 = self.max_pool2(conv2)

        conv3 = self.blook3(maxpool2)
        maxpool3 = self.max_pool3(conv3)

        conv4 = self.blook4(maxpool3)
        maxpool4 = self.max_pool4(conv4)

        conv5 = self.blook5(maxpool4)
        
        pooled = F.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)
        
        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(conv3, conv5)
        g_conv2, att2 = self.compatibility_score2(conv4, conv5)

        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)
        
        # Aggreate
        feature_3 = self.act1(self.aggr_bn3(self.aggregation_features1(torch.cat((g1,g2,pooled), dim=1))))     
        feature_3 = self.dropout3(feature_3) 
        
        #############
        # Estimate transform
        out = self.act1(self.red_bn(self.reduce_dim(torch.cat((feature_1,feature_2,feature_3), dim=1))))
        out = self.dropout4(out)  
        #out = self.act1(self.red_bn2(self.reduce_dim2(out)))
        #out = self.dropout5(out) 
        if self.last_bn:
            out = self.act2(self.pred_bn(self.predict_transform(out)))
        else:
            out = self.act2(self.predict_transform(out))
        
        tran = out[:,0:self.output_dim_t]*self.limit_tran
        
        rot = out[:,self.output_dim_t:self.output_dim+1]
        rot = F.normalize(rot, dim =1)
             
        return tran, rot, g1, g2
