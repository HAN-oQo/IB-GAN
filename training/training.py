from dataloader.dataloader import load_dataloader
import os
import json
import numpy as np
import time
import datetime
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from training.logger import Logger

class Trainer():
    
    def __init__(self, device, train, directory, dataset, path_to_data, batch_size, size, z_dim , r_dim, ngf, ndf, nc, max_iters, noise_iters, resume_iters, restored_model_path, lr_E,lr_G, lr_Q,lr_D,weight_decay, beta1, beta2, milestones, scheduler_gamma, gan_loss_type, opt_type, start, end, steps, gan_weight, upper_weight,  print_freq, sample_freq, model_save_freq, test_iters, test_path, test_seed):

        self.device = device
        self.train_bool = train
        ##############
        # Directory Setting
        ###############
        self.directory = directory
        log_dir = os.path.join(directory, "logs")
        sample_dir = os.path.join(directory, "samples")
        result_dir = os.path.join(directory, "results")
        model_save_dir = os.path.join(directory, "models")

        if not os.path.exists(os.path.join(directory, "logs")):
            os.makedirs(log_dir)
        self.log_dir = log_dir

        if not os.path.exists(os.path.join(directory, "samples")):
            os.makedirs(sample_dir)
        self.sample_dir = sample_dir

        if not os.path.exists(os.path.join(directory, "results")):
            os.makedirs(result_dir)
        self.result_dir = result_dir

        if not os.path.exists(os.path.join(directory, "models")):
            os.makedirs(model_save_dir)
        self.model_save_dir = model_save_dir

        ##################
        # Data Loader
        ##################
        self.dataset = dataset
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.size = size
        self.data_loader , data_length = load_dataloader(dataset = self.dataset, 
                                                        path_to_data = self.path_to_data,
                                                        train = self.train_bool,
                                                        nc = nc,
                                                        size = self.size,
                                                        batch_size= self.batch_size
                                                        )

        self.z_dim = z_dim
        self.r_dim = r_dim
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc


        #################
        # Iteration Setting
        ################
        self.max_iters = max_iters
        self.resume_iters = resume_iters
        self.noise_iters = noise_iters
        self.global_iters = self.resume_iters
        self.restored_model_path = restored_model_path
        

        ##################
        # Optimizer, Scheduler setting
        ###############
        self.lr_E = lr_E
        self.lr_G = lr_G
        self.lr_Q = lr_Q
        self.lr_D = lr_D
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.milestones = milestones
        self.scheduler_gamma = scheduler_gamma
        
        #################
        # Loss hyperparameters 
        ################
        self.gan_loss_type =gan_loss_type
        self.opt_type =opt_type
        self.gan_weight = gan_weight
        self.upper_weight = upper_weight

        self.start = start
        self.end = end
        self.steps = steps
        
        #################
        # Log Setting
        #################
        self.print_freq = print_freq
        self.sample_freq = sample_freq
        self.model_save_freq = model_save_freq

        #################
        # Constant Tensor
        ##################
        # self.normal_mu = torch.tensor(np.float32(0)).to(self.device)
        # self.normal_log_var = torch.tensor(np.float32(0)).to(self.device)
        

        ################
        # Test Setting
        ################
        self.test_iters = test_iters
        self.test_path = test_path
        self.test_seed =  test_seed
    
        self.build_model()
        self.build_tensorboard()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    
    def build_model(self):
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        
        if self.opt_type == "RMS":
            self.opt_G = torch.optim.RMSprop(self.G.)
            self.opt_Q = 
            self.opt_D = 
        else:
            self.opt_G = 
            self.opt_Q = 
            Self.opt_D = 
        
        self.optimizer_exc_enc= torch.optim.Adam(itertools.chain(self.zx_encoder.parameters(), self.zy_encoder.parameters()), lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.weight_decay)
        self.optimizer_shr_enc= torch.optim.Adam(itertools.chain(self.FE.parameters(), self.zs_encoder.parameters(),self.zx_s_encoder.parameters(), self.zy_s_encoder.parameters()), 
                                                lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.weight_decay)
        self.optimizer_dec = torch.optim.Adam(itertools.chain(self.x_decoder.parameters(), self.y_decoder.parameters()), lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.weight_decay)


        # self.print_network(self.zx_encoder, 'ZX_Encoder')
        # self.print_network(self.zy_encoder, 'ZY_Encoder')
        # self.print_network(self.FE, 'Feature Extractor')
        # self.print_network(self.zx_s_encoder, 'ZX_S_Encoder')
        # self.print_network(self.zy_s_encoder, 'ZY_S_Encoder')
        # self.print_network(self.zs_encoder, 'ZS_Encoder')
        # self.print_network(self.decoder, 'decoders')
        

    def load_model(self, path, resume_iters):
        """Restore the trained generator and discriminator."""
        resume_iters = int(resume_iters)
        print('Loading the trained models from iters {}...'.format(resume_iters))
        path = os.path.join( path , '{}-checkpoint.pt'.format(resume_iters))
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_iters = checkpoint['iters']
        
        # self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)
        
    
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def reconstruction_loss(self, recon, input, name):
        if name == "L1":
            rec_loss = nn.L1Loss()
        elif name == "MSE":
            rec_loss = nn.MSELoss()
        else:
            rec_loss = nn.L1Loss()

        return rec_loss(recon, input)
    
    def KLD_loss_v1(self, mu, log_var):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss
    
    def KLD_loss(self, mu0, log_var0, mu1, log_var1):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + (log_var0 - log_var1) - ((mu0 - mu1) ** 2 + log_var0.exp()/log_var1.exp()), dim = 1), dim = 0)
        return self.kld_weight * kld_loss


    
    def train(self):
        one_label =torch.ones(self.batch_size, 1).to(self.device)
        zero_label = torch.zeros(self.batch_size, 0).to(self.device)

    
    def test(self):