from torch.nn import parameter
from torch.nn.modules.loss import MSELoss
from dataloader.dataloader import load_dataloader
from models import *
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
    
    def __init__(self, device, train, directory, dataset, path_to_data, batch_size, size, z_dim , r_dim, ngf, ndf, nc, weight_init,max_iters, noise_iters, resume_iters, restored_model_path,lr_E, lr_G,lr_D,lr_Q,weight_decay, beta1, beta2, milestones, scheduler_gamma, gan_loss_type, opt_type, label_smoothing, instance_noise, noise_start, noise_end, start, end, steps, gan_weight, upper_weight,  print_freq, sample_freq, model_save_freq, test_iters, test_path, test_seed):
        
        self.device = device
        print(self.device)
        self.train_bool = train
        ###########################
        # Directory Setting
        ###########################
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
        ##################################
        # Network Setting
        ###################################
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.weight_init = weight_init

        ###################################
        # Noise activation Setting
        #################################
        self.z_lower_limit = 0

        #######################
        # Iteration Setting
        ########################
        self.max_iters = max_iters
        self.resume_iters = resume_iters
        self.noise_iters = noise_iters
        self.global_iters = self.resume_iters
        self.restored_model_path = restored_model_path
        

        #######################################
        # Optimizer, Scheduler setting
        ####################################
        self.lr_E = lr_E
        self.lr_G = lr_G
        self.lr_Q = lr_Q
        self.lr_D = lr_D
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.milestones = milestones
        self.scheduler_gamma = scheduler_gamma
        
        ############################
        # Loss, optim type
        ##########################
        self.gan_loss_type =gan_loss_type
        self.opt_type =opt_type

        ##########################################
        # Discriminator Regularization Setting
        ########################################
        self.label_smoothing = label_smoothing
        self.instance_noise = instance_noise
        self.noise_start = noise_start
        self.noise_end = noise_end

        ###########################
        # Loss hyperparameters 
        ##########################
        self.gan_weight = gan_weight
        self.upper_weight = upper_weight
        self.kld_weight = self.batch_size/data_length

        ##################################
        #Latent traversal setting
        ###################################
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
    
        ###################
        # init
        #####################
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
        self.G = Generator(self.ngf, self.nc, self.z_dim, self.r_dim, self.weight_init)
        if self.dataset == "dsprites":
            self.D = Weight_Seperated_Discriminator(self.ndf, self.nc, self.z_dim, self.weight_init)
            print("weight seperated")
        elif self.dataset == "celeba" or self.dataset == "3dchairs":
            self.D = Weight_Shared_Discriminator(self.ndf, self.nc, self.z_dim, self.weight_init)
            print("weight shared")
        else:
            raise(RuntimeError("Dataset not exists..."))
        
        if self.opt_type == "RMS":
            self.opt_G = torch.optim.RMSprop([{"params": self.G.r.parameters(), "lr" : self.lr_E}, 
                                            {"params": self.G.r_to_g.parameters(), "lr" : self.lr_G},
                                            {"params": self.G.g.parameters(), "lr" : self.lr_G},
                                            {"params": self.D.q.parameters(), "lr": self.lr_Q}], lr = self.lr_G, momentum = 0.9)
            # self.opt_FE = torch.optim.RMSprop(itertools.chain)
            # self.opt_Q = torch.optim.RMSprop(itertools.chain(self.D.FE.parameters(), self.D.q.parameters()), lr = self.lr_Q, momentum = 0.9)
            self.opt_D = torch.optim.RMSprop([{"params": self.D.FE.parameters()}, {"params": self.D.d.parameters()}], lr=self.lr_D, momentum = 0.9)
            
        else:
            self.opt_G = torch.optim.Adam([{"params": self.G.r.parameters(), "lr" : self.lr_E}, 
                                            {"params": self.G.r_to_g.parameters(), "lr" : self.lr_G},
                                            {"params": self.G.g.parameters(), "lr" : self.lr_G},
                                            {"params": self.D.q.parameters(), "lr": self.lr_Q}], lr = self.lr_G, beta = (self.beta1, self.beta2))
            self.opt_D = torch.optim.Adam([{"params": self.D.FE.parameters()}, {"params": self.D.d.parameters()}], lr=self.lr_D, beta = (self.beta1, self.beta2))
        
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_D, milestones = self.milestones, gamma = self.scheduler_gamma)
        # self.print_network(self.G, "Generator" )
        # self.print_network(self.D, "Discriminator")

        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)


    def load_model(self, path, resume_iters):
        """Restore the trained generator and discriminator."""
        resume_iters = int(resume_iters)
        print('Loading the trained models from iters {}...'.format(resume_iters))
        path = os.path.join( path , '{}-checkpoint.pt'.format(resume_iters))
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_iters = checkpoint['iters']
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.opt_G.load_state_dict(checkpoint['opt_G'])
        self.opt_D.load_state_dict(checkpoint['opt_D'])
        
        self.G.to(self.device)
        self.D.to(self.device)
        # self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)
        
    
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)
    

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return torch.clamp(out,0, 1)
    
    def add_gaussian_noise(self, input, sigma):
        """ Instance Noise for Discriminator"""
        if sigma <= 0:
            return input
        else:
            noise = torch.randn_like(input).to(self.device) * sigma
            new_input = input + noise
            new_input = torch.clamp(new_input, -1, 1)
            return new_input
    
    def reconstruction_loss(self, input, recon, name):
        if name == "L1":
            rec_loss = nn.L1Loss()
            return rec_loss(recon, input)
        elif name == "MSE":
            rec_loss = nn.MSELoss()
            return rec_loss(recon, input)
        elif name == "IBGAN":

            # (input - z_lower_limit).pow(2) : to prevent vanishing z / z should be bigger than lower limit
            rec_loss = 0.5 * torch.mean(torch.sum(((input - recon).pow(2)- (input-self.z_lower_limit).pow(2)), dim = 1), dim=0)  
            return rec_loss
        
        else:
            raise(RuntimeError("Wrong reconstruction loss type"))
        
    
    def KLD_loss_v1(self, mu, log_var):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss 
    
    def KLD_loss(self, mu0, log_var0, mu1, log_var1):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + (log_var0 - log_var1) - ((mu0 - mu1) ** 2 + log_var0.exp()/log_var1.exp()), dim = 1), dim = 0)
        return self.kld_weight * kld_loss

    def GAN_loss(self, input, label):
        gan_loss = nn.BCEWithLogitsLoss()
        return gan_loss(input, label)
    
    def train(self):
        
        print(self.device)
        if self.label_smoothing:
            print("use label smoothing")
        else:
            print("not using label smoothing")

        one_label =torch.ones(self.batch_size, 1).to(self.device)
        fake_label = torch.zeros(self.batch_size, 1).to(self.device)
        z_fixed = torch.randn(16, self.z_dim).to(self.device)
        r_fixed = torch.randn(16, self.r_dim).to(self.device)

        data_iter = iter(self.data_loader)

        if self.resume_iters > 0:
            self.load_model(self.restored_model_path, self.resume_iters)
            self.global_iters = self.resume_iters
        

        print("Start Training...")
        start_time = time.time()
        while self.global_iters <= self.max_iters:

            try:
                real_X = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_X = next(data_iter)
            

            if self.label_smoothing:
                """one sided label smoothing for Discriminator"""
                real_label = ((1.2-0.7)*torch.rand((self.batch_size, 1)) + 0.7).to(self.device)
            else:
                real_label = one_label
            
            ##############################
            # Feed Forward               #
            ##############################
            real_X = real_X.to(self.device)
            z = torch.randn(self.batch_size, self.z_dim).to(self.device) + self.z_lower_limit
            r_mu, r_log_var, r, fake_X = self.G(z)
            loss = {}

            sigma = max( 0.0, self.noise_start - (self.global_iters / self.noise_iters) * (self.noise_start - self.noise_end))
            noise_fake_X = self.add_gaussian_noise(fake_X, sigma)
            z_rec, pred_fake = self.D(noise_fake_X.detach())

            noise_real_X = self.add_gaussian_noise(real_X, sigma)
            z_rec_real, pred_real = self.D(noise_real_X)
            
            ##############################
            ##Train D                    #
            ##############################
            # with torch.autograd.set_detect_anomaly(True):
            loss_real = self.GAN_loss(pred_real, real_label)
            loss_fake = self.GAN_loss(pred_fake, fake_label)
            D_loss = loss_real + loss_fake

            self.opt_D.zero_grad()
            D_loss.backward()
            self.opt_D.step()

            loss["D/loss_real"] = loss_real.item()
            loss["D/loss_fake"] = loss_fake.item()
            loss["D/loss_total"] = D_loss.item()

            #############################
            ##Train G, Q                #
            #############################
            z_rec, pred_fake = self.D(noise_fake_X)
            loss_fake = self.GAN_loss(pred_fake, real_label)
            loss_rec = self.reconstruction_loss(input = z, recon = z_rec, name ="IBGAN")
            loss_kld = self.KLD_loss_v1(r_mu, r_log_var)
            if loss_kld.item() >= 20:
                kld_step = 100
            else:
                kld_step = self.upper_weight
            G_loss = loss_fake + loss_rec + kld_step * loss_kld
            
            self.opt_G.zero_grad()
            G_loss.backward()
            self.opt_G.step()
            
            loss["G/loss_fake"] = loss_fake.item()
            loss["G/loss_rec"] = loss_rec.item()
            loss["G/loss_kld"] = loss_kld.item()
            loss["G/loss_total"] = G_loss.item()

            loss["noise_sigma"] = sigma
        

            ###############################
            # Logging                     #
            ###############################
            if self.global_iters % self.print_freq == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration[{}/{}]".format(et, self.global_iters, self.max_iters)
                for tag, value in loss.items():
                    log += ", {}: {:4f}".format(tag, value)
                print(log)

                for tag, value in loss.items():
                    self.logger.scalar_summary(tag, value, self.global_iters)
            
            ####################################
            # Generate Sample with fixed Z, R  #
            ####################################
            if self.global_iters % self.sample_freq == 0:
                with torch.no_grad():
                    ith_sample_dir = os.path.join(self.sample_dir, str(self.global_iters))
                    if not os.path.exists(ith_sample_dir):
                        os.makedirs(ith_sample_dir)
                    
                    r_mu_fixed, _ , _, _ = self.G(z_fixed)
                    for d in range(self.r_dim):
                        img0 = self.G.traverse_latents(self.device,r_mu_fixed, d, start = self.start, end = self.end, steps = self.steps)
                        img1 = self.G.traverse_latents(self.device,r_fixed, d, start = self.start, end = self.end, steps = self.steps)
                        result_path0 = os.path.join(ith_sample_dir, 'dim_zfixed-{}.jpg'.format(d))
                        result_path1 = os.path.join(ith_sample_dir, 'dim_rfixed-{}.jpg'.format(d))
                        save_image(self.denorm(img0.cpu()), result_path0, nrow=self.steps, padding = 0)
                        save_image(self.denorm(img1.cpu()), result_path1, nrow=self.steps, padding = 0)
                    print('Saved samples into {}...'.format(ith_sample_dir))

            ########################################
            # Save model                           #
            ########################################
            if self.global_iters % self.model_save_freq == 0:
                    model_path = os.path.join(self.model_save_dir, "{}-checkpoint.pt".format(self.global_iters))
                    torch.save({
                        'iters': self.global_iters,
                        'G': self.G.state_dict(),
                        'D': self.D.state_dict(),
                        'opt_G': self.opt_G.state_dict(),
                        'opt_D': self.opt_D.state_dict(),
                    }, model_path)
                    # torch.save(self.model.state_dict(), model_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            
            
            self.global_iters += 1
            # if self.global_iters == 2:
            #     break
            
    
    def test(self):
        ###############################################################################
        # Disentangling Score, FID Score Calculation Implementation needed            #
        # Almost same code with generating smaples while training network             #   
        ###############################################################################
        self.load_model(self.test_path, self.test_iters)
        with torch.no_grad():

            z_list = []
            r_list = []
            for seed in self.test_seed:
                np.random.seed(seed)
                z = np.random.normal(size = self.z_dim)
                r = np.random.normal(size = self.r_dim)
                z = np.float32(z)
                r = np.float32(r)
                z = torch.tensor(z)
                r = torch.tensor(r)
                z_list.append(z)
                r_list.append(r)

            test_z = torch.stack(z_list, dim=0)
            test_r = torch.stack(r_list, dim=0)
            r_from_z, _ , _, _ = self.G(test_z)
            for d in range(self.r_dim):
                img0 = self.G.traverse_latents(self.device, r_from_z, d, start = self.start, end = self.end, steps = self.steps)
                img1 = self.G.traverse_latents(self.device, test_r, d, start = self.start, end = self.end, steps = self.steps)
                result_path0 = os.path.join(self.result_dir, 'dim_zfixed-{}.jpg'.format(d))
                result_path1 = os.path.join(self.result_dir, 'dim_rfixed-{}.jpg'.format(d))
                save_image(self.denorm(img0.cpu()), result_path0, nrow=self.steps, padding = 0)
                save_image(self.denorm(img1.cpu()), result_path1, nrow=self.steps, padding = 0)
            print('Saved samples into {}...'.format(self.result_dir))


        return