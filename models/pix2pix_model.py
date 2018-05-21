import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG1,self.netG2,self.netG3,self.sscale,self.cls = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netDs = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D-4, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG1, 'G1', opt.which_epoch)
            self.load_network(self.netG2, 'G2', opt.which_epoch)
            self.load_network(self.netG3, 'G3', opt.which_epoch)
            #self.load_network(self.sscale, 'sscale', opt.which_epoch)
            self.load_network(self.cls, 'cls', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netDs, 'Ds', opt.which_epoch)
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_sAB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCls = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam([{'params' : self.netG1.parameters()},
                                                 {'params' : self.netG2.parameters()},
                                                 {'params' : self.netG3.parameters()},
                                                 {'params' : self.sscale.parameters()},
                                                 {'params' : self.cls.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.95))
            self.optimizer_D = torch.optim.Adam([{'params' : self.netD.parameters()},
                                                 {'params' : self.netDs.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.95))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG1)
        networks.print_network(self.netG2)
        networks.print_network(self.netG3)
        networks.print_network(self.sscale)
        networks.print_network(self.cls)
        if self.isTrain:
            networks.print_network(self.netD)
            networks.print_network(self.netDs)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        if self.isTrain:
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            input_sA = input['sA' if AtoB else 'sB']
            input_sB = input['sB' if AtoB else 'sA']
        else:
            input_A = input['A']
            input_B = input['B']


        label = input['label']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        if self.isTrain:
            input_sA = input_sA.cuda(self.gpu_ids[0], async=True)
            input_sB = input_sB.cuda(self.gpu_ids[0], async=True)
            self.input_sA = input_sA
            self.input_sB = input_sB
        self.input_A = input_A
        self.input_B = input_B

        self.label = label      
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.map1 = self.netG1(self.real_A)
        self.map2 = self.netG2(self.map1)
        self.fake_B = self.netG3(self.map2)
        self.fake_sB = self.sscale(self.map2)
        #print(type(self.fake_sB))
        self.real_B = Variable(self.input_B)
        self.pred_label = self.cls(self.map1)
        self.real_sA = Variable(self.input_sA)
        self.real_sB = Variable(self.input_sB)
        #self.fake_B = Variable(self.fake_B)
        _, preds = torch.max(self.pred_label.data,1)
       
        self.label = Variable(self.label.view(-1).cuda())
        self.accuracy = torch.sum(preds == self.label.data)/torch.numel(preds)

    # no backprop gradients
    def test(self):
        self.input_A = torch.chunk(self.input_A,10,2)
        self.fake_B = []
        #print(self.input_A[0].shape)
        for i in range(0,10):
            self.real_A = Variable(self.input_A[i], volatile=True)
            self.map1 = self.netG1(self.real_A)
            self.map2 = self.netG2(self.map1)
            self.fake_BB = self.netG3(self.map2)
            self.fake_B.append(self.fake_BB)
            
        self.real_B = self.input_B
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        fake_sAB = self.fake_sAB_pool.query(torch.cat((self.real_sA, self.fake_sB), 1).data)
        pred_sfake = self.netDs(fake_sAB.detach())

        self.loss_Ds_fake = self.criterionGAN(pred_sfake, False)
        real_sAB = torch.cat((self.real_sA, self.real_sB), 1)

        pred_sreal = self.netDs(real_sAB)
        self.loss_Ds_real = self.criterionGAN(pred_sreal, True)
        #print(pred_sreal.shape)
        #print(pred_real.shape)
        # Combined loss

        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.loss_Ds_fake + self.loss_Ds_real) * 0.25

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_cls = self.criterionCls(self.pred_label, self.label)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_Gs_L1 = self.criterionL1(self.fake_sB,self.real_sB) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_cls + 4*self.loss_Gs_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0]),
                            ('Ds_real', self.loss_Ds_real.data[0]),
                            ('Ds_fake', self.loss_Ds_fake.data[0]),
                            ('Cls_loss', self.loss_cls.data[0]),
                            ('Accuracy', self.accuracy*20),
                            ('Gs_L1',self.loss_Gs_L1.data[0])
                            ])

    def get_current_visuals(self):
        n,c,h,w = self.real_A.shape
        mask_A = self.real_A.data[:,3,:,:]
        zero_mask = mask_A<0.0
        one_mask = mask_A>=0.0
        mask_A[zero_mask] = 1.0
        mask_A[one_mask] = 0.5
        mask_A = mask_A.contiguous()
        #print(mask_A.shape)
        mask_A = mask_A.view(n,1,h,w)
        mask_A = mask_A.expand(n,3,h,w)
        real_A = util.tensor2im(self.real_A.data[:,0:3,:,:]*mask_A)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_sB = util.tensor2im(self.fake_sB.data)
        real_sB = util.tensor2im(self.real_sB.data)
        real_sA = util.tensor2im(self.real_sA.data)
        #print(type(fake_sB),type(fake_B))
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('fake_sB', fake_sB), ('real_B', real_B), ('real_sB', real_sB), ('real_sA', real_sA)])
    def get_current_visual_test(self):
        ori_img = util.tensor2im(self.input_B)
        #print(type(self.fake_B[0].data))
        pose_1 = util.tensor2im(self.fake_B[0].data)
        pose_2 = util.tensor2im(self.fake_B[1].data)
        pose_3 = util.tensor2im(self.fake_B[2].data)
        pose_4 = util.tensor2im(self.fake_B[3].data)
        pose_5 = util.tensor2im(self.fake_B[4].data)
        pose_6 = util.tensor2im(self.fake_B[5].data)
        pose_7= util.tensor2im(self.fake_B[6].data)
        pose_8 = util.tensor2im(self.fake_B[7].data)
        pose_9 = util.tensor2im(self.fake_B[8].data)
        pose_10 = util.tensor2im(self.fake_B[9].data)
        return OrderedDict([('ori_img', ori_img), ('pose_1', pose_1), ('pose_2', pose_2),\
            ('pose_3', pose_3), ('pose_4', pose_4), ('pose_5', pose_5), ('pose_6', pose_6), ('pose_7', pose_7), ('pose_8', pose_8), ('pose_9', pose_9)])
    def save(self, label):
        self.save_network(self.netG1, 'G1', label, self.gpu_ids)
        self.save_network(self.netG2, 'G2', label, self.gpu_ids)
        self.save_network(self.netG3, 'G3', label, self.gpu_ids)
        self.save_network(self.cls, 'cls', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
