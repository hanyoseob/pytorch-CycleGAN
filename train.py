from model import *
from dataset import *

import itertools

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_c_a = args.wgt_c_a
        self.wgt_c_b = args.wgt_c_b
        self.wgt_i = args.wgt_i

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.direction = args.direction
        self.name_data = args.name_data

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                    'netD_a': netD_a.state_dict(), 'netD_b': netD_b.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch, mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])
            netD_a.load_state_dict(dict_net['netD_a'])
            netD_b.load_state_dict(dict_net['netD_b'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch

        elif mode == 'test':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])

            return netG_a2b, netG_b2a, epoch

    def preprocess(self, data):
        nomalize = Nomalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(nomalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_c_a = self.wgt_c_a
        wgt_c_b = self.wgt_c_b
        wgt_i = self.wgt_i

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')

        transform_train = transforms.Compose([Nomalize(), RandomFlip(), Rescale((self.ny_load, self.nx_load)), RandomCrop((self.ny_in, self.nx_in)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = Dataset(dir_data_train, direction=self.direction, data_type=self.data_type, transform=transform_train)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        netG_a2b = UNet(nch_in, nch_out, nch_ker, norm)
        netG_b2a = UNet(nch_in, nch_out, nch_ker, norm)
        # netG_a2b = ResNet(nch_in, nch_out, nch_ker, norm)
        # netG_b2a = ResNet(nch_in, nch_out, nch_ker, norm)

        netD_a = Discriminator(nch_in, nch_ker, norm)
        netD_b = Discriminator(nch_in, nch_ker, norm)
        
        init_net(netG_a2b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG_b2a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        init_net(netD_a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD_b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_Cycle = nn.L1Loss().to(device)   # L1
        fn_GAN = nn.BCELoss().to(device)
        fn_Ident = nn.L1Loss().to(device)   # L1

        paramsG_a2b = netG_a2b.parameters()
        paramsG_b2a = netG_b2a.parameters()
        paramsD_a = netD_a.parameters()
        paramsD_b = netD_b.parameters()

        optimG = torch.optim.Adam(itertools.chain(paramsG_a2b, paramsG_b2a), lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(itertools.chain(paramsD_a, paramsD_b), lr=lr_D, betas=(self.beta1, 0.999))

        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, st_epoch = \
                self.load(dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netD_b.train()

            loss_G_a2b_train = 0
            loss_G_b2a_train = 0
            loss_D_a_train = 0
            loss_D_b_train = 0
            loss_C_a_train = 0
            loss_C_b_train = 0
            loss_I_a_train = 0
            loss_I_b_train = 0

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input_a = data['dataA'].to(device)
                input_b = data['dataB'].to(device)

                # forward netG
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)

                recon_b = netG_a2b(output_a)
                recon_a = netG_b2a(output_b)

                # backward netD
                set_requires_grad([netD_a, netD_b], True)
                optimD.zero_grad()

                # backward netD_a
                pred_real_a = netD_a(input_a)
                pred_fake_a = netD_a(output_a.detach())

                loss_D_a_real = fn_GAN(pred_real_a, torch.ones_like(pred_real_a))
                loss_D_a_fake = fn_GAN(pred_fake_a, torch.zeros_like(pred_fake_a))
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                # backward netD_b
                pred_real_b = netD_b(input_b)
                pred_fake_b = netD_b(output_b.detach())

                loss_D_b_real = fn_GAN(pred_real_b, torch.ones_like(pred_real_b))
                loss_D_b_fake = fn_GAN(pred_fake_b, torch.zeros_like(pred_fake_b))
                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

                # backward netD
                loss_D = loss_D_a + loss_D_b
                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad([netD_a, netD_b], False)
                optimG.zero_grad()

                if wgt_i > 0:
                    ident_b = netG_a2b(input_b)
                    ident_a = netG_b2a(input_a)

                    loss_I_a = fn_Ident(ident_a, input_a)
                    loss_I_b = fn_Ident(ident_b, input_b)
                else:
                    loss_I_a = 0
                    loss_I_b = 0

                pred_fake_a = netD_a(output_a)
                pred_fake_b = netD_b(output_b)

                loss_G_a2b = fn_GAN(pred_fake_b, torch.ones_like(pred_fake_b))
                loss_G_b2a = fn_GAN(pred_fake_a, torch.ones_like(pred_fake_a))

                loss_C_a = fn_Cycle(input_a, recon_a)
                loss_C_b = fn_Cycle(input_b, recon_b)

                loss_G = (loss_G_a2b + loss_G_b2a) + \
                         (wgt_c_a * loss_C_a + wgt_c_b * loss_C_b) + \
                         (wgt_c_a * loss_I_a + wgt_c_b * loss_I_b) * wgt_i

                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_a2b_train += loss_G_a2b.item()
                loss_G_b2a_train += loss_G_b2a.item()

                loss_D_a_train += loss_D_a.item()
                loss_D_b_train += loss_D_b.item()

                loss_C_a_train += loss_C_a.item()
                loss_C_b_train += loss_C_b.item()

                if wgt_i > 0:
                    loss_I_a_train += loss_I_a.item()
                    loss_I_b_train += loss_I_b.item()

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'G_a2b: %.4f G_b2a: %.4f D_a: %.4f D_b: %.4f C_a: %.4f C_b: %.4f I_a: %.4f I_b: %.4f'
                      % (epoch, i, num_batch_train,
                         loss_G_a2b_train / i, loss_G_b2a_train / i,
                         loss_D_a_train / i, loss_D_b_train / i,
                         loss_C_a_train / i, loss_C_b_train / i,
                         loss_I_a_train / i, loss_I_b_train / i))

                if should(num_freq_disp):
                    ## show output
                    input_a = transform_inv(input_a)
                    output_a = transform_inv(output_a)
                    input_b = transform_inv(input_b)
                    output_b = transform_inv(output_b)

                    writer_train.add_images('input_a', input_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_a', output_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('input_b', input_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_b', output_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_G_a2b', loss_G_a2b_train / num_batch_train, epoch)
            writer_train.add_scalar('loss_G_b2a', loss_G_b2a_train / num_batch_train, epoch)
            writer_train.add_scalar('loss_D_a', loss_D_a_train / num_batch_train, epoch)
            writer_train.add_scalar('loss_D_b', loss_D_b_train / num_batch_train, epoch)
            writer_train.add_scalar('loss_C_a', loss_C_a_train / num_batch_train, epoch)
            writer_train.add_scalar('loss_C_b', loss_C_b_train / num_batch_train, epoch)
            writer_train.add_scalar('loss_I_a', loss_I_a_train / num_batch_train, epoch)
            writer_train.add_scalar('loss_I_b', loss_I_b_train / num_batch_train, epoch)

            # # update schduler
            # # schedG.step()
            # # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(self.dir_result, self.scope)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        dir_data_test = os.path.join(self.dir_data, 'test')

        transform_test = transforms.Compose([Nomalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        netG_a2b = UNet(nch_in, nch_out, nch_ker, norm)
        netG_b2a = UNet(nch_in, nch_out, nch_ker, norm)
        # netG_a2b = ResNet(nch_in, nch_out, nch_ker, norm)
        # netG_b2a = ResNet(nch_in, nch_out, nch_ker, norm)

        netD_a = Discriminator(nch_in, nch_ker, norm)
        netD_b = Discriminator(nch_in, nch_ker, norm)

        init_net(netG_a2b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG_b2a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        init_net(netD_a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD_b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## load from checkpoints
        st_epoch = 0

        netG_a2b, netG_b2a, st_epoch = self.load(dir_chck, netG_a2b, netG_b2a, mode=mode)

        ## test phase
        with torch.no_grad():
            netG_a2b.eval()
            netG_b2a.eval()
            # netG_a2b.train()
            # netG_b2a.train()

            gen_loss_l1_test = 0
            for i, data in enumerate(loader_test, 1):
                input_a = data['dataA'].to(device)
                input_b = data['dataB'].to(device)

                # forward netG
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)

                recon_b = netG_a2b(output_a)
                recon_a = netG_b2a(output_b)

                input_a = transform_inv(input_a)
                input_b = transform_inv(input_b)
                output_a = transform_inv(output_a)
                output_b = transform_inv(output_b)
                recon_a = transform_inv(recon_a)
                recon_b = transform_inv(recon_b)

                for j in range(batch_size):
                    name = batch_size * (i - 1) + j
                    fileset = {'name': name,
                               'input_a': "%04d-input_a.png" % name,
                               'input_b': "%04d-input_b.png" % name,
                               'output_a': "%04d-output_a.png" % name,
                               'output_b': "%04d-output_b.png" % name,
                               'recon_a': "%04d-recon_a.png" % name,
                               'recon_b': "%04d-recon_b.png" % name}

                    plt.imsave(os.path.join(dir_result_save, fileset['input_a']), input_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['input_b']), input_b[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['output_a']), output_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['output_b']), output_b[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['recon_a']), recon_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['recon_b']), recon_b[j, :, :, :].squeeze())

                    append_index(dir_result, fileset)



def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input_a</th><th>input_b</th><th>output_a</th><th>output_b</th><th>recon_a</th><th>recon_b</th></tr>")

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    for kind in ["input_a", "input_b", "output_a", "output_b", "recon_a", "recon_b"]:
        index.write("<td><img src='images/%s'></td>" % fileset[kind])

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
