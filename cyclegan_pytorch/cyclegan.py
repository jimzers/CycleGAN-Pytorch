import torch
from torch.optim import Adam
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib.pyplot import imsave

from cyclegan_pytorch.dataset import make_training_dataloader
from cyclegan_pytorch.networks import Generator, Discriminator


class CycleGAN:
    """
    CycleGAN class to do stuff
    """
    def __init__(self, a_dir, z_dir, file_extension='*.jpg', batch_size=1, num_workers=1,
                 channels=3, img_h=256, img_w=256,
                 lr=0.001, epochs=100, starting_epoch=0, loss_gan_mult=5, loss_cycle_mult=10, loss_identity_mult=10,
                 gen_model_path='./models/gen_model', dis_model_path='./models/dis_model', load_prev_model=False):
        """
        Constructor for the CycleGAN class. Uses GPU if available
        TODO: make work without data directories

        :param a_dir:
        :param z_dir:
        :param file_extension:
        :param batch_size:
        :param num_workers:
        :param channels:
        :param img_h:
        :param img_w:
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_loader = make_training_dataloader(a_dir, z_dir, batch_size, img_h, img_w, num_workers, file_extension)

        self.lr = lr
        self.epochs = epochs
        self.curr_epoch = starting_epoch

        self.gen_az_model_path = gen_model_path + "_az"
        self.gen_za_model_path = gen_model_path + "_za"
        self.dis_a_model_path = dis_model_path + "_a"
        self.dis_z_model_path = dis_model_path + "_z"

        self.img_h = img_h
        self.img_w = img_w

        # TODO: send to device. after debugging loop
        self.g_az = Generator(channels, img_h, img_w).to(self.device)
        self.g_za = Generator(channels, img_h, img_w).to(self.device)

        self.d_a = Discriminator(channels, img_h, img_w).to(self.device)
        self.d_z = Discriminator(channels, img_h, img_w).to(self.device)

        g_params = list(self.g_az.parameters()) + list(self.g_za.parameters())
        d_params = list(self.d_a.parameters()) + list(self.d_z.parameters())

        if load_prev_model:
            self.load_model()

        # todo: use LRScheduler instead
        self.g_opt = Adam(g_params, lr=self.lr)
        self.d_opt = Adam(d_params, lr=self.lr)

        self.loss_gan_mult = loss_gan_mult
        self.loss_cycle_mult = loss_cycle_mult
        self.loss_identity_mult = loss_identity_mult

    def learn(self, batch):
        """
        single step training
        """
        a_imgs_real = batch['a'].to(self.device)
        z_imgs_real = batch['z'].to(self.device)

        # Discriminator Training
        # note: can you run detach on generator outputs here?

        self.d_opt.zero_grad()
        # real images
        guess_d_a = self.d_a(a_imgs_real)
        d_a_same_loss = torch.mean((1 - guess_d_a) ** 2)

        guess_d_z = self.d_z(z_imgs_real)
        d_z_same_loss = torch.mean((1 - guess_d_z) ** 2)

        total_d_real_loss = d_a_same_loss + d_z_same_loss  # TODO: MULTIPLY HERE

        total_d_real_loss.backward()
        self.d_opt.step()

        self.d_opt.zero_grad()

        # generated fake images
        generated_img_a = self.g_za(z_imgs_real)
        guess_d_a = self.d_a(generated_img_a)  # for the resultant of a to z
        d_a_fake_loss = torch.mean(guess_d_a ** 2)  # same as (0 - generated_img_a)**2

        generated_img_z = self.g_az(a_imgs_real)
        guess_d_z = self.d_z(generated_img_z)
        d_z_fake_loss = torch.mean(guess_d_z ** 2)

        total_d_fake_loss = d_a_fake_loss + d_z_fake_loss  # TODO: MULTIPLY HERE

        total_d_fake_loss.backward()
        self.d_opt.step()

        # Generator Training

        self.g_opt.zero_grad()
        # a - z - a cycle loss (and also generator loss)
        generated_img_z = self.g_az(a_imgs_real)
        guess_d_z = self.d_z(generated_img_z)

        g_az_loss = torch.mean((1 - guess_d_z) ** 2)
        # continue the cycle
        generated_cycled_img_a = self.g_za(generated_img_z)

        aza_cycle_loss = torch.mean((a_imgs_real - generated_cycled_img_a) ** 2)

        total_aza_loss = g_az_loss * self.loss_gan_mult + aza_cycle_loss * self.loss_cycle_mult
        total_aza_loss.backward()
        self.g_opt.step()

        self.g_opt.zero_grad()
        # z - a - z cycle loss (and also generator loss)
        generated_img_a = self.g_za(z_imgs_real)
        guess_d_a = self.d_a(generated_img_a)

        g_za_loss = torch.mean((1 - guess_d_a) ** 2)
        # continue the cycle
        generated_cycled_img_z = self.g_az(generated_img_a)

        zaz_cycle_loss = torch.mean((z_imgs_real - generated_cycled_img_z) ** 2)

        total_zaz_loss = g_za_loss * self.loss_gan_mult + zaz_cycle_loss * self.loss_cycle_mult
        total_zaz_loss.backward()
        self.g_opt.step()

        # Identity Loss
        self.g_opt.zero_grad()

        generated_identity_img_z = self.g_az(z_imgs_real)
        identity_loss_az = torch.mean(torch.abs(z_imgs_real - generated_identity_img_z))

        generated_identity_img_a = self.g_za(a_imgs_real)
        identity_loss_za = torch.mean(torch.abs(a_imgs_real - generated_identity_img_a))

        total_identity_loss = (identity_loss_az + identity_loss_za) * self.loss_identity_mult

        total_identity_loss.backward()
        self.g_opt.step()

        return total_d_fake_loss, total_d_real_loss, total_aza_loss, total_zaz_loss, total_identity_loss

    def train(self, log_every=100):
        """
        training loop for CycleGAN

        :param log_every: interval of epochs to regularly run the logging function
        :return:
        """
        print("Starting training at epoch " + str(self.curr_epoch) + "...")
        print("Running for " + str(self.epochs - self.curr_epoch) + " epochs...")
        for epoch_idx in range(self.curr_epoch, self.epochs):
            log_tuple = None
            for i, batch in enumerate(self.data_loader):
                log_tuple = self.learn(batch)

            if epoch_idx % log_every == 0 and log_tuple:
                # total_d_fake_loss, total_d_real_loss, total_aza_loss, total_zaz_loss, total_identity_loss = log_tuple
                self.log_values(epoch_idx, *log_tuple)

    def translate_pillow_img(self, img, a_to_z=True):
        """
        performs domain translation
        :param img:
        :return:
        """
        # force resize and tensorify the image
        predict_transform = transforms.Compose([
            transforms.Resize((int(self.img_h), int(self.img_w)), Image.BICUBIC),
            transforms.ToTensor()
        ])
        if a_to_z:  # a to z
            # turn off grad tracking
            self.g_az.eval()
            # run image modification
            feed_img = predict_transform(img)
            # get prediction
            res = self.g_az(feed_img.unsqueeze(0).to(self.device))
            # set back to training mode
            self.g_az.train()
        else:  # z to a
            # turn off grad tracking
            self.g_za.eval()
            # run image modification
            feed_img = predict_transform(img)
            # get prediction
            res = self.g_za(feed_img.unsqueeze(0).to(self.device))
            # set back to training mode
            self.g_za.train()

        return res.cpu().detach().squeeze().permute(1, 2, 0)  # (H, W, C)

    def translate_img(self, img_path, a_to_z=True, write_to=None):
        """
        performs domain translation given an img path
        :param img_path:
        :param a_to_z: boolean, if false the domain translation will be Z to A instead
        :return: the img
        """
        assert write_to is None or isinstance(write_to, str)
        img = Image.open(img_path)
        res = self.translate_pillow_img(img, a_to_z).numpy()
        if write_to is not None:
            imsave(write_to, res)
        return res

    def log_values(self, epoch, total_d_fake_loss, total_d_real_loss, total_aza_loss, total_zaz_loss, total_identity_loss):
        # TODO: Write to a file output
        print('Epoch:')
        print(epoch)
        print('Total Discriminator Fake Loss:')
        print(total_d_fake_loss)
        print('Total Discriminator Real Loss:')
        print(total_d_real_loss)
        print('Total AZA Cycle Loss:')
        print(total_aza_loss)
        print('Total ZAZ Cycle Loss:')
        print(total_zaz_loss)
        print('Total Identity Loss:')
        print(total_identity_loss)

    def save_model(self):
        # TODO
        print("<-------------------- Saving model... -------------------->")
        torch.save(self.g_az.state_dict(), self.gen_az_model_path)
        torch.save(self.g_za.state_dict(), self.gen_za_model_path)
        torch.save(self.d_a.state_dict(), self.dis_a_model_path)
        torch.save(self.d_z.state_dict(), self.dis_z_model_path)

    def load_model(self):
        # TODO
        print("<-------------------- Loading model... -------------------->")
        self.g_az.load_state_dict(torch.load(self.gen_az_model_path))
        self.g_za.load_state_dict(torch.load(self.gen_za_model_path))
        self.d_a.load_state_dict(torch.load(self.dis_a_model_path))
        self.d_z.load_state_dict(torch.load(self.dis_z_model_path))
