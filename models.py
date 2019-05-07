import torch
import torch.nn as nn

# 32 x 32 ==============================================================================================================
class multi_generator_32(nn.Module):
    """
    Generative Network for images with the size of 32 x 32
    Parameters:
        ngf: number of feature channels
        num_gens: number of generators
    """

    def __init__(self, z_size=100, out_size=3, ngf=128, num_gens=2):
        super(multi_generator_32, self).__init__()
        self.z_size = z_size
        self.ngf = ngf
        self.out_size = out_size

        self.main = nn.Sequential(
            # state size: (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            # state size: ngf x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.out_size, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: out_size x 32 x 32
        )

        self.gs = []
        for i in range(num_gens):
            g = nn.Sequential(
                # input size is z_size
                nn.ConvTranspose2d(self.z_size, self.ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(inplace=True),
            )
            setattr(self, 'G_{}'.format(i), g)
            self.gs.append(g)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        sp_size = (len(x) - 1) // len(self.gs) + 1
        y = []
        for _x, _g in zip(torch.split(x, sp_size, dim=0), self.gs):
            y.append(_g(_x))
        y = torch.cat(y, dim=0)

        output = self.main(y)

        return output


# multiple discriminators 32x32
class multi_discriminator_32(nn.Module):
    """
    Discriminative Network
    Parameters:
        ndf: number of feature channels
        num_dics: number of discriminators
    """

    def __init__(self, in_size=3, ndf=128, num_discs=2, unbound_output=False):
        super(multi_discriminator_32, self).__init__()
        self.in_size = in_size
        self.ndf = ndf

        self.ds = nn.Conv2d(self.ndf * 4, num_discs, 4, 1, 0, bias=False)
        self.n_ds = num_discs

        self.main = nn.Sequential(
            # input size is in_size x 32 x 32
            nn.Conv2d(self.in_size, self.ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 8 x 8
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 4 x 4
            self.ds,
        )

        if not unbound_output:
            self.main = nn.Sequential(
                self.main,
                nn.Sigmoid(),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        output = self.main(input)
        return output


class multi_discriminator_with_history(nn.Module):
    """
    Discriminative Network revised for ProbGAN
    Keep historial samples of discriminators

    Parameters:
        ndf: number of feature channels
        num_dics: number of discriminators
    """

    def __init__(self, in_size=3, ndf=128, num_discs=2, unbound_output=False):
        super(multi_discriminator_with_history, self).__init__()
        self.unbound_output = unbound_output
        self.in_size = in_size
        self.ndf = ndf

        self.ds = nn.Conv2d(self.ndf * 4, num_discs, 4, 1, 0, bias=False)
        self.n_ds = num_discs

        with torch.no_grad():
            self.ds_hist_avg = nn.Conv2d(self.ndf * 4, num_discs, 4, 1, 0, bias=False)
            self.len_hist = 1.0

        self.backbone = nn.Sequential(
            # input size is in_size x 32 x 32
            nn.Conv2d(self.in_size, self.ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 8 x 8
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 4 x 4
        )

        self.main = nn.Sequential(self.backbone, self.ds, )
        self.main_hist = nn.Sequential(self.backbone, self.ds_hist_avg, )

        if not unbound_output:
            self.main = nn.Sequential(self.main, nn.Sigmoid(), )
            self.main_hist = nn.Sequential(self.main_hist, nn.Sigmoid(), )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.eps = 1e-3

    def forward(self, input):
        output = self.main(input)
        if not self.unbound_output:
            output = output * (1 - 2 * self.eps) + self.eps
        return output

    def forward_by_hist(self, input):
        output = self.main_hist(input)
        if not self.unbound_output:
            output = output * (1 - 2 * self.eps) + self.eps
        return output

    def update_hist(self):
        self.len_hist += 1
        alpha = 1.0 / self.len_hist
        # alpha = 1e-3
        self.ds_hist_avg.weight.data = self.ds_hist_avg.weight.data * (1 - alpha) + self.ds.weight.data * alpha


# 48 x 48 ==============================================================================================================
class multi_generator_48(nn.Module):
    """
        Generative Network
    """

    def __init__(self, z_size=100, out_size=3, ngf=128, num_gens=2):
        super(multi_generator_48, self).__init__()
        self.z_size = z_size
        self.ngf = ngf
        self.out_size = out_size

        self.main = nn.Sequential(
            # state size: (ngf * 8) x 3 x 3
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            # state size: (ngf * 4) x 6 x 6
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) x 12 x 12
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            # state size: ngf x 24 x 24
            nn.ConvTranspose2d(self.ngf, self.out_size, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: out_size x 48 x 48
        )

        self.gs = []
        for i in range(num_gens):
            g = nn.Sequential(
                # input size is z_size
                nn.ConvTranspose2d(self.z_size, self.ngf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(inplace=True),
            )
            setattr(self, 'G_{}'.format(i), g)
            self.gs.append(g)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        sp_size = (len(x) - 1) // len(self.gs) + 1
        y = []
        for _x, _g in zip(torch.split(x, sp_size, dim=0), self.gs):
            y.append(_g(_x))
        y = torch.cat(y, dim=0)

        output = self.main(y)

        return output


class multi_discriminator_48(nn.Module):
    """
        Discriminative Network
    """

    def __init__(self, in_size=3, ndf=128, num_discs=2, unbound_output=False):
        super(multi_discriminator_48, self).__init__()
        self.in_size = in_size
        self.ndf = ndf

        self.ds = nn.Conv2d(self.ndf * 8, num_discs, 3, 1, 0, bias=False)
        self.n_ds = num_discs

        self.main = nn.Sequential(
            # input size is in_size x 48 x 48
            nn.Conv2d(self.in_size, self.ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 24 x 24
            nn.Conv2d(self.ndf, self.ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 12 x 12
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 6 x 6
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8) x 3 x 3
            self.ds,
        )

        if not unbound_output:
            self.main = nn.Sequential(
                self.main,
                nn.Sigmoid(),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        output = self.main(input)
        return output


class multi_discriminator_48_with_history(nn.Module):
    """
        Discriminative Network
    """

    def __init__(self, in_size=3, ndf=128, num_discs=2, unbound_output=False):
        super(multi_discriminator_48_with_history, self).__init__()
        self.unbound_output = unbound_output
        self.in_size = in_size
        self.ndf = ndf

        self.ds = nn.Conv2d(self.ndf * 8, num_discs, 3, 1, 0, bias=False)
        self.n_ds = num_discs

        with torch.no_grad():
            self.ds_hist_avg = nn.Conv2d(self.ndf * 8, num_discs, 3, 1, 0, bias=False)
            self.len_hist = 1.0

        self.backbone = nn.Sequential(
            # input size is in_size x 48 x 48
            nn.Conv2d(self.in_size, self.ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 24 x 24
            nn.Conv2d(self.ndf, self.ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 12 x 12
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 6 x 6
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8) x 3 x 3
        )

        self.main = nn.Sequential(self.backbone, self.ds, )
        self.main_hist = nn.Sequential(self.backbone, self.ds_hist_avg, )

        if not unbound_output:
            self.main = nn.Sequential(self.main, nn.Sigmoid(), )
            self.main_hist = nn.Sequential(self.main_hist, nn.Sigmoid(), )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.eps = 1e-3

    def forward(self, input):
        output = self.main(input)
        if not self.unbound_output:
            output = output * (1 - 2 * self.eps) + self.eps
        return output

    def forward_by_hist(self, input):
        output = self.main_hist(input)
        if not self.unbound_output:
            output = output * (1 - 2 * self.eps) + self.eps
        return output

    def update_hist(self):
        self.len_hist += 1
        alpha = 1.0 / self.len_hist
        # alpha = 1e-3
        self.ds_hist_avg.weight.data = self.ds_hist_avg.weight.data * (1 - alpha) + self.ds.weight.data * alpha


# ======================================================================================================================
if __name__ == '__main__':
    G = multi_generator_48()
    z = torch.zeros(10, 100, 1, 1).to(torch.float)
    x = G(z)
    print(x.shape)
    D = multi_discriminator_48()
    y = D(x)
    print(y.shape)

    D2 = multi_discriminator_48_with_history()
    y2 = D2(x)
    y2_ = D2.forward_by_hist(x)
    print(y2.shape, y2_.shape)
