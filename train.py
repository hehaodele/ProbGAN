from torch_head import *
from common_head import *
from models import *
from utils import *
from train_utils import *
from inception.inception_score_tf import get_inception_score
from easydict import EasyDict


def parse():
    parser = argparse.ArgumentParser()
    """
    Hyper-parameter for Stochastic Gradient Hamiltonian Monte Carlo
    """
    parser.add_argument('--sghmc_alpha', default=0.01, type=int, dest='sghmc_alpha', help='number of generators')
    parser.add_argument('--g_noise_loss_lambda', default=3e-2, type=float, dest='g_noise_loss_lambda')
    parser.add_argument('--d_noise_loss_lambda', default=3e-2, type=float, dest='d_noise_loss_lambda')
    parser.add_argument('--d_hist_loss_lambda', default=1.0, type=float, dest='d_hist_loss_lambda')
    """
    GAN objectives
    NS: original GAN (Non-saturating version)
    MM: original GAN (Min-max version)
    W: Wasserstein GAN
    LS: Least-Square GAN
    """
    parser.add_argument('--gan_obj', default='NS', type=str, dest='gan_obj', help='[NS | MM | LS | W]')

    """
    Paths
    """
    parser.add_argument('--dataset', default='cifar', type=str, dest='dataset', help='dataset: [cifar10, stl10, imagenet]')
    parser.add_argument('--save_dir', default='none', type=str, dest='save_dir', help='save_path')

    return parser.parse_args()


def construct_model(args, config):
    '''
    :param args: Experiment Information
    :param config: Neural Network Architecture Configurations
    :return:
        G: generator structure
        D: discriminator structure
    '''
    D_unbound_output = args.gan_obj in ['W', 'LS']
    if config.image_size == 32:
        G = multi_generator_32(z_size=config.z_size, out_size=config.channel_size, ngf=config.ngf,
                               num_gens=config.num_gens).cuda()
        D = multi_discriminator_with_history(in_size=config.channel_size, ndf=config.ndf, num_discs=config.num_discs,
                                             unbound_output=D_unbound_output).cuda()
    if config.image_size == 48:
        G = multi_generator_48(z_size=config.z_size, out_size=config.channel_size, ngf=config.ngf,
                               num_gens=config.num_gens).cuda()
        D = multi_discriminator_48_with_history(in_size=config.channel_size, ndf=config.ndf, num_discs=config.num_discs,
                                                unbound_output=D_unbound_output).cuda()

    print('G #parameters: ', count_parameters(G))
    print('D #parameters: ', count_parameters(D))
    return G, D


def train_net(G, D, args, config):
    cudnn.benchmark = True

    noise_std = np.sqrt(2 * args.sghmc_alpha)
    G_noise_sampler = [get_sghmc_noise(g) for g in G.gs]
    D_noise_sampler = get_sghmc_noise(D)

    if args.save_dir == 'none':
        args.save_dir = './dump/train_{}_{}'.format(args.dataset, args.gan_obj)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_loader = torch.utils.data.DataLoader(
        dataset=config.get_dataset(),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True)

    # setup loss function
    criterion_bce = nn.BCELoss().cuda()
    criterion_mse = nn.MSELoss().cuda()
    if args.gan_obj == 'NS':
        phi_1 = lambda dreal, lreal, lfake: criterion_bce(dreal, lreal)
        phi_2 = lambda dfake, lreal, lfake: criterion_bce(dfake, lfake)
        phi_3 = lambda dfake, lreal, lfake: criterion_bce(dfake, lreal)
    elif args.gan_obj == 'MM':
        phi_1 = lambda dreal, lreal, lfake: criterion_bce(dreal, lreal)
        phi_2 = lambda dfake, lreal, lfake: criterion_bce(dfake, lfake)
        phi_3 = lambda dfake, lreal, lfake: - criterion_bce(dfake, lfake)
    elif args.gan_obj == 'LS':
        phi_1 = lambda dreal, lreal, lfake: criterion_mse(dreal, lreal)
        phi_2 = lambda dfake, lreal, lfake: criterion_mse(dfake, lfake)
        phi_3 = lambda dfake, lreal, lfake: criterion_mse(dfake, lreal)
    elif args.gan_obj == 'W':
        phi_1 = lambda dreal, lreal, lfake: - dreal.mean()
        phi_2 = lambda dfake, lreal, lfake: dfake.mean()
        phi_3 = lambda dfake, lreal, lfake: - dfake.mean()

    num_gs = len(G.gs)
    num_ds = D.n_ds

    # setup optimizer
    optimizerD = torch.optim.Adam(D.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))

    # setup some varibles
    batch_time = AverageMeter()
    data_time = AverageMeter()
    D_losses = AverageMeter()
    G_losses = AverageMeter()
    G_n_losses = AverageMeter()

    fixed_noise = torch.FloatTensor(10 * 10, config.z_size, 1, 1).normal_(0, 1)
    fixed_noise = Variable(fixed_noise.cuda(), volatile=True)

    end = time.time()

    D.train()
    G.train()

    D_loss_list = []
    G_loss_list = []
    G_loss_by_hist_list = []
    score_list = []

    for epoch in range(config.epoches):
        bar = ProgressBar()
        i = 0
        for input, _ in bar(train_loader):
            '''
                Update D network:
            '''
            data_time.update(time.time() - end)

            batch_size = input.size(0)
            g_batch_size = config.g_batch_size
            assert g_batch_size >= batch_size

            input_var = Variable(input.cuda())

            # Train discriminator with real data
            label_real = torch.ones(max(batch_size, g_batch_size))
            label_real_var = Variable(label_real.cuda())

            D_real_result = D(input_var).mean(-1).mean(-1).mean(-1)
            D_real_loss = phi_1(D_real_result, label_real_var[:batch_size], None)

            # Train discriminator with fake data
            label_fake = torch.zeros(g_batch_size)
            label_fake_var = Variable(label_fake.cuda())

            noise = torch.randn((g_batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).mean(-1).mean(-1).mean(-1)
            D_fake_loss = phi_2(D_fake_result, None, label_fake_var)

            # Back propagation
            D_train_loss = D_real_loss + D_fake_loss
            D_losses.update(D_train_loss.item())
            D_noise_loss = args.d_noise_loss_lambda * noise_loss(model=D, noise_sampler=D_noise_sampler,
                                                                 alpha=noise_std)
            D_train_loss += D_noise_loss

            if args.gan_obj == 'W':
                gradient_penalty = calc_gradient_penalty(D, input_var.data, G_result[:batch_size].data)
                D_train_loss += gradient_penalty

            D.zero_grad()
            D_train_loss.backward()
            optimizerD.step()

            '''
                Update G network:
            '''
            noise = torch.randn((g_batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).mean(-1).mean(-1).mean(-1)
            G_train_loss = phi_3(D_fake_result, label_real_var, label_fake_var)

            G_losses.update(G_train_loss.item())

            G_noise_loss = args.g_noise_loss_lambda * \
                           sum([noise_loss(model=g, noise_sampler=s, alpha=noise_std) for g, s in zip(G.gs,
                                                                                                      G_noise_sampler)])
            G_train_loss += G_noise_loss

            D_fake_result_hist = D.forward_by_hist(G_result).mean(-1).mean(-1).mean(-1)
            G_train_loss_by_hist = phi_3(D_fake_result_hist, label_real_var, label_fake_var)
            G_train_loss += G_train_loss_by_hist * args.d_hist_loss_lambda
            G_n_losses.update(G_train_loss_by_hist.item())

            # Back propagation
            D.zero_grad()
            G.zero_grad()
            G_train_loss.backward()
            optimizerG.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                """ update history discriminators aggregations
                """
                # print(F.l1_loss(D.ds.weight.data, D.ds_hist_avg.weight.data))
                D.update_hist()

            if (i + 1) % config.display == 0:
                print_log_2(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr,
                            config.display, batch_time, data_time, D_losses, G_losses, G_n_losses)
                batch_time.reset()
                data_time.reset()
            elif (i + 1) == len(train_loader):
                print_log_2(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr,
                            (i + 1) % config.display, batch_time, data_time, D_losses, G_losses, G_n_losses)
                batch_time.reset()
                data_time.reset()

            i += 1

        D_loss_list.append(D_losses.avg)
        G_loss_list.append(G_losses.avg)
        G_loss_by_hist_list.append(G_n_losses.avg)

        D_losses.reset()
        G_losses.reset()
        G_n_losses.reset()

        if (epoch + 1) < config.dump_ep:
            plot_result(G, fixed_noise, config.image_size, epoch + 1, args.save_dir, is_gray=(config.channel_size == 1))
            plot_loss_my(D_loss_list, G_loss_list, G_loss_by_hist_list, epoch + 1, config.epoches, args.save_dir)

        if (epoch + 1) % config.dump_ep == 0:
            # plt the generate images and loss curve
            plot_result(G, fixed_noise, config.image_size, epoch + 1, args.save_dir, is_gray=(config.channel_size == 1))
            plot_loss_my(D_loss_list, G_loss_list, G_loss_by_hist_list, epoch + 1, config.epoches, args.save_dir)
            # save the D and G.
            save_checkpoint({'epoch': epoch, 'state_dict': D.state_dict(), },
                            os.path.join(args.save_dir, 'D_epoch_{}'.format(epoch)))
            save_checkpoint({'epoch': epoch, 'state_dict': G.state_dict(), },
                            os.path.join(args.save_dir, 'G_epoch_{}'.format(epoch)))
            # plot gradient information
            tmp = grad_info(G.parameters())
            print('G grad l2-norm: {}, value max: {}'.format(tmp[0], tmp[1]))
            tmp = grad_info(D.parameters())
            print('D grad l2-norm: {}, value max: {}'.format(tmp[0], tmp[1]))

        if (epoch + 1) % config.dump_ep == 0:
            batch_size = 100
            total_size = 5000
            x = []
            for i in range(total_size // batch_size):
                noise = torch.randn((batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
                noise_var = Variable(noise.cuda())
                G_result = G(noise_var)
                x.append(G_result.detach().cpu().numpy())
            x = np.concatenate(x, axis=0)
            imgs = x
            m, v = get_inception_score(images=imgs)
            # fid = get_fid_score(images=imgs)
            fid = 0
            print('Epoch {} Inception Score: mean {:.6f} std {:.6f}'.format(epoch + 1, m, v))
            print('Epoch {} FID Score: {:.6f}'.format(epoch + 1, fid))

            score_list.append([epoch + 1, m, v, fid])
            plot_scores(score_list, args.save_dir)


if __name__ == '__main__':
    os.system('mkdir -p logs')

    args = parse()
    print(args)

    config = EasyDict()


    """
    Number of Generator/Discriminator Monte-Carlo samples
    """
    config.num_gens = 10
    config.num_discs = 4

    """
    Architecture Hyper-parameters
    """
    config.z_size = 100
    config.channel_size = 3
    config.ngf = 128
    config.ndf = 128

    """
    Training Hyper-parameters
    """
    config.workers = 10
    config.display = 800

    config.batch_size = 64
    config.g_batch_size = 128
    config.base_lr = 0.0001
    config.beta1 = 0.5


    if args.dataset == 'cifar10':
        from datasets import get_cifar10

        config.get_dataset = get_cifar10

        config.epoches = 250
        config.image_size = 32
        config.dump_ep = 20

    if args.dataset == 'stl10':
        from datasets import get_stl10

        config.get_dataset = get_stl10

        config.epoches = 250
        config.image_size = 48
        config.dump_ep = 10

    if args.dataset == 'imagenet':
        from datasets import get_imagenet

        config.get_dataset = get_imagenet

        config.epoches = 50
        config.image_size = 32
        config.dump_ep = 5

    G, D = construct_model(args=args, config=config)
    train_net(G, D, args, config)
