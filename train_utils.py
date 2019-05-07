from torch_head import *
from common_head import *


def to_numpy(x):
    return x.detach().cpu().numpy()


def to_tensor(x):
    return torch.tensor(x).cuda()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_gradient_penalty(netD, real_data, fake_data, gp_lambda=10):
    # print real_data.size()
    assert len(real_data) == len(fake_data)
    alpha = torch.rand(len(real_data), 1, 1, 1)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def calc_gradient_penalty_mgan(netD, real_data, fake_data, gp_lambda=10):
    # print real_data.size()
    assert len(real_data) == len(fake_data)
    alpha = torch.rand(len(real_data), 1, 1, 1)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def noise_loss(model,
               noise_sampler,
               alpha):
    loss = 0
    for p, n in zip(model.parameters(), noise_sampler):
        n.normal_(mean=0, std=alpha)
        loss += torch.sum(p * n)
    return loss


def get_sghmc_noise(model):
    return [to_tensor(torch.zeros(p.size())) for p in model.parameters()]


# ======================================================================================================================
class AverageMeter(object):
    """ Computes ans stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_scores(score_list, save_dir):
    x = np.array(score_list)
    ep = x[:, 0]
    IS = x[:, 1]
    IS_std = x[:, 2]
    FID = x[:, 3]
    fig, ax = plt.subplots(1, 2, sharex='all', figsize=(12, 4.8))
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    ax[0].set_ylabel('Inception Score')
    ax[1].set_ylabel('FID')
    ax[0].plot(ep, IS, 'r-', linewidth=3)
    ax[0].plot(ep, IS - IS_std, 'r--', linewidth=1)
    ax[0].plot(ep, IS + IS_std, 'r--', linewidth=1)
    ax[1].plot(ep, FID, 'r-', linewidth=3)

    plt.savefig(os.path.join(save_dir, 'score.png'))
    plt.close()
    np.save(os.path.join(save_dir, 'score.npy'), x)


def plot_loss(d_loss, g_loss, num_epoch, epoches, save_dir):
    fig, ax = plt.subplots()
    ax.set_xlim(0, epoches + 1)
    ax.set_ylim(min(0, min(np.min(g_loss), np.min(d_loss))), max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')

    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3)

    plt.legend()
    plt.savefig(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(num_epoch)))
    plt.close()


def plot_loss_my(d_loss, g_loss, g_loss_hist, num_epoch, epoches, save_dir):
    fig, ax = plt.subplots()
    ax.set_xlim(0, epoches + 1)
    ax.set_ylim(min(np.min(g_loss_hist), min(np.min(g_loss), np.min(d_loss))) - 0.1,
                max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')

    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3,
             alpha=0.5)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss_hist, label='Generator - (hist)', color='green', linewidth=3,
             alpha=0.5)

    plt.legend()
    plt.savefig(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(num_epoch)))
    plt.close()


def plot_result(G, fixed_noise, image_size, num_epoch, save_dir, fig_size=(10, 10), is_gray=False, n_side=10):
    G.eval()
    generate_images = G(fixed_noise)
    G.train()

    n_rows = n_cols = n_side
    fig_size = (n_side, n_side)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(image_size, image_size).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
                np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')

    plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(num_epoch)))
    plt.close()


def print_log_2(epoch, epoches, iteration, iters, learning_rate,
                display, batch_time, data_time, D_losses, G_losses, G_n_losses):
    print('epoch: [{}/{}] iteration: [{}/{}]\t'
          'Learning rate: {}'.format(epoch, epoches, iteration, iters, learning_rate))
    print('Time {batch_time.sum:.3f}s / {0} iters, ({batch_time.avg:.3f})\t'
          'Data load {data_time.sum:.3f}s / {0} iters, ({data_time.avg:3f})\n'
          'Loss_D = {loss_D.val:.8f} (ave = {loss_D.avg:.8f})\n'
          'Loss_G = {loss_G.val:.8f} (ave = {loss_G.avg:.8f})\n'
          'Loss_GN = {loss_GN.val:.8f} (ave = {loss_GN.avg:.8f})\n'.format(
        display, batch_time=batch_time,
        data_time=data_time, loss_D=D_losses, loss_G=G_losses, loss_GN=G_n_losses))
    print(time.strftime('%Y-%m-%d %H:%M:%S '
                        '-----------------------------------------------------------------------------------------------------------------\n',
                        time.localtime()))


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')

def grad_info(parameters):
    total_norm = 0
    total_abs_max = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        vmax = p.grad.data.abs().max().item()
        if vmax > total_abs_max:
            total_abs_max = vmax
    total_norm = total_norm ** (1. / 2)
    return total_norm, total_abs_max