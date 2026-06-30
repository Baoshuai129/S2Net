import time
import argparse

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils.utils2 import *
from model_gpu3 import Net
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc
import pywt
import math
import torch.nn.utils as utils


# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument("--angRes_in", type=int, default=2, help="input angular resolution")
    parser.add_argument("--angRes_out", type=int, default=7, help="output angular resolution")
    # parser.add_argument('--trainset_dir', type=str, default='../Data/TrainData_Lytro_2x2-8x8_extro2/')
    # parser.add_argument('--testset_dir', type=str, default='../Data/TestData_RE_Lytro_2x2-8x8_extro2/')
    # parser.add_argument('--model_name', type=str, default='TransFreq_Lytro_2x2-8x8_extro2')
    # parser.add_argument('--model_name', type=str, default='TransFreq_HCI7_2x2-8x8_extro0')
    parser.add_argument('--trainset_dir', type=str, default='../Data/TrainData_HCI_2x2-7x7/')
    parser.add_argument('--testset_dir', type=str, default='../Data/TestData_RE_HCI_2x2-7x7/')
    parser.add_argument('--model_name', type=str, default='TransFreq_gpu3_model3_HCI_2x2-7x7')
    # parser.add_argument('--trainset_dir', type=str, default='../Data/TrainData_Lytro_2x2-7x7/')
    # parser.add_argument('--testset_dir', type=str, default='../Data/TestData_RE_Lytro_2x2-7x7/')
    # parser.add_argument('--model_name', type=str, default='5weight_Lytro_2x2-7x7')
    parser.add_argument('--log_dir', type=str, default='./logtxt/')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches for validation")
    parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches for validation")
    parser.add_argument("--stride", type=int, default=64, help="LFs are cropped into patches for validation")

    parser.add_argument('--load_pretrain', type=bool, default=False)  # 加载预训练模型
    parser.add_argument('--model_path', type=str, default='./log/TransFreq_Lytro_2x2-8x8_extro2.pth.tar')
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()


def train(cfg, train_loader, test_Names, test_loaders):
    if cfg.parallel:
        cfg.device = 'cuda:3'

    net = Net(cfg.angRes_in, cfg.angRes_out)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:3': cfg.device})
            net.load_state_dict(model['state_dict'])  # 预训练的参数权重加载
            epoch_state = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[3])  # GPU并行处理

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)  # 优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps,
                                                gamma=cfg.gamma)  # 每隔setp_size 将学习率乘以gamma替换
    scheduler._step_count = epoch_state  # 参数更新
    loss_epoch = []
    loss_list = []
    i = 0
    k = 0
    psnr_avg_all = []
    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):  # 进度条
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out = net(data)  # 前向传播计算预测值

            # epi_loss1 = epi_loss(out, label)
            out_numpy = out.cpu().detach().numpy()
            label_numpy = label.cpu().detach().numpy()
            coeffs_out = pywt.wavedec2(out_numpy[1][0], 'haar', level=1)
            coeffs_label = pywt.wavedec2(label_numpy[1][0], 'haar', level=1)
            # 显示小波系数的图像表示
            LL, (LH, HL, HH) = coeffs_out
            LL_label, (LH_label, HL_label, HH_label) = coeffs_label
            loss_ll = criterion_Loss(torch.from_numpy(LL), torch.from_numpy(LL_label))
            # loss_lh = criterion_Loss(torch.from_numpy(LL), torch.from_numpy(LL_label))
            loss_hl = criterion_Loss(torch.from_numpy(HL), torch.from_numpy(HL_label))
            # loss_hh = criterion_Loss(torch.from_numpy(HH), torch.from_numpy(HH_label))

            loss = criterion_Loss(out, label) + 0.4 * loss_ll + 0.4 * loss_hl  # 计算损失

            # loss = criterion_Loss(out, label) + epi_loss1
            optimizer.zero_grad()  # 将模型的参数梯度初始化为0
            loss.backward()  # 反向传播计算梯度
            # === 梯度裁剪 ===
            utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()  # 更新所有参数
            loss_epoch.append(loss.data.cpu())

            i = i + 1
            writer.add_scalar('loss11', float(loss), i)

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile = open('./epochlog/gpu3.txt', 'a')
            txtfile.write('\n' + time.ctime()[4:-5] + '\n Epoch----%5d, loss---%f' % (
            idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile.close()

            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + '.pth.tar')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + '.pth.tar')

            loss_epoch = []

        ''' evaluation '''

        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []

            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Dataset----%15s, PSNR---%f, SSIM---%f' % (
                test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile = open('./epochlog/gpu3.txt', 'a')
                txtfile.write(
                    '\n Dataset----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile.close()
                k = k + 1
                writer.add_scalar('psnr', float(psnr_epoch_test), k)
                writer.add_scalar('ssim', float(ssim_epoch_test), k)

                pass
            pass
            psnr_avg_all.append(sum(psnr_testset) / len(psnr_testset))
        best_psnr = max(psnr_avg_all)
        if best_psnr == psnr_avg_all[-1]:
            print('Best PSNR: %f' % best_psnr)
            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='./log/', filename='bestmodel_' + cfg.model_name + '.pth.tar')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./log/', filename='bestmodel_' + cfg.model_name + '.pth.tar')
        gc.collect()
        torch.cuda.empty_cache()
        scheduler.step()
        pass


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []

    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze().to(cfg.device)

        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
                outLF = outLF.squeeze()
        else:
            uh, vw = data.shape
            h0, w0 = uh // cfg.angRes_in, vw // cfg.angRes_in
            subLFin = LFdivide(data, cfg.angRes_in, cfg.patchsize, cfg.patchsize // 2)  # numU, numV, h*angRes, w*angRes
            numU, numV, H, W = subLFin.shape
            subLFout = torch.zeros(numU, numV, cfg.angRes_out * cfg.patchsize, cfg.angRes_out * cfg.patchsize)

            for u in range(numU):
                for v in range(numV):
                    tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        out = net(tmp.to(cfg.device))
                        # loss = criterion_Loss(out, label)
                        # loss_valid.append(loss.data.cpu())
                        subLFout[u, v, :, :] = out.squeeze()
            outLF = LFintegrate(subLFout, cfg.angRes_out, cfg.patchsize, cfg.stride, h0, w0)

        # psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out)
        # psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out)
        metrics_path = cfg.save_path + '/' + cfg.model_name + '/' + test_loader.dataset.file_list[
                                                                        idx_iter][0:-3]
        if not (os.path.exists(metrics_path)):
            os.makedirs(metrics_path)
        psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out, metrics_path,
                                    test_loader.dataset.file_list[idx_iter][0:-3])

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def main(cfg):
    seed = 10000

    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也需要设置这个
    torch.backends.cudnn.deterministic = True  # 如果使用了CUDA，这个也需要设置
    torch.backends.cudnn.benchmark = False

    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)


if __name__ == '__main__':
    writer: SummaryWriter = SummaryWriter()
    print("1111")
    cfg = parse_args()

    # 创建一个文件对象

    main(cfg)
    writer.close()


