import scipy.io
import scipy.misc
from torch.autograd import Variable
import argparse
from torch.backends import cudnn
from utils.utils2 import *
from model_gpu3 import Net
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes_in", type=int, default=2, help="input angular resolution")
    parser.add_argument("--angRes_out", type=int, default=7, help="output angular resolution")
    # parser.add_argument("--model_name", type=str, default='5weight_Lytro_2x2-7x7')
    # parser.add_argument("--testset_dir", type=str, default='../Data/TestData_RE_Lytro_2x2-7x7/')

    parser.add_argument("--model_name", type=str, default='weight_HCI_2x2-7x7')
    parser.add_argument("--testset_dir", type=str, default='../Data1/TestData_RE_HCI_2x2-7x7/')

    parser.add_argument('--crop', type=bool, default= True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()

def test(cfg):

    test_Names, test_loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    net = Net(cfg.angRes_in, cfg.angRes_out)
    net.to(cfg.device)
    cudnn.benchmark = True
    model = torch.load('./log/' + cfg.model_name + '.pth.tar', map_location={'cuda:3': cfg.device}, weights_only= True)
    net.load_state_dict(model['state_dict'])

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = valid(test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print('Dataset----%10s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile = open(cfg.save_path + cfg.model_name + '_metrics.txt', 'a')
            txtfile.write('Dataset----%10s,\t PSNR---%f,\t SSIM---%f\n' % (test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile.close()
            pass
        pass


def valid(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []

    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze().to(cfg.device)
        if cfg.crop == False:
            with torch.no_grad():
                data1 = data.unsqueeze(0).unsqueeze(0).to(cfg.device)
                outLF = net(data1)
                outLF = outLF.squeeze()
        else:
            patchsize = cfg.patchsize
            stride = patchsize // 2
            uh, vw = data.shape
            h0, w0 = uh // cfg.angRes_in, vw // cfg.angRes_in
            subLFin = LFdivide(data, cfg.angRes_in, patchsize, stride)  # numU, numV, h*angRes, w*angRes
            numU, numV, H, W = subLFin.shape
            subLFout = torch.zeros(numU, numV, cfg.angRes_out * patchsize, cfg.angRes_out * patchsize)
            for u in range(numU):
                for v in range(numV):
                    tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        tmp1 = tmp.to(cfg.device)
                        out = net(tmp1)
                        subLFout[u, v, :, :] = out.squeeze()

            outLF = LFintegrate(subLFout, cfg.angRes_out, patchsize, stride, h0, w0)

        metrics_path = cfg.save_path + '/' + cfg.model_name + '/' +  test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3]
        if not (os.path.exists(metrics_path)):
            os.makedirs(metrics_path)
        psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out, metrics_path,
                                    test_loader.dataset.file_list[idx_iter][0:-3])

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        total_metrics_path = cfg.save_path + '/' + cfg.model_name + '/' + test_name + '/'
        print('fileName: ----%10s, PSNR---%f, SSIM---%f' % (test_loader.dataset.file_list[idx_iter][0:-3], psnr, ssim))
        txtfile_0 = open(total_metrics_path + 'ablation_' + 'metrics.txt', 'a')
        txtfile_0.write( 'fileName: ----%10s, PSNR---%f, SSIM---%f\n' % (test_loader.dataset.file_list[idx_iter][0:-3], psnr, ssim))
        txtfile_0.write('------------------------------------------\n')
        txtfile_0.close()

        txtfile_1 = open(metrics_path + '/' + str(test_loader.dataset.file_list[idx_iter][0:-3]) + '_metrics.txt', 'a')
        txtfile_1.write('total_PSNR---%f, total_SSIM---%f\n' % (psnr, ssim))
        txtfile_1.write('------------------------------------------\n')
        txtfile_1.close()

        if not (os.path.exists(cfg.save_path + '/' + test_name)):
            os.makedirs(cfg.save_path + '/' + test_name)
        scipy.io.savemat(cfg.save_path + '/' + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat', {'LF': outLF.numpy()})


        pass


    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    cfg = parse_args()
    test(cfg)