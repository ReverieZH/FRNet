import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
from tqdm import tqdm
import torch_dct as DCT
from config import *
from misc import *
from FRNet import FRNet

torch.manual_seed(2022)
device_ids = [0]
torch.cuda.set_device(device_ids[0])


exp_name = 'FRNet'   # ckpt的文件夹名称
net_path = os.path.join('./ckpt', exp_name, 'model.pth')


results_path = './results'
save_name = 'FRNet'  # 保存路径名称
save_dir = os.path.join(results_path, save_name)
check_mkdir(save_dir)

args = {
    'scale': 416,
    'save_results': True,
    'save_middle_results': False
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

freq_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
                       ('CHAMELEON', chameleon_path),
                       ('CAMO', camo_path),
                       ('COD10K', cod10k_path),
                       ('NC4K', nc4k_path)
                       ])

results = OrderedDict()

def main():
    net = FRNet(efficientnet_backbone_path, backbone="EfficientNet").cuda(device_ids[0]).train()
    tmp = torch.load(net_path)
    state = tmp['state']
    epoch = tmp['state']
    mad = tmp['mad']
    net.load_state_dict(state)
    print('Load {} succeed!'.format('PFNet.pth'))
    print('best model epoch: ', epoch)
    print('best model mad: ', mad)

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')
            mask_path = os.path.join(root, 'mask')
            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in  enumerate(tqdm(img_list)):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
                mask = Image.open(os.path.join(mask_path, img_name + '.png')).convert('L')
                ycbcr_image = Image.open(os.path.join(image_path, img_name + '.jpg')).convert("YCbCr")

                w, h = img.size

                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
                batch_size = img_var.size(0)
                size = img_var.size(2)
                ycbcr_img_var = Variable(freq_transform(ycbcr_image).unsqueeze(0)).cuda(device_ids[0])
                ycbcr_img_var = ycbcr_img_var.reshape(batch_size, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)

                ycbcr_img_var = DCT.dct_2d(ycbcr_img_var, norm='ortho')
                ycbcr_img_var = ycbcr_img_var.reshape(batch_size, size // 8, size // 8, -1).permute(0, 3, 1, 2)
                start_each = time.time()

                predict_1,  predict2, predict3, predict4, prediction, freq_output_2, edge_predict = net(img_var, ycbcr_img_var)

                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                predict4 = np.array(transforms.Resize((h, w))(to_pil(predict4.data.squeeze(0).cpu())))
                predict3 = np.array(transforms.Resize((h, w))(to_pil(predict3.data.squeeze(0).cpu())))
                predict2 = np.array(transforms.Resize((h, w))(to_pil(predict2.data.squeeze(0).cpu())))
                predict1 = np.array(transforms.Resize((h, w))(to_pil(predict_1.data.squeeze(0).cpu())))

                freq_output_2 = np.array(transforms.Resize((h, w))(to_pil(freq_output_2.data.squeeze(0).cpu())))
                edge_predict = np.array(transforms.Resize((h, w))(to_pil(edge_predict.data.squeeze(0).cpu())))

                test_dataset_save_dir = os.path.join(save_dir, name)
                check_mkdir(test_dataset_save_dir)
                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(
                        os.path.join(test_dataset_save_dir, img_name + '.png'))
                if args['save_middle_results']:
                    img.save(os.path.join(test_dataset_save_dir, img_name + '_img.png'))
                    mask.save(os.path.join(test_dataset_save_dir, img_name + '_gt.png'))
                    Image.fromarray(predict4).convert('L').save(
                        os.path.join(test_dataset_save_dir, img_name + '_predict4.png'))
                    Image.fromarray(predict3).convert('L').save(
                        os.path.join(test_dataset_save_dir, img_name + '_predict3.png'))
                    Image.fromarray(predict2).convert('L').save(
                        os.path.join(test_dataset_save_dir, img_name + '_predict2.png'))
                    Image.fromarray(predict1).convert('L').save(
                        os.path.join(test_dataset_save_dir, img_name + '_predict1.png'))
                    Image.fromarray(freq_output_2).convert('L').save(
                        os.path.join(test_dataset_save_dir, img_name + '_freq_predict2.png'))
                    Image.fromarray(edge_predict).convert('L').save(
                        os.path.join(test_dataset_save_dir, img_name + '_edge.png'))
            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()