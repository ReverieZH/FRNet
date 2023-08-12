
import os


efficientnet_backbone_path = './backbone/efficientnet/adv-efficientnet-b4-44fb3a87.pth'


datasets_root = './data/COD/'
cod_training_root = './data/COD/TrainDataset'

chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
camo_path = os.path.join(datasets_root, 'test/CAMO')
cod10k_path = os.path.join(datasets_root, 'test/COD10K')
nc4k_path = os.path.join(datasets_root, 'test/NC4K')

fossil_train_root = './data/Fossil_3k/train'
fossil_val_root = './data/Fossil_3k/val'
