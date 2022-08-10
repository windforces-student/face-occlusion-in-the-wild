from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

from imbalanced import ImbalancedDatasetSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.Resize((112, 112)),
        trans.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        trans.Grayscale(num_output_channels=3),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    print(f"{imgs_folder} has been loaded, {len(ds)}")
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_val_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.Resize((112, 112)),
        trans.Grayscale(num_output_channels=3),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(conf):
    if conf.data_mode in ['facebank']:
        ds, class_num = get_train_dataset(conf.facebank_folder / 'train_v2')
        print('facebank loader generated')
    loader = DataLoader(ds,
                        sampler=ImbalancedDatasetSampler(ds),
                        # sampler=None,
                        batch_size=conf.batch_size,
                        shuffle=False,
                        pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers)
    return loader, class_num


def get_val_data(conf):
    if conf.data_mode in ['facebank']:
        ds, class_num = get_val_dataset(conf.facebank_folder / 'test')
        print('facebank loader generated')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers)
    return loader, class_num
