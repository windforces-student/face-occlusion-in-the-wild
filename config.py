from pathlib import Path

import torch

from easydict import EasyDict as edict
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
from loss import FocalLoss


def get_config(training=True):
    conf = edict()
    conf.data_path = Path('data')
    conf.work_path = Path('work_space')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    conf.data_mode = 'facebank'
    conf.vgg_folder = conf.data_path / 'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path / 'faces_emore'
    conf.facebank_folder = conf.data_path / 'facebank'
    conf.batch_size = 256  # irse net depth 50
    #   conf.batch_size = 200 # mobilefacenet
    # --------------------Training Config ------------------------
    if training:
        #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [0, 1, 2]
        conf.momentum = 0.9
        conf.pin_memory = True
        #         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()
        conf.focal_loss = FocalLoss(alpha=1.0, gamma=4.0).to(conf.device)
    # --------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path / 'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
