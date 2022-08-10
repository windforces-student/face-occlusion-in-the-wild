import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch import (
    optim,
    nn
)
from tqdm import tqdm

from data.data_pipe import get_train_loader, get_val_data
from model import MobileFaceNet

plt.switch_backend('agg')
from utils import get_time, gen_plot, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
from sklearn.metrics import (
    roc_curve,
    accuracy_score,

)


def save_state(model, conf, accuracy, to_save_folder=False, extra=None):
    if to_save_folder:
        save_path = conf.save_path
    else:
        save_path = conf.model_path
    torch.save(model.state_dict(), save_path /
               ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, model.step,
                                                             extra)))


class face_learner(nn.Module):
    def __init__(self, conf, inference=False):
        super(face_learner, self).__init__()
        print(conf)
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')

        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)

            self.writer = SummaryWriter(str(conf.log_path))
            self.step = 0
            # self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            self.head = nn.Sequential(
                nn.Linear(in_features=conf.embedding_size, out_features=conf.embedding_size),
                nn.ReLU(),
                nn.Linear(in_features=conf.embedding_size, out_features=self.class_num),
                nn.Dropout()
            ).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                self.optimizer = optim.Adam([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head[0].weight, self.head[2].weight], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head[0].weight, self.head[2].weight], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
            #             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')
            self.board_loss_every = len(self.loader) // 50
            self.evaluate_every = len(self.loader) // 50
            self.save_every = len(self.loader) // 10
            self.val_loader, self.val_class_num = get_val_data(conf)
        else:
            self.threshold = conf.threshold

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    def evaluate(self, conf, val_loader):
        self.model.eval()
        self.head.eval()
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(conf.device), label.to(conf.device)
                # label = label - 1
                embeddings = self.model(image)
                outputs = self.head(embeddings)
                _, predicted = torch.max(outputs, 1)
                # print(label, predicted)
                all_y_true, all_y_pred = np.append(all_y_true, label.cpu().numpy()), np.append(all_y_pred,
                                                                                               predicted.cpu().numpy())

        accuracy = self.calculate_metrics(all_y_true, all_y_pred)
        return accuracy

    def forward(self, image):
        embeddings = self.model(image)
        outputs = self.head(embeddings)
        return outputs

    def test(self, conf, val_loader):
        self.model.eval()
        self.head.eval()
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(conf.device), label.to(conf.device)

                embeddings = self.model(image)
                outputs = self.head(embeddings)
                _, predicted = torch.max(outputs, 1)

                all_y_true, all_y_pred = np.append(all_y_true, label.cpu().numpy()), np.append(all_y_pred,
                                                                                               predicted.cpu().numpy())
        return all_y_true, all_y_pred

    def calculate_metrics(self, y_true, y_pred):
        y_pred = y_pred.astype(np.uint8)
        y_true = y_true.astype(np.uint8)
        ## Score
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy

    def experiment(self, conf, epochs):
        self.model.train()
        self.head.train()
        running_loss, loss_board, accuracy = 0., 0., 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            pbar = tqdm(iter(self.loader))
            for imgs, labels in pbar:
                imgs, labels = imgs.to(conf.device), labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                preds = self.head(embeddings)
                loss = conf.ce_loss(preds, labels)
                # loss = conf.focal_loss(preds, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    pbar.set_postfix({'train_loss': loss_board, 'accuracy': accuracy})
                    running_loss = 0.
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy = self.evaluate(conf, self.val_loader)
                    # self.board_val('facebank', accuracy, roc_curve_tensor)
                    pbar.set_postfix({'train_loss': loss_board, 'accuracy': accuracy})
                # if self.step % self.save_every == 0 and self.step != 0:
                #     save_state(self, conf, accuracy)

                self.step += 1

        save_state(self, conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)
