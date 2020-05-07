import torch
import torch.nn.functional as F
import numpy as np
from NLP1.cnn_lstm.model import Model, ModelParam
from NLP1.cnn_lstm.dataset import DatasetIterator
import logging
import os
import time


class TrainParam(object):
    def __init__(self):
        self.learning_rate = 0.001
        self.epoches = 0
        self.cuda = False  # True
        self.log_interval = 3
        self.test_interval = 50
        self.save_interval = 100
        self.train_batch_size = 64
        self.test_rate = 0.1
        self.test_batch_size = 100
        self.model_save_dir = "model_bin"
        self.model_name = "cnnlstmmodel"
        self.continue_train = True


class ModelTrain(object):
    def __init__(self, args: TrainParam, model_args: ModelParam):
        self.args = args
        self.model = Model(model_args)
        if not self._load():
            logging.info("Created model with fresh parameters.")
        # if self.args.cuda:
        #     self.model.cuda()

    def _eval(self, data_iter: DatasetIterator):
        with torch.no_grad():
            # 中间计算不需要梯度，节省显存
            x_batch, y_batch = data_iter.rand_testdata(self.args.test_batch_size)
            x_batch = torch.from_numpy(np.array(x_batch)).long()
            y_batch = torch.from_numpy(np.array(y_batch)).long()
            batch_size = y_batch.size()[0]

            if self.args.cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            logit = self.model(x_batch)
            loss = F.cross_entropy(logit, y_batch)
            correct_num = (torch.max(logit, 1)[1].view(y_batch.size()).long() == y_batch).sum()
            accurary = 100.0 * correct_num / batch_size
            logging.info("\t Test - loss: %.4f  acc:%.4f%% %d/%d" % (loss.item(), accurary, correct_num, batch_size))

    def _load(self) -> bool:
        if not self.args.continue_train:
            return False

        snapshot = os.path.join(self.args.model_save_dir, "snapshot")
        if not os.path.exists(snapshot):
            return False

        checkpoint = os.path.join(self.args.model_save_dir, "checkpoint")
        with open(checkpoint, "r", encoding="utf-8") as rf:
            steps = int(rf.readline().strip())
            fpath = os.path.join(snapshot, "%d.pt" % steps)
            if os.path.exists(fpath):
                self.model.load_state_dict(torch.load(fpath))
                logging.info("Reading model parameters from %s" % fpath)
                return True
        return False

    def _save(self, steps=-1):
        if not os.path.exists(self.args.model_save_dir):
            os.makedirs(self.args.model_save_dir)

        if steps > 0:
            snapshot = os.path.join(self.args.model_save_dir, "snapshot")
            if not os.path.exists(snapshot):
                os.makedirs(snapshot)
            save_path = os.path.join(snapshot, "%d.pt" % steps)
            torch.save(self.model.state_dict(), save_path)

            checkpoint = os.path.join(self.args.model_save_dir, "checkpoint")
            with open(checkpoint, "w", encoding="utf-8") as wf:
                wf.write(str(steps) + "\n")
        else:
            ext = os.path.splitext(self.args.model_name)[1]
            if len(ext) == 0:
                save_path = os.path.join(self.args.model_save_dir, self.args.model_name + ".pt")
            else:
                save_path = os.path.join(self.args.model_save_dir, self.args.model_name)
            # torch.save(self.model, save_path)
            state = {
                "model_args": self.model.get_predict_args(),
                "state_dict": self.model.state_dict()
            }
            torch.save(state, save_path)

    def valid(self, data_iter: DatasetIterator):
        self.model.eval()
        with torch.no_grad():
            # 中间计算不需要梯度，节省显存
            b = time.time()
            correct_num = 0
            batch_num = data_iter.test_num // self.args.test_batch_size
            if data_iter.test_num % self.args.test_batch_size != 0:
                batch_num += 1
            count = 0
            for x_batch, y_batch in data_iter.next_testdata(self.args.test_batch_size):
                count += 1
                logging.info("%.f%%, %d/%d" % (count * 100.0 / batch_num, count, batch_num))
                x_batch = torch.from_numpy(np.array(x_batch)).long()
                y_batch = torch.from_numpy(np.array(y_batch)).long()
                # if self.args.cuda:
                #     x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                logit = self.model(x_batch)
                correct_num += (torch.max(logit, 1)[1].view(y_batch.size()).long() == y_batch).sum()
            acc = 100.0 * correct_num / data_iter.test_num
            logging.info("-----> acc:%.f%%, cost:%f" % (acc, time.time() - b))

    def train(self, data_iter: DatasetIterator):
        b = time.time()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        steps = 0
        self.model.train()
        for epoch in range(1, self.args.epoches + 1):
            batch_count = 0
            for x_batches, y_batches in data_iter.next_batch(self.args.train_batch_size):
                batch_count += 1
                fetures = torch.from_numpy(np.array(x_batches)).long()
                labels = torch.from_numpy(np.array(y_batches)).long()
                # if self.args.cuda:
                #     fetures, labels = fetures.cuda(), labels.cuda()
                batch_size = labels.size()[0]

                optimizer.zero_grad()
                logit = self.model(fetures)

                loss = F.cross_entropy(logit, labels)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % self.args.log_interval == 0:
                    # labels是（batch_size, 1)的矩阵
                    correct_num = (torch.max(logit, 1)[1].view(labels.size()).long() == labels).sum()
                    accuracy = 100.0 * correct_num / batch_size
                    print("epoch-%d step-%d batch-%d - loss: %.4f  acc: %.2f%% %d/%d" % (
                        epoch, steps, batch_count, loss.item(), accuracy, correct_num, batch_size))
                    logging.info("epoch-%d step-%d batch-%d - loss: %.4f  acc: %.2f%% %d/%d" % (
                        epoch, steps, batch_count, loss.item(), accuracy, correct_num, batch_size))

                if steps % self.args.test_interval == 0:
                    self.model.eval()
                    self._eval(data_iter)
                    self.model.train()

                if steps % self.args.save_interval == 0:
                    self._save(steps)
        self._save()
        logging.info("train finished, cost:%f" % (time.time() - b))
