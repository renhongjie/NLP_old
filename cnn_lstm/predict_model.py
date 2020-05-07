import torch
import torch.nn.functional as F
from NLP1.cnn_lstm.model import Model as _Model
from NLP1.cnn_lstm.model import ModelParam as _ModelParam
from NLP1.cnn_lstm.dataset import DataTransform as _Transform
import os


class ModelPredict(object):
    def __init__(self, model_file, vocab_file, tokenizer, cuda=False, use_openmp=False):
        assert tokenizer
        if not os.path.exists(model_file) or not os.path.exists(vocab_file):
            raise Exception("model file or vocab file not exist")
        self._model_file = model_file
        self.cuda = cuda and torch.cuda.is_available()
        self.model = self._load_model(model_file, )
        self.transform = _Transform(vocab_file, tokenizer)

        if self.cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.eval()

    @classmethod
    def _load_model(cls, model_file, cuda=False) -> _Model:
        map_location = "cpu" if not cuda else None
        state = torch.load(model_file, map_location=map_location)
        _model = _Model(_ModelParam.from_dict(state["model_args"]))
        _model.load_state_dict(state["state_dict"])
        del state
        return _model

    def get_model_file(self):
        return self._model_file

    def get_max_sentence_len(self):
        return self.transform.max_sent_len

    # return: [标签, 概率]
    def predict(self, text: str) -> list:
        res_list = self.predict_batch([text])
        return res_list[0]

    # return: [[标签，概率], [标签，概率], ...]
    def predict_by_splitext(self, text: str) -> list:
        size = len(text)
        sent_len = self.get_max_sentence_len()
        batch_size = size // sent_len
        if size % sent_len != 0:
            batch_size += 1

        text_list = []
        for i in range(0, batch_size):
            offset = i * sent_len
            subtext = text[offset: offset + sent_len]
            text_list.append(subtext)

        return self.predict_batch(text_list)

    # return: [[标签，概率], [标签，概率], ...]
    def predict_batch(self, text_list: list) -> list:
        batch_size = len(text_list)
        if batch_size == 0:
            return []

        # 中间计算不需要梯度，节省内存
        with torch.no_grad():
            ids_list = []
            for text in text_list:
                ids_list.append(self.transform.text2ids(text, pad_empty=True))
            ids_list = torch.LongTensor(ids_list)
            if self.cuda:
                ids_list = ids_list.cuda()
            output = self.model(ids_list)
            output = F.softmax(output, dim=1)
            probs, labels = torch.max(output, 1)
            res = []
            for i in range(batch_size):
                res.append([int(labels[i]), float(probs[i])])
            return res
