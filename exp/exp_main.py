from exp.exp_basic import Exp_Basic

import torch.nn as nn


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if self.args.data == 'WindGraph':
            self.args.seq_len = self.args.label_len

    def _build_model(self):
        model_dict = {
            #'Autoformer': Autoformer,
            #'Transformer': Transformer,
            #'Informer': Informer,
            #'LogSparse': LogSparseTransformer,
            #'FFTransformer': FFTransformer,
            #'LSTM': LSTM,
            #'MLP': MLP,
            #'persistence': persistence,
            #'GraphTransformer': GraphTransformer,
            #'GraphLSTM': GraphLSTM,
            #'GraphFFTransformer': GraphFFTransformer,
            #'GraphInformer': GraphInformer,
            #'GraphLogSparse': GraphLogSparse,
            #'GraphMLP': GraphMLP,
            #'GraphAutoformer': GraphAutoformer,
            #'GraphPersistence': GraphPersistence,
        }

        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            #if self.args.data == 'WindGraph':
            #    model = DataParallelGraph(model, device_ids=self.args.device_ids)
            #else:
            #    model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model
    
