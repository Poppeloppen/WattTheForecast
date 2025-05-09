import os
import torch
import argparse

class Exp_Basic(object):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _build_model(self) -> None:
        raise NotImplementedError
    
    def _acquire_device(self) -> torch.device:
        if self.args.use_mps:
            os.environ["MPS_VISIBLE_DEVICES"] = str(self.args.mps)
            device = torch.device("mps")
            print('Use MPS: mps:{}'.format(self.args.mps))
            
        elif self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass




def main():
    exp = Exp_Basic()
    return


if __name__ == "__main__":
    main()