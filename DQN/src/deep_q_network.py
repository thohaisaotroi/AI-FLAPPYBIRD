import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.tangTichChap1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.tangTichChap2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.tangTichChap3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))

        self.tangKetNoi1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.tangRa = nn.Linear(512, 2)
        self.khoiTaoTrongSo()

    def khoiTaoTrongSo(self):
        for lop in self.modules():
            if isinstance(lop, nn.Conv2d) or isinstance(lop, nn.Linear):
                nn.init.uniform_(lop.weight, -0.01, 0.01)
                nn.init.constant_(lop.bias, 0)

    def tienTrinh(self, dauVao):
        dauRa = self.tangTichChap1(dauVao)
        dauRa = self.tangTichChap2(dauRa)
        dauRa = self.tangTichChap3(dauRa)
        dauRa = dauRa.view(dauRa.size(0), -1)
        dauRa = self.tangKetNoi1(dauRa)
        dauRa = self.tangRa(dauRa)
        return dauRa
