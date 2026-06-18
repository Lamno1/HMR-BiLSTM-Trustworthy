import torch
import torchattacks

class Mock(torch.nn.Module):
    def forward(self, x):
        print('X shape:', x.shape)
        return torch.zeros(x.shape[0], 5).cuda()

m = Mock().cuda()
atk = torchattacks.AutoAttack(m, version='standard', n_classes=5)
bx = torch.randn(128, 1, 187, 1).cuda()
by = torch.zeros(128).long().cuda()
atk.set_device('cuda')
atk(bx, by)
