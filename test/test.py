# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torch.nn.functional as F
from julia.api import Julia
jl = Julia(compiled_modules = False)
from julia import Main
from project import project, backproject
import numpy as np
np.random.seed(0)
jl.eval('include("SPECTrecon_julia.jl")')

def gen_data(nx, ny, nz, nview, px, pz):
    img = np.zeros([nx, ny, nz])
    img[1:-1,1:-1,1:-1] = np.random.rand(nx-2,ny-2,nz-2)
    mumap = np.zeros([nx, ny, nz])
    mumap[1:-1, 1:-1, 1:-1] = np.random.rand(nx - 2, ny - 2, nz - 2)
    psfs = np.ones([px, pz, ny, nview]) / (px * pz)
    return img, mumap, psfs


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nx = 8
    ny = 8
    nz = 6
    nview = 7
    px = 3
    pz = 3
    dy = 2
    img, mumap, psfs = gen_data(nx, ny, nz, nview, px, pz)
    Main.img = img
    Main.mumap = mumap
    Main.psfs = psfs
    Main.dy = dy
    view1 = jl.eval("gen_project(img, mumap, psfs, dy)")
    view2 = project(torch.from_numpy(img),
                    torch.from_numpy(mumap),
                    torch.from_numpy(psfs), dy).detach().cpu().numpy()
    print('forward nrmse:', np.linalg.norm(view1-view2) / np.linalg.norm(view1)) # 8.6e-8
    views = np.zeros([nx, nz, nview])
    views[1:-1,1:-1,:] = np.random.rand(nx-2, nz-2, nview)
    Main.views = views
    img1 = jl.eval("gen_backproject(views, mumap, psfs, dy)")
    img2 = backproject(torch.from_numpy(views),
                       torch.from_numpy(mumap),
                       torch.from_numpy(psfs), dy).detach().cpu().numpy()
    print('backward nrmse: ', np.linalg.norm(img1-img2) / np.linalg.norm(img1)) # 0.075
    x1 = np.zeros([nx, ny, nz])
    x1[1:-1,1:-1,1:-1] = np.random.rand(nx-2,ny-2,nz-2)
    y1 = np.zeros([nx, nz, nview])
    y1[1:-1,1:-1,:] = np.random.rand(nx-2,nz-2,nview)
    xout = project(torch.from_numpy(x1),
                   torch.from_numpy(mumap),
                   torch.from_numpy(psfs), dy).detach().cpu().numpy()
    yout = backproject(torch.from_numpy(y1),
                       torch.from_numpy(mumap),
                       torch.from_numpy(psfs), dy).detach().cpu().numpy()
    print('<xout, y1>:', np.dot(xout.reshape(-1), y1.reshape(-1))) # 33.35
    print('<yout, x1>:', np.dot(yout.reshape(-1), x1.reshape(-1))) # 33.41

    # test backpropagation on GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net = net.to(device)
    x = torch.rand(nx, ny, nz).unsqueeze(0).unsqueeze(0).to(device)
    x1 = net.forward(x)
    x2 = project(x1.squeeze(), torch.from_numpy(mumap).to(device), torch.from_numpy(psfs).to(device), dy)
    x3 = backproject(x2, torch.from_numpy(mumap).to(device), torch.from_numpy(psfs).to(device), dy)
    criterion = nn.MSELoss()
    loss = criterion(x3, torch.zeros_like(x3))
    torch.autograd.set_detect_anomaly(True)
    loss.backward()

