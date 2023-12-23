
import torch
from torch import nn
import numpy as np

class Bezier(nn.Module):
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Bezier, self).__init__()
        self.x1 = nn.Parameter(torch.as_tensor(ctps[0], dtype=torch.float64))
        self.x2 = nn.Parameter(torch.as_tensor(ctps[2], dtype=torch.float64))
        self.y1 = nn.Parameter(torch.as_tensor(ctps[1], dtype=torch.float64))
        self.y2 = nn.Parameter(torch.as_tensor(ctps[3], dtype=torch.float64))
        self.x0 = ps[0, 0]
        self.x3 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y3 = ps[-1, 1]
        self.inner_ps = torch.as_tensor(ps[1:-1, :], dtype=torch.float64)
        self.t = torch.as_tensor(np.linspace(0, 1, 81))

    def forward(self):
        x0, x1, x2, x3, y0, y1, y2, y3 = self.control_points()
        t = self.t
        bezier_x = (1-t)*((1-t)*((1-t)*x0+t*x1)+t*((1-t)*x1+t*x2))+t*((1-t)*((1-t)*x1+t*x2)+t*((1-t)*x2+t*x3))
        bezier_y = (1-t)*((1-t)*((1-t)*y0+t*y1)+t*((1-t)*y1+t*y2))+t*((1-t)*((1-t)*y1+t*y2)+t*((1-t)*y2+t*y3))
        bezier = torch.stack((bezier_x, bezier_y), dim=1)
        diffs = bezier.unsqueeze(0) - self.inner_ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.y0, self.y1, self.y2, self.y3

    def control_points_f(self):
        return self.x0, self.x1.item(), self.x2.item(), self.x3, self.y0, self.y1.item(), self.y2.item(), self.y3
