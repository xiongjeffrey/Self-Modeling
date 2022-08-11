# I'm Dumb and I Give Up GRRRR

import numpy as np
import torch
import torch.nn as nn

class NaivePWM(nn.Module):

    def __init__(self, init_state, entity_dir, grid_size = 5) -> None:
        
        self.state = init_state

        self.DIR_TO_VEC = [
            np.array((1, 0)), # Pointing right (positive X) : 0
            np.array((1,1)), # Downright (positive x, positive y)
            np.array((0, 1)), # Down (positive Y) : 2
            np.array((-1,1)), # Downleft (negative x, positive y)
            np.array((-1, 0)), # Pointing left (negative X) : 4 
            np.array((-1,-1)), # Upleft (negative x, negative y)
            np.array((0, -1)), # Up (negative Y) : 6 
            np.array((1,-1)), # Upright (positive x, negative y)
        ]

        if not isinstance(entity_dir, np.ndarray):
            self.dir   = self.DIR_TO_VEC[entity_dir]
        self.entity_dir = entity_dir

        self.grid_size = grid_size

    def forward(self, pos):

        ogx = pos[0]
        ogy = pos[1]

        x = (ogx + self.dir[0])
        y = (ogy + self.dir[1])
        new_x = x % self.grid_size
        new_y = y % self.grid_size
        if self.entity_dir == 1:
            if ogx == self.grid_size - 1:
                new_y = new_x
                new_x = abs(ogx-ogy)
            elif ogy == self.grid_size - 1:
                new_x = new_y
                new_y = abs(ogx-ogy)
        elif self.entity_dir == 3:
            if new_x == 0 or new_y == 0:
                tempx = ogx
                new_x = ogy
                new_y = tempx
        elif self.entity_dir == 5:
            if ogx == 0:
                new_y = new_x
                new_x = (self.grid_size-1)-abs(ogx-ogy)
            elif ogy == 0:
                new_x = new_y
                new_y =  (self.grid_size-1)-abs(ogx-ogy)
        elif self.entity_dir == 7:
            if new_x == 0 or new_y == self.grid_size - 1:
                tempx = ogx
                new_x = ogy
                new_y = tempx

        self.state = torch.Tensor([new_x, new_y])

        return self.state