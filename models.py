import torch.nn as nn
import torch

def conv_block():
    pass

def encoder():
    pass

def decoder():
    pass

def UNet():
    encoder()

    conv_block()

    decoder()

# TODO: patch size needs to be divisible by patch size(256/4=64)
class PatchEmbed(nn.Module): #patch partition and linear embedding
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.split = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.split(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        x = self.norm(x)

        return x
        
def window_partition(x, win):
    B, H, W, C = x.shape
    x = x.view(B, H//win, win, W//win, win, C)
    x = x.permute(0,1,3,2,4,5).reshape(-1, win, win, C) # -1 = B * H//win *  W//win
    return x

def window_reverse(windows, win, H, W):
    B = windows[0]/(H//win * W//win) #divide to get the number of batches
    x = windows.view(B, H//win, W//win, win, win, -1)
    x = x.permute(0,1,3,2,4,5)
    x = x.reshape(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    def __init__(self, dim, num_heads, win):
        super().__init__()
        self.dim = dim # embedding dimension
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # dimension per head
        self.scale = self.head_dim**-0.5 # 1/sqrt(dim)
        self.win = win

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim,dim) #projection layer to concat all items coming from different heads


        #relative positional bias
        #total query-key value pairs are win*win**2
        coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing = 'ij'))
        coords_flat = coords.flatten(1)
        rel = coords_flat[:,:, None] - coords_flat[:, None, :]
        rel = rel.permute(1,2,0)

        rel[:,:,0] = rel[:,:,0] + (win-1) # range starting from 0 instead of negative
        rel[:,:,1] = rel[:,:,1] + (win-1)

        rel[:,:,0] = rel[:,:,0] * (2*win - 1)
        index = rel.sum(-1)

        self.register_buffer("pos_index", index)
        self.rel_bias = nn.Parameter(torch.zeros((2*win-1) * (2*win-1), num_heads))

    def forward(self, x, mask=None):

        B_, N, C =x.shape
        q = self.q(x)
        v = self.v(x)
        k = self.k(x)

        q = q*self.scale
        attn = q @ k.transpose(-2,-1)

        rb = self.rel_bias[self.pos_index.view(-1)].view(N,N,-1)
        attn = attn + rb.permute(2,0,1).unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_//nw, nw, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B_,N,C)
        out = self.proj(out)
        return out

class SwinTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(4*dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x0 = x0.reshape(B, -1, C)
        x1 = x1.reshape(B, -1, C)
        x2 = x2.reshape(B, -1, C)
        x3 = x3.reshape(B, -1, C)

        x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)

        return x, H//2, W//2