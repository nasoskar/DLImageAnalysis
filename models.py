import torch.nn as nn
import torch

def bottleneck():
    pass
    #swin transformerx1
    #swin transformerx1

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, win):
        super().__init__()
        # create a list of SwinTransformerBlocks
        # depth = number of blocks (usually 2)
        # alternate shift: block i has shift = 0 if i%2==0 else win//2
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                win=win,
                shift=0 if i%2==0 else win//2,
                heads=num_heads
            )
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        for block in self.blocks:
            x = block(x)
        return x

class encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, win, heads, swin_depth):
        super().__init__()
        self.patchembed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        self.layer1 = BasicLayer(dim=embed_dim, swin_depth=swin_depth, heads=heads, win=win)
        self.merge1  = PatchMerging(embed_dim)

        self.layer2 = BasicLayer(dim=embed_dim*2, swin_depth=swin_depth, heads=heads, win=win)
        self.merge2  = PatchMerging(embed_dim*2)

        self.layer3 = BasicLayer(dim=embed_dim*4, swin_depth=swin_depth, heads=heads, win=win)
        self.merge3  = PatchMerging(embed_dim*4)


    def forward(self, x):
        x, H, W = self.patchembed(x)
        
        x = self.layer1(x, H, W)
        skip1 = x
        x, H, W = self.merge1(x, H, W)

        x = self.layer2(x, H, W)
        skip2 = x
        x, H, W  = self.merge2(x, H, W)

        x = self.layer3(x, H, W)
        skip3 = x
        x, H, W = self.merge3(x, H, W)

        return x, H, W, skip1, skip2, skip3

    

class decoder(nn.Module):
    def __init__(self, dim, depth, num_heads, win):
        super().__init__()

        self.exp1 = PatchExpanding(dim*4)
        self.layer1 = BasicLayer(dim*2, depth, num_heads, win)

        self.exp2 = PatchExpanding(dim*2)
        self.layer2 = BasicLayer(dim, depth, num_heads, win)

        self.exp3 = PatchExpanding(dim)
        self.layer3 = BasicLayer(dim//2, depth, num_heads, win)

        self.finalexp = FinalPatchExpand(dim//2, scale=4)

    def forward(self, x, H, W, skip1, skip2, skip3):

        x, H, W = self.exp1(x, H, W)
        x = self.layer1(x + skip3,H,W)

        x, H, W = self.exp2(x, H, W)
        x = self.layer2(x + skip2,H,W)

        x, H, W = self.exp3(x, H, W)
        x = self.layer3(x + skip1,H,W)

        x, H, W = self.finalexp(x, H, W)

        return x, H, W



def UNet():
    encoder()

    bottleneck()

    decoder()

    #linearprojection()
    #skip connections()

# TODO: patch size needs to be divisible by patch size(256/4=64)
class PatchEmbed(nn.Module): #patch partition and linear embedding
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.split = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.split(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2)
        x = x.transpose(1,2)
        x = self.norm(x)

        return x, H, W
        
def window_partition(x, win):
    B, H, W, C = x.shape
    x = x.view(B, H//win, win, W//win, win, C)
    x = x.permute(0,1,3,2,4,5).reshape(-1, win, win, C) # -1 = B * H//win *  W//win
    return x

def window_reverse(windows, win, H, W):
    B = int(windows.shape[0] / (H//win * W//win)) #divide to get the number of batches
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
        
        q = q.reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B_, heads, N, head_dim)
        k = k.reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

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
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, res, win, shift, heads):
        super().__init__()
        self.dim = dim
        self.res = res
        self.win = win 
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, win)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential( 
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim))
        
        H, W = res

        if shift > 0:
            self.mask = self.create_mask(H, W, win, shift)
        else:
            self.mask = None

    def create_mask(self, H, W, win, shift):
        img_mask = torch.zeros(1,H,W,1)
        count = 0

        for h in (slice(0,-win)), (slice(-win,-shift)), slice(-shift, None):
            for w in (slice(0,-win)), (slice(-win,-shift)), slice(-shift, None):
                img_mask[:,h,w,:] = count
                count += 1
            
        mask = window_partition(img_mask, win)
        mask = mask.view(-1, win*win)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask = mask.masked_fill(mask!=0, -10000.0)

        return mask
    
    def forward(self, x):
        B, L, C = x.shape
        H, W = self.res

        residual = x
        x = self.norm1(x)

        if self.shift > 0:
            x = torch.roll(x, shifts = (-self.shift, -self.shift), dims=(1,2))
        
        x = x.view(B, H, W, C)
        win_x = window_partition(x, self.win).view(-1, self.win*self.win, C)
        attn_out = self.attn(win_x, self.mask.to(x.device) if self.mask is not None else None)

        x = window_reverse(attn_out, self.win, H, W)

        if self.shift > 0:
            x = torch.roll(x, shifts = (+self.shift, +self.shift), dims=(1,2))
        
        x = residual + x.view(B,L,C)
        residual2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual2

        return x



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
    
class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape

        # expand channels
        x = self.expand(x)  # B, L, 2C

        # reshape to image
        x = x.view(B, H, W, 2 * C)

        x = x.view(B, H, W, 2, 2, C // 2)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)

        x = self.norm(x)

        return x, H * 2, W * 2
    
class FinalPatchExpand(nn.Module):
    def __init__(self, dim, scale=4):
        super().__init__()
        self.expand = nn.Linear(dim, dim * scale * scale, bias=False)
        self.norm = nn.LayerNorm(dim)

        self.scale = scale
        self.dim = dim

    def forward(self, x, H, W):
        B, L, C = x.shape

        x = self.expand(x)
        x = x.view(B, H, W, self.scale, self.scale, self.dim)

        x = x.permute(0,1,3,2,4,5).contiguous()
        x = x.view(B, H*self.scale, W*self.scale, self.dim)

        x = x.view(B, -1, self.dim)
        x = self.norm(x)

        return x, H*self.scale, W*self.scale