import torch.nn as nn

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
    
class PatchMerging(nn.Module):
    def __init__(self,input_resolution,dim):
        super().__init__()

    def forward(self, x):
        x = x.split(input_resolution, dim)
        for each i of the same group:
            concatenate(x[i])
    
        return x
    
class SelfAttention():

    def forward(x):
        for query in tokens:
            new_value = Vector()

            for (key, value) in tokens:
                new_value += value * similarity(key, query)


    
def window_partition(x, win):
    B, H, W, C = x.shape
    x = x.view(B, H//win, win, W//win, win, C)
    return x

def window_reverse(windows, win, H, W):

    return x



class SwinTransformerBlock(nn.Module):
    def __init__():
        super().__init__()