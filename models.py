import torch
import torch.nn.functional as F

def total_variation(feats: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(feats):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(feats)}")
    feats_shape = feats.shape
    if  len(feats_shape) == 4:
        pixel_dif1 = feats[..., 1:, :] - feats[..., :-1, :]
        pixel_dif2 = feats[..., :, 1:] - feats[..., :, :-1]
        reduce_axes = (-3, -2, -1)
    else:
        raise ValueError("Expected input tensor to be of ndim 4, but got " + str(len(feats_shape)))
    tv_loss = pixel_dif1.abs().sum(dim=reduce_axes) + pixel_dif2.abs().sum(dim=reduce_axes)
    tv_loss = tv_loss.mean(axis=0)
    return tv_loss

class Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, tv_losses=[]):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.tv_losses = tv_losses

    def forward(self, x):
        x = super(Conv2d, self).forward(x)
        if self.training:
            self.tv_losses.append(total_variation(x))
        return x

class TVModel(torch.nn.Module):
    def __init__(self, model, num_tv_layers=None, layer_name=None):
        super(TVModel, self).__init__()
        self.model = model
        self.num_tv_layers = num_tv_layers
        self.tv_losses = []
        self.tv_layer_cnt = 0
        if layer_name:
            self._add_tv_to_layer( model, layer_name)
        else:
            self._add_tv_to_conv(self.model, self.tv_losses)

    def _add_tv_to_layer(self, model, layer_name):
        if self.training:
            print(f"Adding tv hook to {layer_name}")
            module = getattr(model, layer_name)
            module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        tv_loss = total_variation(output)
        self.tv_losses.append(tv_loss)

    def _add_tv_to_conv(self, model, losses):
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Conv2d):
                setattr(model, child_name, Conv2d(in_channels=child.in_channels,
                                                  out_channels=child.out_channels,
                                                  kernel_size=child.kernel_size,
                                                  stride=child.stride,
                                                  padding=child.padding,
                                                  bias=child.bias,
                                                  tv_losses=losses))
                self.tv_layer_cnt+=1

            else:
                if self.num_tv_layers and self.tv_layer_cnt == self.num_tv_layers:
                    break
                self._add_tv_to_conv(child, losses)

    def _reset_losses(self):
        self.tv_losses.clear()

    def forward(self, x):
        self._reset_losses()
        x = self.model(x)
        return x


class LFHFModel(torch.nn.Module):
    def __init__(self, lf_model, hf_model, lf_ckpt, hf_ckpt, normalize=True):
        super(LFHFModel, self).__init__()
        self.lf_model = lf_model
        self.hf_model = hf_model
        self.normalize = normalize
        if lf_ckpt:
            self._load_ckpt(lf_model, lf_ckpt)
        if hf_ckpt:
            self._load_ckpt(hf_model.model, hf_ckpt)

    def forward(self, x, return_all_logits=False):
        lf_logits = self.lf_model(x)
        hf_logits = self.hf_model(x)
        wts = [0.5, 0.5]
        if return_all_logits:
            return lf_logits, hf_logits
        else:
            if self.normalize:
                p1 = F.softmax(lf_logits, dim=1)
                p2 = F.softmax(hf_logits, dim=1)
                return p1*wts[0] + p2*wts[1]
            else:
                return wts[0]*lf_logits + wts[1]*hf_logits

    def _load_ckpt(self, model, path):
        checkpoint = torch.load(path)
        new_state_dict = {}
        for k,v in checkpoint['state_dict'].items():
            new_key  = k.replace("module.", "").replace("model.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
