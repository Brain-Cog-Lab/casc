import torch
import torch.nn as nn
from einops import rearrange


def temporal_flatten(x: torch.tensor):
    shape = x.shape
    if len(x.shape) == 4:
        return x.reshape(1, shape[0] * shape[1], *shape[-2:])
    elif len(x.shape) == 5:
        return x.reshape(1, shape[0], shape[1] * shape[2], *shape[-2:])


class Encoder(nn.Module):
    '''
    (step, batch_size, )
    '''

    def __init__(self, step, encode_type='direct', **kwargs):
        super(Encoder, self).__init__()
        self.step = step
        self.fun = getattr(self, encode_type)
        self.encode_type = encode_type
        self.temporal_flatten = kwargs['temporal_flatten'] if 'temporal_flatten' in kwargs else False
        self.layer_by_layer = kwargs['layer_by_layer'] if 'layer_by_layer' in kwargs else False

    def forward(self, inputs, deletion_prob=None, shift_var=None):
        if len(inputs.shape) != 4:  # DVS data
            outputs = inputs.permute(1, 0, 2, 3, 4).contiguous()  # t, b, c, w, h

        else:  # static data
            if self.encode_type == 'auto':
                if self.fun.device != inputs.device:
                    self.fun.to(inputs.device)

            outputs = self.fun(inputs)
            if deletion_prob:
                outputs = self.delete(outputs, deletion_prob)
            if shift_var:
                outputs = self.shift(outputs, shift_var)

        if self.temporal_flatten:
            outputs = temporal_flatten(inputs)
        if self.layer_by_layer:
            outputs = rearrange(outputs, 't b c w h -> (t b) c w h')

        return outputs

    @torch.no_grad()
    def direct(self, inputs):
        shape = inputs.shape
        outputs = inputs.unsqueeze(0).repeat(self.step, *([1] * len(shape)))
        return outputs

    def auto(self, inputs):
        # TODO: Calc loss for firing-rate
        shape = inputs.shape
        outputs = self.fun(inputs)
        print(outputs.shape)
        return outputs

    @torch.no_grad()
    def ttfs(self, inputs):
        # print("ttfs")
        shape = (self.step,) + inputs.shape
        outputs = torch.zeros(shape, device=self.device)
        for i in range(self.step):
            mask = (inputs * self.step <= (self.step - i)
                    ) & (inputs * self.step > (self.step - i - 1))
            outputs[i, mask] = 1 / (i + 1)
        return outputs

    @torch.no_grad()
    def rate(self, inputs):
        shape = (self.step,) + inputs.shape
        return (inputs > torch.rand(shape, device=self.device)).float()

    @torch.no_grad()
    def phase(self, inputs):
        shape = (self.step,) + inputs.shape
        outputs = torch.zeros(shape, device=self.device)
        inputs = (inputs * 256).long()
        val = 1.
        for i in range(self.step):
            if i < 8:
                mask = (inputs >> (8 - i - 1)) & 1 != 0
                outputs[i, mask] = val
                val /= 2.
            else:
                outputs[i] = outputs[i % 8]
        return outputs

    @torch.no_grad()
    def delete(self, inputs, prob):
        mask = (inputs >= 0) & (torch.randn_like(
            inputs, device=self.device) < prob)
        inputs[mask] = 0.
        return inputs

    @torch.no_grad()
    def shift(self, inputs, var):
        # TODO: Real-time shift
        outputs = torch.zeros_like(inputs)
        for step in range(self.step):
            shift = (var * torch.randn(1)).round_() + step
            shift.clamp_(min=0, max=self.step - 1)
            outputs[step] += inputs[int(shift)]
        return outputs

    @torch.no_grad()
    def urate(self, inputs):
        """
        输入：累加膜电势数值a，阈值b，总仿真时间t
        输出：脉冲串
        对输入电流进行累加，使用阈值b进行归一化，根据t向下取整，能得到累积脉冲数
        然后做差分
        """
        curr = inputs.repeat((self.step, 1, 1, 1, 1)).cumsum(0).floor()
        return torch.cat([curr, torch.zeros_like(curr[0].unsqueeze(0))], 0).diff(dim=0).float()
