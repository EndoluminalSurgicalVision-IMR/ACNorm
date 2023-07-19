import torch
import numpy as np


class AC_Corr(object):
    """
        Calculate the AC-Corr for transferability estimation.
        source-info: {'model_path':'pretrained_model_path', prefix:'module.'}
        target-info: {'model_path':'target_model_path_at_one_epoch', prefix:'module.'}
        layer_key: At which layer to compute the AC-Corr
    """

    def __init__(self, source_info: dict, target_info: dict, layer_key: list):
        self.source_model_path = source_info['model_path']
        self.target_model_path = target_info['model_path']
        self.source_prefix = source_info['prefix']
        self.target_prefix = target_info['prefix']
        self.layer_key = layer_key

    def get_corr(self, weight, bias, weight_target, bias_target, eps=1e-5):
        y_s = bias / (weight.abs() + eps)
        y_t = bias_target.data / (weight_target.data.abs() + eps)
        y_s = y_s.view(-1, 1)
        y_t = y_t.view(-1, 1)
        corr = -torch.abs(y_t - y_s.t())
        corr = torch.softmax(corr, dim=1)
        corr = torch.where((corr - torch.diag(corr).unsqueeze(1)) >= 0, corr, torch.zeros_like(corr))
        corr = corr.cpu().numpy()
        return corr

    def get_state_dict(self, load_from, prefix='module.'):
        # load a pretrained model
        checkpoint = torch.load(load_from)['state_dict']
        print('Load model from:', load_from)

        # remove prefix
        state_dict = {k[len(prefix):]: v for k, v in checkpoint.items()}

        # select the transferred layer
        layer_names = []
        for (k, v) in state_dict.items():
            for layer in self.layer_key:
                if layer in k:
                    #if 'up_conv' not in k:
                    start = len(layer) #+ 7
                    if k[:start] not in layer_names:
                        layer_names.append(k[:start])

        return state_dict, layer_names

    def cal_ac_corr(self):
        source_dict, source_conv_names = self.get_state_dict(self.source_model_path, self.source_prefix)
        target_dict, target_conv_names = self.get_state_dict(self.target_model_path, self.target_prefix)

        # get each bn layer
        corr_sum = []
        for layer in source_conv_names:
            bn_layer_name = layer + 'bn1'
            print('Layer:', bn_layer_name)
            weight = source_dict[bn_layer_name + '.weight']
            bias = source_dict[bn_layer_name + '.bias']
            weight_target = target_dict[bn_layer_name + '.weight_target']
            bias_target = target_dict[bn_layer_name + '.bias_target']
            corr = self.get_corr(weight, bias, weight_target, bias_target, eps=1e-5)
            corr_sum.append(corr.sum())

        ac_corr = np.mean(corr_sum)
        print('AC-Corr:', ac_corr)
        return ac_corr


if __name__ == '__main__':
    source_info = {'model_path': 'the path to the pretrained model',
     'prefix': 'module.'}
    target_info = {'model_path': 'the path to the target model which is fine-tuned with AC-Norm for only one    epoch',
    'prefix': 'module.'}
    AC_Corr = AC_Corr(source_info, target_info, layer_key=['down_tr512.ops.1.'])
    Trans_value = AC_Corr.cal_ac_corr()

