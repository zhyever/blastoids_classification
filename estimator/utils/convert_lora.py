import torch
from torch import nn
# import loralib as lora
from mmengine import print_log

def convert_zoed_lora(model, lora_linear_rank, cfg=None):
    modules = dict()
    for name, layer in model.named_modules():
        if 'core.core.pretrained' in name:
            modules[name] = layer

    for name, attn_processor in list(modules.items()):
        branches = name.split('.')
        basename = branches.pop(-1)
        parent_layer = modules.get('.'.join(branches), model)
        # bugs in conv2d, just skip it now
        # if isinstance(attn_processor, nn.Conv2d) and not isinstance(attn_processor, lora.ConvLoRA):
        #     # self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs
        #     print_log('Convert {} to lora.ConvLoRA'.format(name), logger='current')
        #     attn_processor = lora.ConvLoRA(
        #         nn.Conv2d,
        #         attn_processor.in_channels,
        #         attn_processor.out_channels,
        #         attn_processor.kernel_size[0],
        #         stride=attn_processor.stride,
        #         padding=attn_processor.padding,
        #         bias=False if attn_processor.bias is None else True,
        #         r=lora_linear_rank
        #     )
        #     setattr(parent_layer, basename, attn_processor)
        if isinstance(attn_processor, nn.Linear) and not isinstance(attn_processor, lora.Linear):
            if 'qkv' in name:
                print_log('Convert {} to lora.MergedLinear'.format(name), logger='current')
                attn_processor = lora.MergedLinear(
                    attn_processor.in_features, 
                    attn_processor.out_features, 
                    enable_lora=[True, False, True],
                    r=lora_linear_rank
                )
            else:
                print_log('Convert {} to lora.Linear'.format(name), logger='current')
                attn_processor = lora.Linear(
                    attn_processor.in_features,
                    attn_processor.out_features,
                    bias=False if attn_processor.bias is None else True,
                    r=lora_linear_rank
                )
            setattr(parent_layer, basename, attn_processor)
    
    for name, param in model.named_parameters():
        if 'core.core.pretrained' in name:
            if 'lora_' not in name:
                print_log('Set {} requires_grad as False'.format(name), logger='current')
                param.requires_grad = False

    # HACK: Here is the hack to reload model parameters
    ckpt = torch.load(cfg.model.deep_branch.pretrained_resource.split('::')[1], map_location='cpu')
    model.deep_model_zoe.load_state_dict(ckpt['model'], strict=False)

def get_model_param_dict(model):
    default_dict = model.state_dict()
    save_dict = dict()
    for k, v in default_dict.items():
        if 'core.core.pretrained' not in k:
            save_dict[k] = v

    lora_dict = lora.lora_state_dict(model)
    
    update_dict = {**save_dict, **lora_dict}
    return update_dict