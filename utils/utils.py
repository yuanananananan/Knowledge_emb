import yaml
import os
import torch
from dataset import get_sar_dataset_full, get_rd_dataset, get_MON_dataset
from torch.utils.data import DataLoader
from model import resnet18, resnet50, KED


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:  # 明确指定utf-8编码
        config = yaml.safe_load(f)
    return config


def get_dataloader(cfg):

    mode = cfg['data']['mode']
    assert mode in ['RD', 'SAR', 'MON']
    if mode == 'SAR':
        train_dataset, test_dataset = get_sar_dataset_full(cfg)
    elif mode == 'RD':
        train_dataset, test_dataset = get_rd_dataset(cfg)
    elif mode == 'MON':
        train_dataset, test_dataset = get_MON_dataset(cfg)
    print(f"Total samples: {len(train_dataset)}")
    # Create data loader
    batch_size = cfg['data']['batch_size']
    shuffle = cfg['data']['shuffle']
    num_workers = cfg['data']['num_workers']
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader


def get_model(cfg):

    if cfg['model']['GNN']['is_GNN']:
        model = KED(cfg=cfg)
    else:
        # cfg['model']['backbone'] -> resnet18, resnet34, resnet50 vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
        model_fnc = eval(cfg['model']['backbone'])
        num_classes = cfg['data']['num_classes']
        model = model_fnc(pretrained=True, num_classes=num_classes)
    return model


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cuda'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件 {path} 不存在")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

