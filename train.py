import os
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataset.sar import get_sar_dataset_full
from model import resnet18, resnet50, KED
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import load_config, get_dataloader, get_model, save_model, load_model
from app.utils.log_streamer import append_log


def train_step(model, im, label, feature, criterion, is_ked):
    model.train()
    im = im.cuda()
    label = label.cuda()
    feature = [f.cuda() for f in feature]

    if is_ked:
        result, feats = model(im, feature)
        loss = criterion(result, label)
    else:
        result, feats = model(im)
        loss = criterion(result, label)

    return loss


def val_step(model, im, label, feature, criterion, is_ked):
    with torch.no_grad():
        model.eval()
        im = im.cuda()
        label = label.cuda()
        feature = [f.cuda() for f in feature]

        if is_ked:
            result, feats = model(im, feature)
        else:
            result, feats = model(im)
        loss = criterion(result, label)

    return loss, result


# def train(cfg, model, train_loader, test_loader, job_id, update_fn=None):
#     epoches = cfg['training']['epochs']
#     checkpoint_dir = cfg['training']['checkpoint_dir']
#     is_ked = cfg['model']['GNN']['is_GNN']
#     num_classes = cfg['data']['num_classes']
#
#     # optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     optim = torch.optim.AdamW(
#         model.parameters(),
#         lr=1e-4,  # 初始学习率
#         betas=(0.9, 0.999),  # β1和β2
#         weight_decay=0.05,  # 权重衰减
#         eps=1e-8  # 数值稳定性项
#     )
#     criterion = torch.nn.CrossEntropyLoss()
#
#     model.cuda()
#     OA = -1
#     F1 = -1
#     LOG = {}
#     train_loss_global = []
#     test_loss_global = []
#     for ep in range(epoches):
#         losses_list = []
#         for im, label, feature in tqdm(train_loader):
#             loss = train_step(model, im, label, feature, criterion, is_ked)
#             loss.backward()
#             optim.step()
#             optim.zero_grad()
#             losses_list.append(loss.mean())
#         losses = 0.
#         for item in losses_list:
#             losses += item
#         losses = losses / len(losses_list)
#         train_loss_global.append(losses.item())
#
#         # val stage
#         correct = 0
#         total = 0
#         losses_list = []
#         class_total = np.zeros(num_classes)
#         class_correct = np.zeros(num_classes)
#
#         all_preds = []  # 用来保存所有预测值
#         all_labels = []  # 用来保存所有真实标签
#         for im, label, feature in tqdm(test_loader):
#             loss, result = val_step(model, im, label, feature, criterion, is_ked)
#             result = result.max(-1)[1].float()  # 选择最大值索引作为预测结果
#             result_bin = torch.where(result == label.cuda(), 1., 0.)  # 计算是否预测正确
#             correct += result_bin.sum()
#             total += len(result_bin)
#             losses_list.append(loss.mean())
#
#             # 计算每个类别的正确数和总数
#             for i in range(label.size(0)):
#                 l = label[i]
#                 class_total[l] += 1
#                 class_correct[l] += result_bin[i].item()
#
#             # 收集所有的预测和标签用于计算Kappa
#             all_preds.extend(result.cpu().numpy())
#             all_labels.extend(label.cpu().numpy())
#         # 计算整体的Precision, Recall, F1
#         precision = precision_score(all_labels, all_preds, average='macro')
#         recall = recall_score(all_labels, all_preds, average='macro')
#         f1 = f1_score(all_labels, all_preds, average='macro')
#
#         # 计算损失
#         losses = 0.
#         for item in losses_list:
#             losses += item
#         losses = losses / len(losses_list)
#         test_loss_global.append(losses.item())
#
#         # 计算每个类别的准确率
#         class_accuracies = [
#             round(100 * class_correct[i] / class_total[i], 2) if class_total[i] > 0 else 0 for i in range(num_classes)
#         ]
#
#         oa = 100 * correct / total
#         kappa = cohen_kappa_score(all_labels, all_preds)
#         print(
#             f"epoch:{ep}, OA: {oa:.2f}%, Kappa: {kappa:.4f}, Cls Acc: {class_accuracies}, P:{precision:.4f},R:{recall:.4f},F1:{f1:.4f},loss:{losses.item():.2f}"
#         )
#         append_log(job_id,
#                    f"epoch:{ep}, OA: {oa:.2f}%, Kappa: {kappa:.4f}, Cls Acc: {class_accuracies}, P:{precision:.4f},R:{recall:.4f},F1:{f1:.4f},loss:{losses.item():.2f}")
#         if oa > OA:
#             OA = oa
#             LOG.update({"OA": OA, "kappa": kappa, "ep": ep})
#             save_model(model, f'model_ckp/model_best.pth')
#         if update_fn:
#             update_fn(ep)
#     print(LOG)
#     plt.plot([i for i in range(len(train_loss_global))], train_loss_global, marker='o', color='b')
#     plt.plot([i for i in range(len(test_loss_global))], test_loss_global, marker='x', color='r')
#     plt.legend()
#     plt.savefig("img/loss.png")
#     return model
def train(cfg, model, train_loader, test_loader, job_id, update_fn=None):
    epoches = int(cfg['training']['epochs'])
    checkpoint_dir = cfg['training']['checkpoint_dir']
    is_ked = bool(cfg['model']['GNN']['is_GNN'])
    num_classes = int(cfg['data']['num_classes'])

    optim = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.cuda()
    OA = -1.0
    F1 = -1
    LOG = {}
    train_loss_global = []
    test_loss_global = []
    metrics_per_epoch = []  # 保存每轮指标

    for ep in range(epoches):
        # ======== Train ========
        losses_list = []
        for im, label, feature in tqdm(train_loader):
            loss = train_step(model, im, label, feature, criterion, is_ked)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses_list.append(loss.mean().item())  # 转成 float
        train_loss = float(sum(losses_list) / len(losses_list))
        train_loss_global.append(train_loss)

        # ======== Validation ========
        correct = 0.0
        total = 0
        losses_list = []
        class_total = np.zeros(num_classes)
        class_correct = np.zeros(num_classes)

        all_preds = []
        all_labels = []
        for im, label, feature in tqdm(test_loader):
            loss, result = val_step(model, im, label, feature, criterion, is_ked)
            result = result.max(-1)[1].float()
            result_bin = torch.where(result == label.cuda(), 1., 0.)
            correct += float(result_bin.sum().item())
            total += len(result_bin)
            losses_list.append(loss.mean().item())

            for i in range(label.size(0)):
                l = int(label[i].item())
                class_total[l] += 1
                class_correct[l] += float(result_bin[i].item())

            all_preds.extend([int(x) for x in result.cpu().numpy()])
            all_labels.extend([int(x) for x in label.cpu().numpy()])

        precision = float(precision_score(all_labels, all_preds, average='macro'))
        recall = float(recall_score(all_labels, all_preds, average='macro'))
        f1 = float(f1_score(all_labels, all_preds, average='macro'))
        val_loss = float(sum(losses_list) / len(losses_list))
        test_loss_global.append(val_loss)

        class_accuracies = [
            round(100.0 * class_correct[i] / class_total[i], 2) if class_total[i] > 0 else 0.0
            for i in range(num_classes)
        ]

        oa = float(100.0 * correct / total)
        kappa = float(cohen_kappa_score(all_labels, all_preds))

        # 保存每轮指标
        metrics_per_epoch.append({
            "epoch": int(ep),
            "OA": round(oa, 4),
            "Kappa": round(kappa, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Train_Loss": round(train_loss, 4),
            "Test_Loss": round(val_loss, 4),
            "Class_Accuracy": [float(x) for x in class_accuracies]
        })

        print(f"epoch:{ep}, OA: {oa:.2f}%, Kappa: {kappa:.4f}, "
              f"Cls Acc: {class_accuracies}, P:{precision:.4f}, R:{recall:.4f}, "
              f"F1:{f1:.4f}, loss:{val_loss:.2f}")
        append_log(
            job_id, f"epoch:{ep}, OA: {oa:.2f}%, Kappa: {kappa:.4f}, "
            f"Cls Acc: {class_accuracies}, P:{precision:.4f}, "
            f"R:{recall:.4f}, F1:{f1:.4f}, loss:{val_loss:.2f}")
        if oa > OA:
            OA = oa
            LOG.update({"OA": round(OA, 4), "kappa": round(kappa, 4), "ep": int(ep)})
            save_model(model, f'model_ckp/{job_id}/model_best.pth')
        if update_fn:
            update_fn(ep)

    # 保存 loss 曲线
    plt.plot(range(len(train_loss_global)), train_loss_global, marker='o', color='b')
    plt.plot(range(len(test_loss_global)), test_loss_global, marker='x', color='r')
    plt.legend()
    plt.savefig("img/loss.png")

    # 汇总结果
    summary = {
        "Best": LOG,
        "Metrics": metrics_per_epoch,
        "Train_Loss": [float(x) for x in train_loss_global],
        "Test_Loss": [float(x) for x in test_loss_global]
    }
    model_path = f"model_ckp/{job_id}/model_best.pth"
    log_path = f"logs/{job_id}.log"
    abs_model = os.path.abspath(model_path)
    abs_log = os.path.abspath(log_path)
    # 返回模型和纯 Python 数据
    return model, summary, abs_model, abs_log


if __name__ == '__main__':
    seed = 10001
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = load_config('configs/mon_config.yaml')
    train_loader, test_loader = get_dataloader(cfg)
    
    # model = resnet18(pretrained=True, num_classes=num_classes)
    # model = KED(cfg)
    model = get_model(cfg)

    train(cfg, model, train_loader, test_loader, job_id="2026_3_3")
