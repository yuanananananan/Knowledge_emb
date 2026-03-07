from ruamel.yaml import YAML
import json
from collections import OrderedDict
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dqs
import ast

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.preserve_quotes = True


def make_cm():
    """快捷创建 CommentedMap"""
    return CommentedMap()

def force_type(key, value, typ):
    """
    根据 type 字段强制类型：
    - string → 强制为 "xxx"
    - boolean → True/False
    - number → int/float（由 JSON 本身决定）
    - list → 保持原样（内部递归处理字符串）
    """
    # ========= 特殊处理 img_size: "[128,128]" → [128, 128] =========
    if key == "img_size" and isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass  # 解析失败就保持原值

    if typ == "string":
        # 强制字符串加引号
        return dqs(str(value))

    if typ == "boolean":
        # JSON 的 true/false 自动变成 Python 的 True/False
        print(value)
        return bool(value)

    if typ == "number":
        return value  # int/float 保持原样

    if typ == "list":
        # 列表中 string 也需要加引号
        out = []
        for v in value:
            if isinstance(v, str):
                out.append(dqs(v))
            else:
                out.append(v)
        return out

    return value

def build_config(extra_list, external_feature_list):
    """
    将 extra[] 结构构造成目标 YAML 配置（使用 CommentedMap，支持行尾注释）
    """
    cfg = make_cm()

    # 顶层结构
    cfg["data"] = make_cm()
    cfg["model"] = make_cm()
    cfg["model"]["GNN"] = make_cm()
    cfg["model"]["Cross_attention"] = make_cm()
    cfg["training"] = make_cm()
    cfg["training"]["optimizer"] = make_cm()
    # feature_combinations 不从 JSON 来，直接使用外部传入
    cfg["feature_combinations"] = external_feature_list
    cfg["is_nas"] = None
    cfg["is_ssl"] = None

    # 分类 key 集合（便于映射）
    data_keys = {
        "mode", "name", "root", "num_classes",
        "batch_size", "num_workers", "shuffle",
        "train_ratio", "img_size", "is_aug"
    }
    model_keys = {"backbone", "pretrained", "output_layer", "d_model"}
    gnn_keys = {"is_GNN", "k", "n_blocks", "n_nodes", "dropout"}
    attn_keys = {"is_attention"}
    train_keys = {"epochs", "device", "checkpoint_dir"}
    optimizer_keys = {"type", "lr", "weight_decay", "momentum"}

    for item in extra_list:
        key = item.get("name")
        # value = item.get("value")
        typ = item["type"]
        value = force_type(key, item["value"], typ)
        desc = item.get("description", "")
        # Skip feature_combinations, because external list overrides it
        if key == "feature_combinations":
            continue

        if key in data_keys:
            cfg["data"][key] = value
            # 在 data 的对应 key 添加行尾注释
            if desc:
                cfg["data"].yaml_add_eol_comment(f" {desc}", key)
            continue

        if key in model_keys:
            cfg["model"][key] = value
            if desc:
                cfg["model"].yaml_add_eol_comment(f" {desc}", key)
            continue

        if key in gnn_keys:
            cfg["model"]["GNN"][key] = value
            if desc:
                cfg["model"]["GNN"].yaml_add_eol_comment(f" {desc}", key)
            continue

        if key in attn_keys:
            cfg["model"]["Cross_attention"][key] = value
            if desc:
                cfg["model"]["Cross_attention"].yaml_add_eol_comment(f" {desc}", key)
            continue

        if key in train_keys:
            cfg["training"][key] = value
            if desc:
                cfg["training"].yaml_add_eol_comment(f" {desc}", key)
            continue

        if key in optimizer_keys:
            cfg["training"]["optimizer"][key] = value
            if desc:
                cfg["training"]["optimizer"].yaml_add_eol_comment(f" {desc}", key)
            continue

        if key == "feature_combinations":
            cfg["feature_combinations"] = value if value is not None else []
            if desc:
                # feature_combinations 在顶层
                cfg.yaml_add_eol_comment(f" {desc}", "feature_combinations")
            continue

        if key == "is_nas":
            cfg["is_nas"] = value
            if desc:
                cfg.yaml_add_eol_comment(f" {desc}", "is_nas")
            continue

        if key == "is_ssl":
            cfg["is_ssl"] = value
            if desc:
                cfg.yaml_add_eol_comment(f" {desc}", "is_ssl")
            continue

        # 如果遇到未分类的字段，默认放到顶层（可根据需要调整）
        cfg[key] = value
        if desc:
            cfg.yaml_add_eol_comment(f" {desc}", key)

    return cfg


def save_yaml(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)


def json_extra_to_yaml(json_data, yaml_path, external_feature_list):
    extra_list = json_data.get("extra", [])
    cfg = build_config(extra_list, external_feature_list)
    save_yaml(cfg, yaml_path)


# ======================
# ⭐ 示例调用
# ======================
if __name__ == "__main__":
    # 你的 JSON 输入
    json_input = {
      "extra": [
        {
          "description": "数据场景",
          "name": "mode",
          "type": "string",
          "value": "SAR"
        },
        {
          "description": "数据集名称",
          "name": "name",
          "type": "string",
          "value": "FUSAR"
        },
        {
          "description": "数据存储路径",
          "name": "root",
          "type": "string",
          "value": "data/FUSAR_Ship_3"
        },
        {
          "description": "类别数量",
          "name": "num_classes",
          "type": "number",
          "value": 4
        },
        {
          "description": "批量大小",
          "name": "batch_size",
          "type": "number",
          "value": 16
        },
        {
          "description": "数据加载线程数",
          "name": "num_workers",
          "type": "number",
          "value": 0
        },
        {
          "description": "是否打乱数据",
          "name": "shuffle",
          "type": "boolean",
          "value": True
        },
        {
          "description": "训练集比例",
          "name": "train_ratio",
          "type": "number",
          "value": 0.1
        },
        {
          "description": "输入尺寸 [通道, 高, 宽]",
          "name": "img_size",
          "type": "list",
          "value": [
            128,
            128
          ]
        },
        {
          "description": "是否采用数据增强\n\n# model_config",
          "name": "is_aug",
          "type": "boolean",
          "value": True
        },
        {
          "description": "识别模型主干",
          "name": "backbone",
          "type": "string",
          "value": "resnet18"
        },
        {
          "description": "是否使用预训练权重",
          "name": "pretrained",
          "type": "boolean",
          "value": False
        },
        {
          "description": "输出层",
          "name": "output_layer",
          "type": "string",
          "value": "fc"
        },
        {
          "description": "中间特征长度",
          "name": "d_model",
          "type": "number",
          "value": 128
        },
        {
          "description": "GNN.is_GNN 配置项。",
          "name": "is_GNN",
          "type": "boolean",
          "value": True
        },
        {
          "description": "GNN.k 配置项。",
          "name": "k",
          "type": "number",
          "value": 4
        },
        {
          "description": "GNN.n_blocks 配置项。",
          "name": "n_blocks",
          "type": "number",
          "value": 4
        },
        {
          "description": "GNN.n_nodes 配置项。",
          "name": "n_nodes",
          "type": "number",
          "value": 64
        },
        {
          "description": "GNN.dropout 配置项。",
          "name": "dropout",
          "type": "number",
          "value": 0.1
        },
        {
          "description": "Cross_attention.is_attention 配置项。",
          "name": "is_attention",
          "type": "boolean",
          "value": True
        },
        {
          "description": "训练轮数",
          "name": "epochs",
          "type": "number",
          "value": 10
        },
        {
          "description": "训练设备",
          "name": "device",
          "type": "string",
          "value": "cuda"
        },
        {
          "description": "模型保存目录",
          "name": "checkpoint_dir",
          "type": "string",
          "value": "model_ckp"
        },
        {
          "description": "optimizer.type 配置项。",
          "name": "type",
          "type": "string",
          "value": "Adam"
        },
        {
          "description": "optimizer.lr 配置项。",
          "name": "lr",
          "type": "number",
          "value": 0.001
        },
        {
          "description": "optimizer.weight_decay 配置项。",
          "name": "weight_decay",
          "type": "number",
          "value": 0.0001
        },
        {
          "description": "optimizer.momentum 配置项。",
          "name": "momentum",
          "type": "number",
          "value": 0.9
        },
        {
          "description": "优选组合特征（数量不定）",
          "name": "feature_combinations",
          "type": "list",
          "value": ""
        },
        {
          "description": "高效训练损失",
          "name": "is_nas",
          "type": "boolean",
          "value": False
        },
        {
          "description": "is_ssl 配置项。",
          "name": "is_ssl",
          "type": "boolean",
          "value": False
        }
      ]
    }
    external_feature_list = [
        "get_spatial_feat",
        "get_statis_feat",
        "get_time_freq_feat"
    ]
    json_extra_to_yaml(json_input, "output.yaml", external_feature_list)
    print("YAML 已保存到 output.yaml")
