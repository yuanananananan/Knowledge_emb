import json
import ast


def force_type_json(key, value, typ):
    """
    按 JSON 格式强制类型：
    - string → str
    - boolean → bool
    - number → int/float
    - list → list
    """

    # ------ 特殊处理 img_size: "[128,128]" → [128,128] ------
    if key == "img_size" and isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    if typ == "string":
        return str(value)

    if typ == "boolean":
        return bool(value)

    if typ == "number":
        return value

    if typ == "list":
        # 保持结构，不需要 YAML 的包装
        return value

    return value


def build_json_config(extra_list, external_feature_list):
    """
    将 extra 数组构造成目标 JSON 配置
    """

    cfg = {
        "data": {},
        "model": {
            "GNN": {},
            "Cross_attention": {}
        },
        "training": {
            "optimizer": {}
        },
        "feature_combinations": external_feature_list,
        "is_nas": False,
        "is_ssl": False
    }

    # 分类字段
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
        key = item["name"]
        typ = item["type"]
        value = force_type_json(key, item["value"], typ)

        # ---------- feature_combinations: JSON 不从 extra 用 ----------
        if key == "feature_combinations":
            continue

        if key in data_keys:
            cfg["data"][key] = value
            continue

        if key in model_keys:
            cfg["model"][key] = value
            continue

        if key in gnn_keys:
            cfg["model"]["GNN"][key] = value
            continue

        if key in attn_keys:
            cfg["model"]["Cross_attention"][key] = value
            continue

        if key in train_keys:
            cfg["training"][key] = value
            continue

        if key in optimizer_keys:
            cfg["training"]["optimizer"][key] = value
            continue

        if key == "is_nas":
            cfg["is_nas"] = value
            continue

        if key == "is_ssl":
            cfg["is_ssl"] = value
            continue

        # 未分类 → 顶层
        cfg[key] = value

    return cfg


def json_extra_to_json(json_data, external_feature_list):
    extra_list = json_data#.get("extra", [])
    cfg = build_json_config(extra_list, external_feature_list)

    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(cfg, f, indent=4, ensure_ascii=False)

    return cfg


if __name__ == '__main__':
    json_data = {
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
    cfg = json_extra_to_json(
        json_data,
        ["get_spatial_feat", "get_statis_feat", "get_time_freq_feat"]
    )
    print(cfg)