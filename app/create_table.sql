# 数据库初始化
-- 创建库
create database if not exists embedded_software
;

-- 切换库
use embedded_software;

# 用户表
create table user
(
    id           bigint auto_increment comment 'id' primary key,
    userName     varchar(256)                       null comment '用户昵称',
    userAccount  varchar(256)                       null comment '账号',
    userPassword varchar(512)                       not null comment '密码',
    createTime   datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime   datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete     tinyint  default 0                 not null comment '是否删除',
    userRole     varchar(256) default 'user'            not null comment '用户角色：user/admin/ban')
    comment '用户';

# 导入示例用户
INSERT INTO embedded_software.user (username, userAccount, userPassword, createTime, updateTime, isDelete, userRole) VALUES ('admin', 'admin', '74e8338cf7389e7508eb85031afc345c', '2025-07-20 14:14:22', '2025-07-20 14:39:37', 0, 'admin');


# 数据集信息表
create table datasets
(
    id           bigint auto_increment comment 'id' primary key,
    name         varchar(128)                       not null  comment '数据集名称',
    modality     varchar(128)                       not null  comment '模态类型(SAR, RD, 1D)',
    path         varchar(256)                       not null  comment '数据集路径（相对路径）',
    labels       varchar(128)                       not null  comment '标签',
    createTime   datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime   datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete     tinyint  default 0                 not null comment '是否删除')
    comment '数据集信息表';

# 图像数据信息表
create table images
(
    id           bigint auto_increment comment 'id' primary key,
    dataset_id   bigint                             not null  comment '所属数据集ID',
    name         varchar(128)                       not null  comment '所属数据集名称',
    modality     varchar(128)                       not null  comment '模态类型(SAR, RD, 1D)',
    path         varchar(256)                       not null  comment '数据文件路径（绝对路径）',
    label        varchar(128)                       not null  comment '标签',
    createTime   datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime   datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete     tinyint  default 0                 not null comment '是否删除')
    comment '图像数据信息表';


# 训练任务记录表
create table train_jobs
(
    id             bigint auto_increment              primary key comment 'ID',
    job_id         VARCHAR(128)                       null comment '任务ID（Celery task_id,用这个作为标识）',
    description    VARCHAR(256)                       null comment '任务描述，默认按照{数据集}_{主干模型}_{创建时间}_{编号}命名',
    user_id        bigint                             not null comment '提交用户ID',
    dataset_id     bigint                             not null comment '数据集ID',
    nodes          json                               not null comment '前端画布节点结构（JSON）',
    backbone_id    bigint                             not null comment '所用主干网络ID',
    classifier_id  bigint                             not null comment '所用分类器ID',
    feature_ids    json                               not null comment '使用的特征提取算子ID列表',
    adaptive       tinyint                            not null comment '是否开启网络自适应优化（0-否，1-是）',
    status         varchar(32)                        not null default 'pending' comment '状态（pending, running, done, failed）',
    progress       int                                default 0 comment '训练进度百分比',
    config         json                               null comment '训练过程的配置文件',
    result         json                               null comment '最终训练结果（如loss, acc）',
    model_path     varchar(128)                       null comment '最终训练生成的权重路径',
    log_path       varchar(128)                       null comment '训练日志路径',
    createTime     datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime     datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete       tinyint  default 0                 not null comment '是否删除'
) comment = '训练任务记录表';

# 模型测试记录表
create table test_records
(
    id             bigint auto_increment              primary key comment 'ID',
    val_job_id     varchar(64)                        null comment '测试任务ID（Celery task_id,用这个作为标识）',
    train_job_id   varchar(64)                        not null comment '关联训练任务ID',
    user_id        bigint                             not null comment '提交用户ID',
    config         json                               null comment '测试过程的配置文件',
    result         json                               null comment '测试输出结果（评估指标（准确率、召回率等）和结果文件的路径）',
    status         varchar(32)                        not null default 'pending' comment '状态（pending, running, done, failed）',
    createTime     datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime     datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete       tinyint  default 0                 not null comment '是否删除'
) comment = '模型测试记录表';

# 特征算子表
create table feature_extractors
(
    id             bigint auto_increment              primary key comment 'id',
    name           varchar(128)                       not null comment '特征算子名称',
    modality       varchar(128)                       not null comment '模态类型(SAR, RD, 1D)',
    params_schema  json                               null comment '参数结构体，JSON格式',
    description    text                               null comment '算子描述',
    createTime     datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime     datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete       tinyint  default 0                 not null comment '是否删除'
) comment = '特征提取算子信息表';

# 特征组合表
create table feature_groups
(
    id           bigint auto_increment primary key comment '特征组合ID',
    name         varchar(128) not null comment '组合名称',
    description  text null comment '组合描述',
    createTime   datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime   datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete     tinyint default 0 not null comment '是否删除'
) comment='特征组合信息表';

# 特征组合-特征算子关联表
create table feature_group_items
(
    id                  bigint auto_increment primary key comment '主键ID',
    group_id            bigint not null comment '特征组合ID',
    feature_extractor_id bigint not null comment '特征算子ID',
    createTime          datetime default CURRENT_TIMESTAMP null comment '创建时间',
    constraint fk_feature_group foreign key (group_id) references feature_groups (id) on delete cascade,
    constraint fk_feature_extractor foreign key (feature_extractor_id) references feature_extractors (id) on delete cascade,
    unique key uk_group_feature (group_id, feature_extractor_id)
) comment='特征组合与特征算子关联表';


# 神经网络骨干模块表
create table backbones
(
    id             bigint auto_increment              primary key comment 'id',
    name           varchar(128)                       not null comment '主干网络名称（如CNN、VGG16）',
    params_schema  json                               null comment '参数结构体，JSON格式',
    description    text                               null comment '主干模块说明',
    createTime     datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime     datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete       tinyint  default 0                 not null comment '是否删除'
) comment = '神经网络主干模块信息表';


# 神经网络分类器信息表
create table classifiers
(
    id             bigint auto_increment              primary key comment 'id',
    name           varchar(128)                       not null comment '分类器名称（如MLP, SVM等）',
    params_schema  json                               null comment '参数结构体，JSON格式',
    description    text                               null comment '分类器说明',
    createTime     datetime default CURRENT_TIMESTAMP null comment '创建时间',
    updateTime     datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    isDelete       tinyint  default 0                 not null comment '是否删除'
) comment = '神经网络分类器信息表';

