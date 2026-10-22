# 工作流配置每卡 batch

无需修改 Kubernetes YAML。两个可选字段放在工作流顶层 `config`，不要声明为
`input.parameters`，也不要在步骤 `parameters.input` 里映射，避免引擎 6.1 再次报缺少外部环境变量。

## 80GB 单卡试跑配置

完整可复制模板：`change_detection_wf_single_gpu_80g.json`，正常新跑，不从旧任务恢复。
原工作流其他部分保持不变，只需更换以下 config：

```json
"config": {
    "fast_run": false,
    "change_batch_size": 8,
    "classification_batch_per_gpu": 16,
    "resume_from": "",
    "resume_allow_legacy": false,
    "classification_resume_from": "",
    "classification_resume_allow_legacy": false
}
```

这是建议的试跑起点，未在新服务器实测，不保证最优或绝不 OOM。
80GB 是显存容量，不代表可用显存或实际算力；GPU 是否独占、模型、切片尺寸、双模型模式均影响峰值。
先比较前几组非空影像的耗时、有效切片/秒、显存峰值和 GPU 利用率，避免拿空图跳过的速度做比较。
不要为了把显存填满而增大 batch；读取/预处理/写盘慢时，增大 batch 可能收益很小。

观察稳定且吞吐确有改善后，可以单独试变化检测 16 或分类 32；一次只改一个算法。
OOM 时新建任务把对应 batch 减半，不会自动吞掉 OOM 并假装成功，也没有自动降 batch 重试。
变化检测已有 `[效率]` 日志；分类可比较同规模非空组的前后时相耗时。

## 含义和范围

| 配置项 | 不填时默认 | 允许范围 | 含义 |
| --- | --- | --- | --- |
| `change_batch_size` | 4 | 1～32 | 每个 GPU 计算进程一次处理的变化检测切片对数 |
| `classification_batch_per_gpu` | 8 | 1～64 | 每张 GPU 一次分类的切片数 |

- batch 不是同时处理多少幅完整大图，不改变逐影像对的任务调度。
- 变化检测每个 GPU 进程仍处理自己的影像对，切片尺寸保持原配置（目前1280）。
- 分类单卡时总 batch 等于配置值；双卡 DataParallel 时总 batch 是每卡值×可见GPU数。
  例如分类16：单卡16、双卡32。仍是同一幅影像切片并行，不是每卡独立一幅影像。
- 分类1024原切片、两倍下采样512模型输入不变。无GPU的CPU模式及农业模型特殊模式仍使用其安全 batch=1。
- `process: "single"` 不改；GPU资源仍由平台分配，这些字段不会申请更多GPU。
- 普通模板 `change_detection_wf.json` 保持4/8；未填字段的旧工作流也保持4/8。

## 生效路径

入口在启动推理前读取配置，写入子进程继承的环境变量：

- `change_batch_size` → `CHANGE_DETECTION_BATCH_SIZE`
- `classification_batch_per_gpu` → `CLIE_BATCH_PER_GPU`

优先级：显式 CLI > 上述环境变量 > `WF_CONFIG.config` > 默认值。
CLI 分别为 `--change_batch_size`、`--classification_batch_per_gpu`。
若平台已有同名环境变量，会覆盖 config，启动日志会明确打印来源：

```text
[Batch配置] change_batch_size=8（每张 GPU 的切片数，来源=WF_CONFIG.config）
[Batch配置] classification_batch_per_gpu=16（每张 GPU 的切片数，来源=WF_CONFIG.config）
```

分类实际设备日志还会显示 `batch_per_gpu=16, batch_size=16`（单卡）。
传入0、负数、布尔值、小数、空值或超出范围会在入口报错，不静默截断成另一个数值。
分类推理原来的16上限已扩到64，续跑清单记录相同的实际每卡值，避免“配置32实际却16”。

## 续跑和 HAMi

新版本完成清单含 batch。修改 batch 会使严格签名校验不匹配，原结果可能重新计算；
续跑任务尽量保持原 batch，不要为了改 batch 开启 legacy 绕过验证。
旧任务本来没有完成清单时，legacy 的人工确认要求仍然适用。

本改动不移除 HAMi 保护。存在 HAMi 时分类仍强制 data_workers=0；
新机没有 HAMi 标记时走普通数据加载配置。修改 batch 不等于修改 data_workers。
更新代码和工作流只对新启动进程生效，运行中的任务不会动态切换 batch。

参考：[PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
说明提高 batch 可以改善GPU利用率，但是否加速应以实际吞吐为准。
