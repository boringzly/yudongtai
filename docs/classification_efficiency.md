# 分类线程预取更新（2026-11-02）

本包只更新 Python 算法，不需要重新编译 CIS，也不包含模型权重。当前适用单张 A100 80GB、HAMi、分类每卡 batch=16 的运行方式。

## 改动

- 新增 `classification_prefetch_threads`（0～8，0 禁用）和 `classification_torch_threads`（1～256，实际不超过 CPU 配额）。环境变量分别为 `CLIE_PREFETCH_THREADS`、`CLIE_TORCH_THREADS`，优先级高于工作流配置。
- 未配置时，HAMi 默认使用 2 个 CPU 预取线程，普通环境保留原 DataLoader 行为；PyTorch CPU 运算线程默认 4。建议本次明确配置 4/4。
- 预取线程只读取、归一化切片，各自打开独立影像、mask 和 ROI 句柄；保持原切片顺序，最多提前准备一批。模型和 CUDA 调用保持在主线程，不创建 CPU 加载子进程。
- 使用预取时，将 GDAL 全局缓存上限限制为不超过 512MiB，保留已经更低的缓存上限。预取仍会增加切片缓冲内存，应观察实际峰值。
- 不改变分类模型、下采样方式、阈值、掩膜、输出坐标和写出顺序。新增 `[分类数据供给]` 和 `[分类效率]` 日志，首批、每 30 秒以及结束时报告耗时。
- 耗时阶段为“等数据”“CPU整理及缩放”“预测含传输及恢复”“写出及消息”。这些是切片循环墙钟耗时，并非纯 GPU kernel 时间；模型加载、整图准备及最终概览/矢量统计不计入此循环。

## Batch 建议

继续正在进行的任务时保留 `classification_batch_per_gpu=16`。此次先改善数据供给，用优化前约 5.9 切片/秒作为对照，不承诺固定提速比例。显存余量充足并不证明 batch 是瓶颈。

优化后若日志显示“预测含传输及恢复”占主要时间，可以在独立小任务试 batch=32，比较吞吐及峰值内存；不直接采用 128。当前分类续跑校验包含 batch，16 改成 32 会拒绝复用 16 的完成凭据。

## 续跑兼容

必须使用本包 `update.py --apply` 更新，不能只手工覆盖几个 Python 文件。更新器在服务器读取本次变更前的完整分类代码指纹，备份后安装，再记录精确的前后指纹对应关系到：

`fenlei/classification_pipeline_compat.json`

只有当前完整代码指纹与记录一致，且旧凭据的指纹在该次转换的列表中，才允许这次性能更新复用旧结果。输入影像路径/大小/修改时间、模型、变化图斑内容、ROI、下采样、batch、分类 SHP 完整性/哈希仍然必须逐项相同。后续修改其他被校验的源文件或 helper 会使这条兼容关系失效。此文件是服务器本地产生的迁移元数据，不是关闭续跑校验。

## 安装

先让使用同一算法目录的任务结束，或按现有任务管理方式停止它们；运行中的分类进程不会热更新参数，后续 CLIE 子进程可能读取新文件，因此不要在活动任务中途替换共享代码。

```bash
CLASS_UPDATE=$(mktemp -d /tmp/classification-prefetch.XXXXXX)
unzip /home/user/jars/算法/classification-prefetch-minimal.zip -d "$CLASS_UPDATE"
python3 "$CLASS_UPDATE/update.py"
# 检查全部通过，且相关任务已结束/停止后：
python3 "$CLASS_UPDATE/update.py" --apply
```

更新器只接受检查过的版本，未知版本或混合更新状态直接拒绝；默认只检查；应用时保存时间戳备份，文件替换失败会回滚。

工作流保留原有其他配置，增加：

```json
"classification_batch_per_gpu": 16,
"classification_prefetch_threads": 4,
"classification_torch_threads": 4
```

从当前任务继续时，`resume_from` 和 `classification_resume_from` 指向需要复用的最新任务。例如 7098：

```json
"resume_from": "/share/ib/3/332/output/7098",
"resume_allow_legacy": false,
"classification_resume_from": "/share/ib/3/332/output/7098",
"classification_resume_allow_legacy": false
```

保留原来的输入、模型、ROI、batch 和对应诊断目录，且先解决之前的存储配额问题。新任务会校验并恢复有效组，未完成或校验失败组重算，不承诺从固定的序号开始。

```bash
POD=替换成新建的Pod名称
kubectl -n ib-theme logs "$POD" --all-containers=true \
  | grep -E '\[Batch配置\]|\[分类数据供给\]|\[分类效率\]|\[分类续跑\]' \
  | tail -n 25
```

预期可看到 `batch_size=16, cpu_prefetch_threads=4, torch_cpu_threads=4, num_workers=0, loader=threads`。

## 验证边界

本地验证线程顺序、并行预取、队列上限、异常释放、真实切片循环前后预测/掩膜/坐标/尾批/进度一致性（使用 CPU 数组和假模型），以及精确指纹迁移、校验拒绝和更新失败回滚。真实 GDAL、PyTorch/CUDA/A100 吞吐与整任务输出仍需服务器验证；本地没有测得生产提速比例。

## 按批刷盘增量更新

线程预取生效后的样例速度约 13.27 切片/秒，写出及消息占 73.3%。第二阶段将分类显式刷盘改为每批一次并细分输出计时，参见 [classification_write_batching.md](classification_write_batching.md)。保留 batch=16 和线程 4/4，使用增量包更新脚本继承续跑兼容链。
