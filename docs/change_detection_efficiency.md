# 变化检测 batch 与效率日志

默认每张 GPU 一个进程、处理一对整图，切片仍为 1280×1280，
现在每批最多推理 **4 对切片**。分类参数不变。单/双 PTH 均使用此批量。

现在也可直接设置工作流 `config.change_batch_size`，详见 [工作流 batch 配置](workflow_batch_config.md)。
可通过变化检测进程的环境变量 `CHANGE_DETECTION_BATCH_SIZE=1` 回退，
或设置为 2、4 等作对照；支持 1～32 的整数，缺省或非法值回退到 4。
它不是每卡进程数。较大 batch 同时增加 CPU 预取/共享内存及显存压力，
4 不保证适配所有显卡或一定更快；没有新增自动 OOM 降批量重试。

搜索现有 `task_diagnostics/<任务>/logs/change_detection_gpu_*.log`
（单图为对应变化检测日志）中的 `[效率]`，不向结果输出目录增加文件。
首批完成、此后每约 30 秒在批次完成时、循环结束各打印统计。
若一个批次等待很久，不会每 30 秒强行唤醒或打印假进度。

- 已处理数包含无效切片；有效推理数只包含实际送模型的切片，两种吞吐分别列出。
- 平均/近期吞吐是切片循环的墙钟速度，包含等待和写出，不是每秒批次数。
- 等数据：主进程等 DataLoader 的暴露等待，包含 worker 启动/结束。
  CPU FFT 在 worker 中运行并与推理重叠，**该时间不是 FFT 总工作时间**。
- CPU 整理：有效像元筛选、批量数组组装。
- 推理含传输：CPU/GPU 拷贝、模型执行、结果回传及双 PTH OR；开启 TTA 时含增强。
  现有推理返回 CPU 数组，计时点已等到 GPU 结果，不增加 CUDA 全设备同步。
- 写 TIFF：掩膜、裁切、写入及每片 FlushCache；保持既有写盘行为。
- PyTorch allocated/reserved 峰值：每对影像加载模型前重置，不等于 nvidia-smi
  的整卡占用（不统计所有 CUDA 库、上下文及其他进程）。读统计失败则显示 N/A。
- TIFF 概览、矢量化、填洞平滑单列耗时；单对算法总耗时包含模型加载和后处理，
  不含外层本地暂存、结果回传及多影像最终合并。

优先对比同样影像及单/双模型设置下的有效吞吐和单对总耗时。
计时阶段不含全部日志/消息开销，且后台预取有重叠，不应当作 CPU/GPU 利用率。


## HAMi 数据供给优化

HAMi + GPU 且 `num_workers=0` 时，默认使用两个 CPU 线程读取和预处理切片。
各线程独立创建 OnlineDataset/GDAL 句柄，不创建子进程、不在加载线程调用 CUDA。
保持切片顺序和 batch 边界，最多缓存下一批切片；异常和提前退出会等待读取结束后关闭句柄。
FFT 算法、有效切片的归一化、模型、切片尺寸、阈值与输出网格保持原设置。
无有效像元的切片跳过 FFT；其模型推理原本也会被跳过。

主进程用 NumPy 扫描掩膜和直接组装六通道数组，避免掩膜求和的 int64 临时数组、
前后时相的大型布尔索引复制。GPU 模式默认 PyTorch CPU 线程数为 4，降低小型 CPU
操作的线程调度开销；CPU 模型推理默认线程数不受此项影响。

对照开关（必须设置在算法进程启动前，正在运行的 Pod 不会自动应用代码或配置）：

```text
CHANGE_DETECTION_PREFETCH_THREADS=2   HAMi GPU 默认；支持 0～8
CHANGE_DETECTION_PREFETCH_THREADS=0   关闭线程预取
CHANGE_DETECTION_TORCH_THREADS=4      GPU 模式下 CPU 运算线程数
```

普通多进程 DataLoader 不叠加线程预取。线程模式不锁页或后台传输 GPU 数据。
`DataLoader配置` 增加 `cpu_prefetch_threads`、`torch_cpu_threads`，线程模式显示
`start_method=threads`；`num_workers=0` 仍表示未创建子进程。
`等数据` 仍表示主循环暴露的等待，不等于后台线程全部读取/FFT 的工作量。

先用相同影像、模型与 batch=16 对照端到端耗时、等数据/CPU整理耗时、宿主内存及输出。
本地缺少 CUDA/GDAL 环境；服务器日志已确认 A100/HAMi 线程预取生效，并取得部分同图耗时对照，
详见 [2026-10-30 更新说明](update_notes_20261030.md)。不保证固定加速倍数，仍需验证最终输出。
新增推理模块已纳入续跑算法签名，旧签名结果不会自动按严格模式复用。
