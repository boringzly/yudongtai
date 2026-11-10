# 分类输出掩膜预取优化（2026-11-02）

服务器实测日志中，切片循环的“结果整理及日志”占约 65%，单纯按批刷盘未消除这一瓶颈。本更新针对其中两条执行路径：避免 NumPy 预测数组通过 Torch 布尔掩膜进行索引；将原本在主线程逐切片进行的无效像素检查前移至现有 CPU 数据加载阶段。

## 执行逻辑

1. 分类主线程每批将 CPU 掩膜转换为 NumPy 视图，然后完成掩膜与类别处理。
2. 数据加载阶段根据原始像素，按现有写出窗口生成“所有所选波段等于 EXCEPTION_VALUE”的无效像素掩膜；在 CLAHE、镜像补边、归一化之前取原始像素。
3. 原始读图窗口覆盖完整写出窗口时复用已读取数组；否则用该工作线程的独立 GDAL 句柄读取原先的精确写出窗口。无法读取时仍报错，不默认为有效区域。
4. 边缘掩膜补齐为固定切片尺寸以支持批处理，主线程按实际写出尺寸裁切。无效像素仍在类别替换、ROI 掩膜及裁切之后清零，保留旧结果语义。
5. 现有 4 个 CPU 预取线程、有序且最多提前一批的加载方式、按批刷盘、最终写盘校验、结构化进度消息不变。未启用 CPU 预取的任务仍可同步加载，不创建额外 GPU 或子进程。
6. 追加“掩膜及类别处理”“无效像素检查”“日志及进度整理”计时。前移至加载阶段的检查耗时不属于主线程“无效像素检查”；若主线程等待其完成，会体现在“等数据”。应比较完整循环及收尾耗时，避免只比较某个被转移的计时字段。

不改变模型、输入归一化、切片顺序、下采样、阈值、ROI、类别映射规则。配置继续使用 batch=16、classification_prefetch_threads=4、classification_torch_threads=4。实际吞吐增益需要服务器同一影像、同一时相验证，本包不承诺固定提速比例。

## 安装

此包基于已安装的“分类线程预取”和“分类按批刷盘”版本，仅替换 3 个 Python 文件。无须编译 CIS 或替换模型。默认只检查；脚本拒绝未知版本，原子替换、自动备份，失败回滚，并继承准确的代码指纹兼容链。不要手工单独覆盖文件或删除兼容元数据。

```bash
OUTPUT_UPDATE=$(mktemp -d /tmp/classification-output.XXXXXX)
unzip /home/user/jars/算法/classification-output-prefetch-minimal.zip -d "$OUTPUT_UPDATE"
python3 "$OUTPUT_UPDATE/update.py"

# 相关任务已结束或通过原页面停止后再执行：
python3 "$OUTPUT_UPDATE/update.py" --apply
```

禁止运行中的共享算法目录热替换；后续分类子进程会重新加载代码。先停止/完成旧任务，再更新，再新建任务。

若要继续当前任务 7102，将下一次任务的两个续跑源均设置为 `/share/ib/3/332/output/7102`，`resume_allow_legacy` 和 `classification_resume_allow_legacy` 均保留 false，并保持输入、模型、ROI、batch=16 不变。只有有效完成清单与结果完整性校验通过的影像才能复用。完成后清空工作流模板中的续跑路径，避免新任务误用。

本包不修改变化检测 ROI 几何求交逻辑，之前失败的影像仍需另行处理。

## 验证

新任务实际开始分类后应看到：

```text
[分类结果整理配置] cpu_mask=numpy, exception_mask=prefetch, raw_window_reuse=true
[分类数据供给] batch_size=16, cpu_prefetch_threads=4, torch_cpu_threads=4, ...
[分类写出配置] flush_mode=batch, tile_log=per_batch, batch_size=16
[分类效率] ... 整理细分: 掩膜及类别处理=..., 无效像素检查=..., 日志及进度整理=...
```

```bash
POD=替换成新Pod名称
kubectl -n ib-theme logs "$POD" --all-containers=true \
  | grep -E '\[分类结果整理配置\]|\[分类数据供给\]|\[分类写出配置\]|\[分类效率\]|Traceback|ERROR' \
  | tail -n 20
```

本地检查覆盖原始窗口复用、边界回退、uint8/uint16/float32、全波段比较、NaN、读图失败、CLAHE 前原始像素、预测输入不变，及真实新旧分类循环的 ROI/无效状态/类别替换/重叠裁切/尾批/写出像素/坐标/进度一致性。GIS、模型和 CUDA 使用替身；生产环境性能与真实 GDAL/CUDA 尚待验证。
