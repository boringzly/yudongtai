# 2026-10-30 算法更新说明

本文对应 `D:/code/yudongtai` 本次待提交的代码差异。CIS Java、Vue 页面与 `ib` 服务管理脚本位于其他项目，不属于本仓库提交。

## 更新目的

解决 HAMi 环境下变化检测数据供给串行、无效区域参与切片准备、分类统计重复读取整图的问题，并增加批次范围约束与 RAM 观测。

本次保留模型权重、有效切片的 FFT 实现、1280×1280 切片、阈值、类别映射及原有推理精度。没有引入 AMP、自动增大 batch、自动 OOM 降 batch 或修改剩余时间估算公式。

## 主要代码变化

| 方向 | 文件 | 行为 |
| --- | --- | --- |
| CPU 数据供给 | `change/inference_pipeline.py`、`change/test_lib_batch_memeff_single_image_nomp.py` | 有界线程预取、独立 GDAL 读取句柄、按原顺序返回批次；减少掩膜扫描临时内存及批次图像复制 |
| 范围工具 | `change/roi_scope.py`、`change/main.py` | 接收范围文件，检查几何/坐标系，重投影、切片相交筛选、窗口级掩膜、最终矢量裁切 |
| 分类范围 | `fenlei/clie_new/CLInferEngine/clie_lib/dataset.py`、`run_lib.py` | 读图前筛选切片，保留原行列索引与 RGB 上下文，工作进程独立重建范围对象，输出范围统计 |
| 分类图斑统计 | `fenlei/classification_core.py` | rasterstats 直接接收 TIFF 路径，按图斑窗口读取，避免统计进程各自复制整幅分类栅格 |
| 续跑签名扩展 | `change/resume_checkpoint.py`、`fenlei/classification_resume.py`、`change/change_detection_core.py` | ROI 内容和相关推理模块参与身份校验；启用 ROI 时拒绝无法验证范围的 legacy 导入 |
| 参数配置 | `workflow_batch_config.py`、三个工作流 JSON 模板 | 支持从工作流 config 设置预取与 PyTorch CPU 线程数，输出配置来源 |
| RAM 保护与采样 | `change/change_detection_core.py` | 变化检测每幅图推理前设置默认 512MiB GDAL 块缓存；记录进程 RSS/峰值、cgroup 内存用量/限额、GDAL 缓存 |

## 更新后的执行逻辑

### 1. 读取任务参数

- 工作流从 `WF_CONFIG.config` 读取 batch、线程和续跑设置。
- batch 仍按已有 CLI > 环境变量 > 工作流 config > 默认值规则解析。
- 预取/PyTorch CPU 线程参数以显式环境变量优先于工作流 config。
- 可选 ROI 来源为 `CHANGE_DETECTION_ROI_WKT_FILE`，或者 `CHANGE_DETECTION_ROI_VECTOR` / `--roi_vector`。
- WKT 文件按 EPSG:4326 解释；矢量文件必须带可用 CRS。SHP 需提供必需配套文件。
- 算法不直接查询 `lin_st_change`，也不自己从任务 ID 生成批次范围；这些工作由 CIS 后端完成。

### 2. 在已有续跑框架中逐幅校验

- 原有框架按影像对建立检查点，本次补入 ROI 内容与共享推理模块校验。
- 输入路径、文件大小/mtime、模型、算法、batch、ROI 和结果配套文件等必须匹配。
- 通过校验的结果复制至新任务目录；未通过的影像进入重算队列。
- 进度日志保留原始影像编号，处理第 20 幅不意味着第 21～33 幅也会重算。
- 中断影像整幅重算，未实现切片级续跑。
- 启用 ROI 时保持 `resume_allow_legacy=false`，不能靠旧日志绕过范围一致性校验。

### 3. 变化检测推理

```text
读取范围并转换到实际工作栅格 CRS
    ↓
生成原切片索引，排除完全不相交的切片
    ↓
线程预取剩余切片：读两期窗口 → 有效区域与范围掩膜 → 有效时执行 FFT → 归一化
    ↓
减少复制地组装有效六通道批次
    ↓
GPU 推理 → 按有效掩膜写 TIFF
    ↓
矢量化 → 填洞/平滑 → 按真实范围再次裁切 → 保存完成检查点
```

- 范围外切片在读图和 FFT 之前排除。
- 部分相交切片保留 RGB 边界上下文，只限制输出有效性。
- ROI 采用窗口级栅格化，保留孔洞和多面，不为大 TIFF 生成整图满分辨率范围掩膜。
- 没有有效像元的切片跳过 FFT 和模型推理。
- 不传范围时沿用整图模式；传入无效范围时明确失败，不静默改为整图。
- 范围过滤与线程预取同时生效，没有用 ROI 覆盖原性能优化。

HAMi + GPU、`num_workers=0` 时默认使用 2 个 CPU 预取线程；任务显式设置 4 时使用 4。
不创建 DataLoader 子进程，外层原有“每 GPU 一个影像工作进程”继续存在。
线程模式与多进程 DataLoader 不叠加，预取队列有界，各线程持有独立 GDAL 句柄。

### 4. 分类与图斑统计

- 两期分类分别在实际工作栅格中使用同一份 ROI。
- 分类读图前排除范围外切片，边界输出应用范围掩膜；不改变归一化、上下文、下采样或类别映射。
- DataLoader 进程重新打开自己的读图句柄和范围对象，避免跨进程传递 GDAL 几何句柄。
- 分类仍以变化检测最终图斑为统计对象，图斑几何已受 ROI 限制。
- `zonal_stats` 保持 `band=1`、`categorical=True`、`nodata=0`、`all_touched=False`，改为文件窗口读取。
- 原有整图准备、部分重采样、预览和影像本地暂存并未全部变为 ROI 小范围处理。

### 5. 内存与诊断

- 默认 GDAL 块缓存上限 512MiB，每个变化检测工作进程分别设置，不等于整个 Pod 只用 512MiB。
- 可设置 `config.change_gdal_cache_mb` 或优先环境变量 `CHANGE_DETECTION_GDAL_CACHE_MB`，范围 64～4096 MiB。
- `[内存配置]` 记录缓存限额；`[内存]` 在影像开始、每约 30 秒进度回调、影像清理后采样。
- RAM 指标与 PyTorch 显存指标分开，采样异常不终止算法。
- 512MiB 是缓存约束，不是所有原生内存分配的总上限，尚不能据此保证不再 OOM。
- 调度层内存采样/缓存修补不参与现有算法源文件哈希，不为此次调度修改绕过或重建旧检查点身份。

## 配置与部署边界

### 普通新任务的运行配置示例

```json
{
  "fast_run": true,
  "change_batch_size": 16,
  "classification_batch_per_gpu": 16,
  "change_prefetch_threads": 4,
  "change_torch_threads": 4,
  "resume_from": "",
  "resume_allow_legacy": false,
  "classification_resume_from": "",
  "classification_resume_allow_legacy": false
}
```

此示例体现本次服务器任务设置，不代表修改仓库三个工作流模板原有 batch 或 fast_run 默认值。
模板本次仅补充线程配置，保留各自原有 batch 和调度设置。

续跑时只将 `resume_from` 指向同一输入/模型/范围的旧任务根目录，例如
`/share/ib/3/332/output/7094`；不要把生产任务路径作为通用模板默认值提交。

128Gi RAM 配额在 CIS 所选计算资源套餐或 Kubernetes 资源中设置，工作流 config 内添加 `memory` 不会改变 Pod 配额。
生产服务器部署算法更新后，用新建任务验证；运行中 Pod 不应被当作自动热更新后的验证对象。

## 本次服务器验证记录与限制

- 任务 7090：手绘范围的变化检测、分类均出现 `[范围]` 候选/保留/跳过日志。
- 任务 7094：批次范围文件已传入，变化检测范围过滤与 4 线程预取同时生效。
- 某幅图候选 2704、保留 2284；切片循环 452.90 秒，其中推理含传输 390.64 秒，约 86.3%；等数据 20.64 秒，约 4.6%。说明该样本瓶颈已主要转到推理阶段。
- 同图对照的两幅算法耗时分别从 31.56 秒降到 30.62 秒、163.05 秒降到 156.89 秒；这不是整批加速比例，也不含全部暂存/结果回传时间。
- 任务 7094 后续因 64Gi RAM 的 cgroup OOM 被杀，内核显示 Python 匿名 RSS 约 63.1GiB；不是 CUDA 显存不足的报错。
- 任务 7097：从 7094 恢复 32 幅。第 20 幅缺少完成清单，1～19 与 21～33 已恢复，因此先补第 20 幅，再处理第 34 幅及之后。
- 内存补丁已在本地准备并提供服务器安装包；当前会话尚无其真实服务器长时间运行的验证结果。
- 最终 SHP 边界、孔洞及分类像素一致性仍需在实际 GIS 环境叠加检查，不能只看范围日志就声称全部结果验证完成。
- 当前 ETA 仍按当前影像速度外推剩余影像，未按全部剩余 ROI 工作量预扫描；预计时间变化不代表实际变慢。

## 测试

- 原有 181 项回归：161 通过、20 因缺少 GIS/PyTorch 环境依赖跳过，无失败。
- 新增 10 项 RAM 缓存与采样专项检查，已纳入 `tests/test_change_memory_guard.py`。
- 跳过的 GIS、CUDA 和真实模型检查不计作通过；未证明整个 548 对生产任务已稳定完成。

## Git 提交范围

- 提交 `change/`、`fenlei/` 中本次算法修改、配置解析、三个工作流模板、相关测试与文档。
- `.idea/` 是本地 IDE 文件，不纳入本次提交。
- 模型权重与编译依赖不属于本次提交；现有 `.gitignore` 已忽略 `*.pth`、`*.pt`、`*.ckpt`、`*.so`。
- 本仓库不包含前后端的范围选择/批次查询适配，也不包含服务器 `ib` 管理脚本更新。
