# 分类 HAMi 保护和跨任务续跑

## 工作流配置（不需要改 Kubernetes YAML）

在原工作流顶层 `config` 增加分类的两个参数，不要加到 `input.parameters` 或步骤输入映射。
引擎 6.1 不会将外部参数 default 自动变为环境变量；分类入口直接读取 `WF_CONFIG.config`。

例如从旧任务 7057 恢复变化检测和分类：

```json
"config": {
    "fast_run": false,
    "resume_from": "7057",
    "resume_allow_legacy": true,
    "classification_resume_from": "7057",
    "classification_resume_allow_legacy": true
}
```

两种算法的源可不同。分类不续跑就设 `classification_resume_from: ""`、
`classification_resume_allow_legacy: false`。新版本产生了完成清单的任务无需开启 legacy。
普通完整模板 `change_detection_wf.json` 默认关闭两种跨任务续跑。

分类源支持同项目任务 ID、旧任务根绝对目录，或旧分类步骤目录（必须包含 `shp_cls`）。
标准路径为 `/share/ib/3/332/output/7057/working/change_classification/shp_cls`。
源目录必须仍挂载可读，不能把旧源指定为新任务输出目录；不要并发写同一任务目录。
旧任务采用自定义 `CD_ARTIFACTS_ROOT` 时，自动定位旧日志不适用，不能保证导入。

优先级：显式 CLI（包含空字符串/false）> 环境变量 > 工作流 config > 默认关闭。
CLI 为 `--classification_resume_from`、`--classification_resume_allow_legacy`；环境变量为
同名小写或 `CLASSIFICATION_RESUME_FROM`、`CLASSIFICATION_RESUME_ALLOW_LEGACY`。

## 恢复粒度与校验

- 每一组前后影像完成分类、统计、SHP 写出后保存原子完成清单，含有效空 SHP。
- 新任务逐组校验，成功就复制 SHP 配套文件到新目录并跳过 GPU；未完成的一组重跑。
  不支持接着某个 TIFF 的中间瓦片或只完成后时相的半组续跑。
- 清单核对前后影像路径/大小/mtime、模型 SHA-256、算法代码与配置哈希、下采样、batch、
  过滤参数、变化 SHP 内容哈希，以及分类 SHP 配套文件哈希、坐标系、字段与图斑数。
- 输入影像未全量哈希；需保持输入、权重和代码在运行期间只读，不能识别保留大小和 mtime 的影像修改。
- legacy 只在显式开启时导入。要求最新逐组尝试有完成日志，输入路径和模型名匹配，
  前后两次下采样记录匹配，旧变化 SHP 与本次内容相同，分类 SHP 可读且字段完整。
  空结果另外检查旧变化 SHP 也为空。后续未完成的重跑会覆盖旧成功日志资格。
- 旧日志无法证明当时权重/影像/代码内容或 batch，开启 legacy 前必须人工确认算法、映射规则、
  模型、输入未变。导入清单始终保留 `legacy_assumed_unchanged` 标记，不能视为严格历史验证。
- 无清单、未授权 legacy、签名变化或损坏结果不会盲目复用；日志解释原因并重新计算。
- 最终合并、匹配省名、发布数据集仍执行；不会删除或修改旧任务分类结果。
  已恢复组不为预览重跑推理，抽样图只选本次新算的组，全恢复时不生成新的抽样图。

## 在哪里查看

诊断文件不放入入库结果目录，统一在 `task_diagnostics/<新任务ID>/logs/`：

- `classification.log`：`[分类续跑] 已恢复 ...`、不复用原因及累计数量。
- `classification_checkpoints/*.json`：每组成功后立即落盘的凭据；跨 Pod 持久保存。
- `classification_resume.json`：批次推理阶段结束或保护中断时的恢复计数。
- `classification_failure.json`：失败堆栈；`classification_summary.json`：最终统计。

## HAMi 保护

- 检测 libvgpu/HAMi 注入标记或共享缓存后，分类 DataLoader 强制 `data_workers=0`，
  即使设置了 `CLIE_DATA_WORKERS=8` 也覆盖为 0。分区统计也不再建立 CPU 进程池。
  不移除 LD_PRELOAD、不绕过平台显存/算力限制。
- 双卡 DataParallel、每卡 batch=8、1024 切片缩小到 512 输入模型保持不变。
  普通无 HAMi 环境仍可多 worker（取批超时120秒），分区统计使用 spawn、超时1800秒。
- CLIE 标准输出和错误实时记入日志；默认无输出1800秒或单次 CLIE 运行7200秒触发停止。
  可通过 `CLIE_STALL_TIMEOUT`、`CLIE_PROCESS_TIMEOUT` 调整（单位秒，正数）。
  无输出不一定是死锁，超大影像的正常长阶段也可能触发，需据日志调整。
- HAMi ERROR、DataLoader worker 退出/超时、CUDA OOM、子进程非零退出会停止批次，
  发送失败状态并保留已完成凭据，不持续启动后续影像制造重复故障。
- Linux 下 CLIE 独立进程组，保护退出清理自己启动的子进程组，不影响其他任务。
  SIGKILL/OOM 杀父进程、内核不可中断 I/O 或驱动故障不能保证被应用代码修复。
- 可选内部 mask 文件不存在时不再反复让 GDAL 打 ERROR 4；这与输入变化 SHP 缺失不同。

更新代码不会修复已启动的旧进程；需新建任务验证实际 HAMi 双卡运行。
