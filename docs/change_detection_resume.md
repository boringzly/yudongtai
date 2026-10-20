# 跨 Pod / 跨任务变化检测续跑

本文说明**批量变化检测的逐影像对恢复**。分类现在也支持独立跨任务续跑，见 [分类续跑与 HAMi 保护](classification_resume.md)。分类参数不继承变化检测的 `resume_from`，默认仍重新分类。
不要在旧任务仍写出结果时启动恢复；先保留旧结果/诊断日志，按平台正常方式停止旧任务。
本功能不会删除旧任务、修改 HAMi、清理 `/tmp/cudevshr.cache` 或自动寻找任意历史任务。

## 从本次旧任务 7052 恢复

更新服务器代码，并使用仓库根目录 `change_detection_wf.json` 的工作流定义。
工作流引擎 6.1 不用输入声明中的 `default` 补齐环境变量，因此续跑设置
**不能放在 `input.parameters` 或步骤的 `parameters.input` 中**。
算法直接读取已有 `WF_CONFIG` 环境变量的 `config` 字段，无需修改 Kubernetes YAML。
新任务将工作流顶层 `config` 改为：

```json
{
  "config": {
    "fast_run": false,
    "resume_from": "7052",
    "resume_allow_legacy": true
  }
}
```

这只是顶层 `config` 片段，不是完整工作流替代配置。
正常新跑用 `resume_from: ""`、`resume_allow_legacy: false`。
完整旧任务恢复示例见 `change_detection_wf_resume_7052.json`。
必须同时部署本次入口修复；仅使用旧版 12f5147 不会读取这些 config 字段。
算法仍兼容独立同名环境变量，以及 `CHANGE_DETECTION_RESUME_FROM` / `CHANGE_DETECTION_RESUME_ALLOW_LEGACY`。
CLI 对应 `--resume_from 7052 --resume_allow_legacy true`。
优先级为 CLI > 独立环境变量 > WF_CONFIG.config > 关闭默认值。
本平台无需设置额外环境变量。不要保留旧版两处外部续跑参数声明，否则工作流引擎
会在执行算法入口之前失败。不要在宿主机简单 `export` 后以为 Pod 会继承。
启动日志应出现 `[续跑配置] resume_from='7052', resume_allow_legacy=True`。

默认两个参数为空字符串、false，不会自动跨任务复用。
7052 的旧版没有完成清单，因此必须显式允许 legacy 导入，并由操作者确认前后影像、
模型权重及影响预测的配置/算法与原任务一致。日志只能验证路径/模型文件名/模式和完成记录，
**不能验证历史文件内容没被原地替换**。无法确认时应关闭旧结果导入、重新计算。

同项目的目录示例：

```text
/share/ib/3/332/output/7052/working/change_detection/shp/   旧结果，仅只读
/share/ib/3/332/task_diagnostics/7052/logs/                旧逐卡日志
/share/ib/3/332/output/NEW/working/change_detection/shp/   新任务结果
/share/ib/3/332/task_diagnostics/NEW/logs/
    change_detection_resume.json                        启动校验报告
    change_detection_checkpoints/<hash>.json             每对完成凭据
    change_detection_summary.json                       批次结束统计
```

若新旧任务不在同一个 `output` 父目录，`resume_from` 改为 Pod 内可访问的绝对路径：
`/share/ib/3/332/output/7052` 或 `/share/ib/3/332/output/7052/working/change_detection`。
任务 ID 只在新任务同级目录中解析，不扫描其他项目。
旧诊断目录按原标准 `task_diagnostics/<task>/logs` 定位；使用过自定义
`CD_ARTIFACTS_ROOT` 的旧任务不属于该自动定位布局，不要静默换用其他任务日志。

## 如何确认续上

新任务日志会出现：

```text
[续跑] 已恢复 (...) ...，图斑数=...
[续跑] 不复用 ...: 具体原因
[续跑] 校验完成：总计=10466，已恢复=N，待计算=10466-N；源=7052
```

7052 本地提供的日志解析到 7969 对成功记录。实际恢复数取决于服务器旧 SHP 是否仍完整；
不能预先保证一定恢复 7969 对。将 `change_detection_resume.json` 和新主日志提供给维护人员核验。
Kafka 会发送校验进度和已恢复/待计算数量，后续任务进度包含已恢复部分。
全部恢复时不启动 GPU 推理，仍生成新任务的输出数据集，随后正常分类。

## 校验与持久化

- 新版本：核对输入规范路径、大小、纳秒修改时间；主/第二权重 SHA-256、推理代码及相关配置文件哈希、batch；结果配套文件哈希、OGR 可读性、坐标系和图斑数。
- 输入影像没有全文件哈希，以避免再次读取全部大 TIFF；不能识别刻意保留大小与 mtime 的原地改写。输入应保持只读不可变。
- 旧版本：显式授权后，匹配最新一次逐图尝试的输入路径、模型文件名/模式及成功日志，再验证文件完整性；后一次未完成的尝试不能复用之前成功记录。
- `.shp/.shx/.dbf/.prj` 必须非零字节；有效的零图斑结果可复用。mask TIFF 不参与续跑，不需要恢复。
- 结果复制后再次校验，再通过临时文件、fsync、原子替换写入完成凭据；中途复制失败不能标记成功。
- 每对结果写完即落盘凭据，发生在清理 GPU 和回传主进程之前。同目录重启自动校验当前任务清单；跨任务仍需指定 `resume_from`。
- 新任务使用新目录内的结果；不会修改旧结果。同一新任务目录不要同时运行多个写入任务。
- legacy 导入来源标记为 `legacy_assumed_unchanged` 并保留到后续清单中，不伪装成完全验证过的历史计算。

## 本次稳定性调整

- HAMi 下 DataLoader 默认 `num_workers=0`，但仍保留每 GPU 一个计算进程、batch=4。此模式降低 CPU 数据加载并发，不承诺比健康的多 worker 更快，目标是避免反复 5 分钟锁超时。
- 显式 `CHANGE_DETECTION_DATA_WORKERS` 仍可覆盖默认值；当前 HAMi 故障环境不建议强制恢复多 worker。
- 多 worker 取批超时为 120 秒，worker 异常或超时后改为 0，并在同一 GPU 进程后续影像中保持该降级。
- 主调度每轮检查 GPU 子进程非零退出码和无进度时间，异常停止批次并保留完成结果，不继续冒充批次完成。
- `CHANGE_DETECTION_STALL_TIMEOUT` 默认 1800 秒（最低60秒），表示一个仍存活 GPU 进程没有向主进程上报事件的时长；复制/矢量化等无事件的长阶段也会计入，可按实际大图耗时调整。
- GPU 子进程每5秒检查父进程是否消失，避免父进程退出后长期成为孤儿。不能保证恢复所有驱动/内核/HAMi 死锁；已挂起的旧 Pod 不会因更新源代码自动恢复。

本地自动测试使用模拟 GPU/OGR 接口验证恢复和调度逻辑；真正的 Linux/GDAL/HAMi 双卡执行需新任务部署后验证。
