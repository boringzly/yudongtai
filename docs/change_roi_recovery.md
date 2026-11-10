# 范围几何修复与变化检测后处理续跑

## 问题与修复

此前 548 对影像中部分影像反复出现 `TopologyException: side location conflict` / `Output geometry intersection failed`。模型推理已执行，但平滑结果与 ROI 求交失败，整对影像没有完成清单；本地暂存被清理后，下次只能重新运行模型。有效空 SHP 原本就可恢复，不能把缺少完成清单解释为没有变化。

本更新只修改变化检测的后处理及续跑逻辑，不调整模型、FFT、切片、batch、线程预取、分类掩膜优化及按批刷盘。

- 输出图斑求交前用 IsValid 检查；对无效几何调用 OGR MakeValid，保留 Polygon/MultiPolygon 部分，再与精确 ROI 求交。保留原要素属性；纯接触的线/点不作为面输出。无法修复、求交失败、写盘失败仍按失败处理，不静默丢弃，也不伪造成功记录。
- 输入 ROI 的校验规则保持严格；坐标转换后的 ROI 会再次检查，必要时修复并记录。日志 `[范围修复]` 给出修复次数、范围修复、保留和范围外移除数量。
- 矢量化及属性写出成功并关闭 OGR 句柄后，将原始 SHP 与完整配套文件复制到持久诊断目录。完成校验后最后原子发布 `stage=polygonized` 记录，不依赖临时目录或只有日志的成功判断。
- 恢复点记录输入文件路径/大小/mtime、模型、ROI、batch、代码指纹、原始 SHP 哈希、坐标系与要素数。恢复先校验；不匹配或损坏时不复用。
- 后处理在原始图斑的工作副本上进行；失败不改变恢复点。跨任务复用时先将恢复点带到当前任务，避免连续换任务后丢失来源。新任务只做填洞、平滑、范围裁切，不复制两期大影像，不调用模型。最终成功后才写整景完成清单。
- 当前实现恢复点在“矢量化完成”之后；在此之前崩溃，仍需重做推理与矢量化。恢复点不包含栅格 TIFF，后处理恢复不会补生原来的中间 TIFF；下游分类使用最终 SHP。

## 旧结果兼容边界

必须用本包 update.py 安装。脚本捕获服务器上准确的旧/新变化检测代码指纹，原子更新 `change/change_resume_compat.json`；同时更新分类兼容记录，因为分类代码指纹也包含共用的 `change/roi_scope.py`，并继承已有分类优化的精确兼容链。

已经成功且完整校验通过的旧结果原样复用，不追溯重裁，也不强制重跑之前成功的约 540 对。新的修复逻辑用于待计算/待后处理的影像。输入、模型、范围、batch、输出文件哈希和要素数仍须一致，不能靠允许任意旧版本或打开 legacy 跳过校验。

旧版本失败的 8 对影像没有本更新的持久恢复点，首次安装后仍需补算一次；如几何能修复，成功结果以后可正常复用。MakeValid 不能修复所有数据异常，不能承诺所有历史失败都自动成功。

## 安装

不需重编译 CIS，不需替换模型。服务器可处于已安装分类线程预取、按批刷盘，或进一步已安装分类输出掩膜预取的状态；分类代码不会被回退覆盖。未知或混合版本会拒绝安装，不使用强制覆盖。

变化检测调度文件接受此前已审阅的范围版、内存保护版两个准确版本；安装后保留内存保护和遥测逻辑。

先检查，不修改文件：

```bash
RECOVERY_UPDATE=$(mktemp -d /tmp/change-roi-recovery.XXXXXX)
unzip /home/user/jars/算法/change-roi-recovery-minimal.zip -d "$RECOVERY_UPDATE"
python3 "$RECOVERY_UPDATE/update.py"
```

相关任务已完成或通过原页面停止后，再应用：

```bash
python3 "$RECOVERY_UPDATE/update.py" --apply
```

共享算法目录不能在任务运行时热替换；后续子进程会重新读取代码。脚本会保存时间戳备份，替换或兼容元数据写入失败会回滚。

需要继续最近已知任务 7102 时，保留原输入、ROI、模型、batch=16、线程设置，临时填写：

```json
"resume_from": "/share/ib/3/332/output/7102",
"resume_allow_legacy": false,
"classification_resume_from": "/share/ib/3/332/output/7102",
"classification_resume_allow_legacy": false
```

若已有更新的任务，应使用最新任务输出目录。新建续跑 Pod 后即可把后台工作流模板中的两个续跑路径恢复为空；已创建 Pod 的 WF_CONFIG 环境变量不随模板修改。

## 验证

```bash
POD=替换成新Pod名称
kubectl -n ib-theme logs "$POD" --all-containers=true \
  | grep -E '\[续跑\] 校验完成|\[后处理续跑\]|\[范围修复\]|处理 .*完成|失败|TopologyException' \
  | tail -n 35
```

首次补算时看：

```text
[后处理续跑] 已保存原始矢量恢复点 ...
[范围修复] ... 几何修复次数=... 保留=...
处理 ... 完成：检测到 ... 个变化图斑
```

以后若在后处理期间中断，恢复时看：

```text
[后处理续跑] 已将 ... 的恢复点保存到当前任务
[后处理续跑] 复用 ... 的原始矢量，仅执行填洞、平滑和范围裁切，不重跑模型
```

恢复点位于容器 `/share/ib/3/332/task_diagnostics/<任务ID>/logs/change_detection_postprocess/`；原最终完成记录仍在同级 `change_detection_checkpoints/`，不要删除这些恢复所需文件。

本包附带 tests/test_change_roi_recovery.py。新 Pod 中可用实际算法环境验证，包括真实 OGR 自相交多边形修复测试：

```bash
kubectl -n ib-theme exec "$POD" -- \
  /root/miniconda/bin/conda run --no-capture-output -n clcd_2 \
  python -m unittest discover \
  -s /app/module/cd_seg_workflow/tests -p test_change_roi_recovery.py -v
```

本地没有真实 GDAL/OGR/CUDA，几何流程和工作进程使用替身验证。2026-11-03 用户在服务器 clcd_2 Conda 环境回传 14 项全部通过，包括真实 OGR 自相交多边形修复、属性及范围校验。测试的 Self-intersection 警告由刻意构造的无效几何产生。直接调用环境 Python 曾因找不到 proj.db 失败，验证应通过上述 conda run 加载环境。历史失败影像的实际修复与完整跨 Pod 恢复仍待任务日志验收。失败处理、文件哈希、阶段记录、复制中断、跨任务恢复、空结果、严格兼容校验、安装回滚均有离线检查。
