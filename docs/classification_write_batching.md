# 分类按批刷盘更新（2026-11-02）

这是上一版分类线程预取包之后的增量更新。仅替换 3 个 Python 文件：`fenlei/classification_pipeline.py`、`fenlei/clie_new/CLInferEngine/clie_lib/imageio.py`、`fenlei/clie_new/CLInferEngine/clie_lib/run_lib.py`。不需要重编译 CIS，不包含模型，不修改工作流 JSON。

## 行为

- 分类的 full/single/color 三种 TIFF 写出器在每个切片 WriteArray 后检查即时错误；一批全部切片写完后统一执行 Dataset.FlushCache，并检查返回值及 GDAL 错误状态。
- 尾批同样刷盘；影像结束 close 时仍执行最终刷盘校验。写入、批次刷盘、最终刷盘的配额/磁盘错误继续作为失败上抛，不能生成有效完成凭据。
- ImageWriter 的其他调用方默认仍逐切片刷盘；仅本分类循环显式关闭逐切片刷盘。
- batch=16 时，显式刷盘由每切片一次改为每批一次，不代表实际磁盘 I/O 必然减少 16 倍；GDAL 仍可能因缓存压力自动写出。
- 每切片的 Process 信息日志缩减为每批一条（保留该批最后切片位置），原结构化进度消息发送内容及其节流逻辑不变；可选的逐块输出功能不变。
- 保留原“写出及消息”汇总计时，增加“结果整理及日志”“TIFF读写及刷盘”“消息发送”细分，不将全部时间归因于磁盘。
- 模型、切片顺序、预测/掩膜/裁剪逻辑、前后时相类别、batch 和线程设置不变。刷盘不使用额外写线程，也不增加无界缓存。

启动预测时可见：

```text
[分类写出配置] flush_mode=batch, tile_log=per_batch, batch_size=16
```

## 安装与续跑

当前任务结束或通过原管理页面停止后应用，避免共享代码被后续 CLIE 子进程中途读取。默认命令只检查版本，不修改文件：

```bash
WRITE_UPDATE=$(mktemp -d /tmp/classification-write.XXXXXX)
unzip /home/user/jars/算法/classification-write-batching-minimal.zip -d "$WRITE_UPDATE"
python3 "$WRITE_UPDATE/update.py"
# 检查通过且相关任务已结束/停止后：
python3 "$WRITE_UPDATE/update.py" --apply
```

必须使用更新脚本，不要单独手工覆盖 Python 文件。脚本校验版本、保存时间戳备份、原子替换，失败回滚；它记录准确的旧/新代码指纹，并继承上一版仍然有效的兼容链。因此可以继续校验并复用线程预取之前和之后的已完成结果。输入、模型、ROI、下采样、batch、结果完整性及内容哈希仍然严格验证。

工作流无需增加配置，保留：

```json
"classification_batch_per_gpu": 16,
"classification_prefetch_threads": 4,
"classification_torch_threads": 4
```

重新启动解译时，两个续跑源都指向准备复用的最新任务，例如目前的 7101：

```json
"resume_from": "/share/ib/3/332/output/7101",
"resume_allow_legacy": false,
"classification_resume_from": "/share/ib/3/332/output/7101",
"classification_resume_allow_legacy": false
```

```bash
POD=替换成新Pod名称
kubectl -n ib-theme logs "$POD" --all-containers=true \
  | grep -E '\[分类续跑\]|\[分类数据供给\]|\[分类写出配置\]|\[分类效率\]' \
  | tail -n 25
```

以生产日志优化前样例约 13.27 切片/秒、写出及消息占 73.3% 作为线索。必须在相同影像和时相下比较完整循环与收尾耗时，才能判断实际提速；不能把不同影像间速度差直接作为性能结论。

本地测试覆盖真实写出器的批次/尾批/最终刷盘错误、RGB 波段映射及重叠区域像素一致性（GDAL 替身），真实新旧分类循环的预测数组/掩膜/坐标/尾批/进度消息一致性，精确续跑兼容链、未知代码拒绝和安装失败回滚。本地缺少真实 GDAL/CUDA，服务器写盘与吞吐效果尚待验证。
