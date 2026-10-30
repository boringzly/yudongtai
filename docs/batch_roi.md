批次实际落图范围优化
==================

本更新包含 CIS 后台源码与 Python 算法两部分。不是可直接替换后端 JAR 的二进制包。

后台：
- STTilesDownloadTaskStartExecutor 优先使用启动请求 wkt；未传时按 baseStChangeId 从
  lin_st_change.geom 转换为 EPSG:4326 WKT。不使用外接矩形替代推理范围。
- 范围作为 StartInferTaskRequest.params 保存到 task_request，远程工作流恢复后仍可使用。
- DockerContainerAlgorithmTools 将大 WKT 写入已挂载的任务 working/inputs/batch_roi.wkt，
  仅把 CHANGE_DETECTION_ROI_WKT_FILE 路径传入 Pod，避免单个环境变量长度限制。
- 参数名 CHANGE_DETECTION_ROI_WKT 为内部任务参数；不要将其注册成第三个文件目录输入，
  避免影响原有前后影像 dataList 的位置映射。

算法：
- 可手动通过 --roi_vector 或 CHANGE_DETECTION_ROI_VECTOR 传入带 CRS 的单图层矢量。
  SHP 必须提供 .shp/.shx/.dbf/.prj。后台自动传入的 WKT 文件约定为 EPSG:4326。
- 多边形按实际工作栅格 CRS 转换，筛掉完全不相交的切片，然后才执行影像读取与 FFT。
- 边界窗口只修改有效像元掩膜，不将上下文 RGB 提前清零。窗口级栅格化保留孔洞和多面，
  不为大 TIFF 创建整幅满分辨率掩膜。
- 填洞/平滑后再按真实范围裁剪最终矢量，避免后处理把图斑扩张到范围外或填平范围孔洞。
- ROI 内容/坐标系侧文件参与续跑签名；启用范围时禁止无法验证范围的 legacy 结果导入。
- 无范围参数时保持整图模式；传了无效、缺失或无 CRS 的范围则失败，不静默退回整图。
- 旧 TIFF 与模型权重保留，无需重新生成 TIFF。分类阶段也在读图前按同一批次范围筛片，
  最终图斑范围继承变化检测结果；图斑统计改为文件窗口读取。

发布：
1. 先在算法实际 Python/GDAL 环境运行 tests/test_roi_scope.py，必须执行 GIS 用例，不能
   仅看到“跳过”就当作 GIS 验证通过。再用一对影像验证边界、孔洞、坐标系与字段。
2. 当前任务完成后，在算法根目录更新包内 payload 文件（配套 update.py 先校验再 --apply）。
3. 构建并发布 CIS 后台，部署对应 Java 适配（包括已有影像直接启动入口）。已经做过基于现有依赖的 javac 检查，
   但仍需按项目正常构建/发布流程生成后端产物；不要把源码当作 JAR 热更新。
4. 新建小样本批次任务，启动请求 wkt 是实际落图范围（EPSG:4326），或由批次表读取。
5. 检查 [范围] 日志中的候选/保留/范围外跳过数。进度分母变为保留切片数，因此对比新旧
   性能时看同一对影像的总耗时，不能只对比切片每秒值。

当前本地缺少 GDAL/OGR：几何交集、重投影、窗口栅格化和最终矢量裁剪用例会显式跳过。
本地未直连生产数据库/K3s。后续服务器日志已确认手绘与批次范围的切片过滤、分类范围过滤生效；
最终边界/孔洞/分类像素与整批稳定性仍需验证，见 [2026-10-30 更新说明](update_notes_20261030.md)。


分类补充优化
------------
- 前后时相分别在实际工作栅格（包括 GF 影像既有 2 米重采样后）应用同一份范围。
- 保留原切片行列号、边界上下文、下采样、TTA、模型和类别映射。范围只参与切片筛选与输出掩膜。
- GDAL 范围对象不跨进程序列化；DataLoader worker 重新创建自己的读取句柄和几何对象。
- 分类 checkpoint 包含范围内容和共享 ROI 实现，禁止用无法验证范围的 legacy 分类结果续跑。
- zonal_stats 接收 TIFF 文件路径而非整图数组，按图斑窗口读取；保留原几何、band=1、
  categorical=True、nodata=0 和 all_touched=False。
- 未配置 ROI 时分类继续整图模式。已有整图重采样/结果预览步骤未在本次缩减。
- 先在部署环境用相同影像/图斑对比输出类别及范围内分类像素；本地缺少 PyTorch/GDAL，
  不能把本地单元测试等同于真实模型/GIS端到端验证。
- 分类日志新增“[范围] 分类候选切片/保留/范围外跳过”，可核对 ROI 已应用。

rasterstats 文件和窗口接口：https://pythonhosted.org/rasterstats/manual.html
