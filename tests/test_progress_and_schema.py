import importlib.util
import ast
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeFuture:
    def get(self, timeout=None):
        return {"timeout": timeout}


class FakeProducer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = []
        self.closed = False
        self.__class__.instances.append(self)

    def send(self, topic, message):
        self.messages.append((topic, message))
        return FakeFuture()

    def flush(self, timeout=None):
        return None

    def close(self, timeout=None):
        self.closed = True


def load_sender_module(path, module_name, producer_class=FakeProducer):
    fake_kafka = types.ModuleType("kafka")
    fake_kafka.KafkaProducer = producer_class
    previous = sys.modules.get("kafka")
    sys.modules["kafka"] = fake_kafka
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("kafka", None)
        else:
            sys.modules["kafka"] = previous


def load_imageio_module(fake_gdal):
    fake_numpy = types.ModuleType("numpy")
    fake_osgeo = types.ModuleType("osgeo")
    fake_osgeo.gdal = fake_gdal
    fake_osgeo.ogr = types.SimpleNamespace()
    fake_osgeo.osr = types.SimpleNamespace()
    replacements = {"numpy": fake_numpy, "osgeo": fake_osgeo}
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        path = ROOT / "fenlei" / "clie_new" / "CLInferEngine" / "clie_lib" / "imageio.py"
        spec = importlib.util.spec_from_file_location("imageio_bigtiff_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class ClassificationSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "fenlei"))
        import classification_schema

        cls.schema = classification_schema

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / "fenlei"))

    def test_chinese_class_names(self):
        expected = {
            1: "耕地",
            2: "林地",
            3: "草地",
            4: "水域",
            5: "建设用地",
            6: "未利用土地",
        }
        self.assertEqual(
            {code: self.schema.get_class_name(code) for code in expected},
            expected,
        )
        self.assertEqual(self.schema.get_class_name(None), "未知")
        self.assertEqual(self.schema.get_class_name(99), "未知")

    def test_output_fields_are_shapefile_safe(self):
        self.assertEqual(
            self.schema.OUTPUT_FIELDS,
            ["uid", "pre_code", "pre_name", "curr_code", "curr_name", "geometry"],
        )
        self.assertTrue(all(len(name) <= 10 for name in self.schema.OUTPUT_FIELDS[:-1]))


class ProgressMessageSenderTests(unittest.TestCase):
    def test_all_senders_connect_lazily_and_confirm_delivery(self):
        sender_paths = [
            ROOT / "change" / "MessageClient" / "ProgressMessageSender.py",
            ROOT / "fenlei" / "MessageClient" / "ProgressMessageSender.py",
            ROOT / "fenlei" / "clie_new" / "MessageClient" / "ProgressMessageSender.py",
        ]

        for index, sender_path in enumerate(sender_paths):
            with self.subTest(sender=str(sender_path)):
                FakeProducer.instances.clear()
                module = load_sender_module(sender_path, f"progress_sender_{index}")
                sender = module.ProgressMessageSender("kafka:9092", "progress", "task-1")
                self.assertIsNone(sender.producer)
                self.assertFalse(sender.is_none())

                original = {"progress": 0, "runningStatus": "running", "runningInfo": "初始化"}
                self.assertTrue(sender.send(original))
                self.assertEqual(original, {"progress": 0, "runningStatus": "running", "runningInfo": "初始化"})

                producer = FakeProducer.instances[-1]
                self.assertEqual(producer.kwargs["acks"], "all")
                self.assertEqual(producer.messages[0][0], "progress")
                payload = json.loads(producer.messages[0][1].decode("utf-8"))
                self.assertEqual(payload["taskId"], "task-1")
                self.assertEqual(payload["messageContent"]["progress"], 0)

                sender.close()
                self.assertTrue(producer.closed)

    def test_sender_reconnects_after_transient_startup_failure(self):
        class FlakyProducer(FakeProducer):
            attempts = 0

            def __init__(self, **kwargs):
                self.__class__.attempts += 1
                if self.__class__.attempts == 1:
                    raise OSError("broker is starting")
                super().__init__(**kwargs)

        module = load_sender_module(
            ROOT / "change" / "MessageClient" / "ProgressMessageSender.py",
            "progress_sender_flaky",
            FlakyProducer,
        )
        sender = module.ProgressMessageSender("kafka:9092", "progress", "task-2")

        self.assertFalse(sender.send({"progress": 0}))
        sender._next_connect_time = 0
        self.assertTrue(sender.send({"progress": 1}))
        producer = FlakyProducer.instances[-1]
        delivered_progress = [
            json.loads(message.decode("utf-8"))["messageContent"]["progress"]
            for _, message in producer.messages
        ]
        self.assertEqual(delivered_progress, [0, 1])

    def test_close_reconnects_and_delivers_latest_pending_status(self):
        sender_paths = [
            ROOT / "change" / "MessageClient" / "ProgressMessageSender.py",
            ROOT / "fenlei" / "MessageClient" / "ProgressMessageSender.py",
            ROOT / "fenlei" / "clie_new" / "MessageClient" / "ProgressMessageSender.py",
        ]

        for index, sender_path in enumerate(sender_paths):
            with self.subTest(sender=str(sender_path)):
                class RecoverOnCloseProducer(FakeProducer):
                    attempts = 0

                    def __init__(self, **kwargs):
                        self.__class__.attempts += 1
                        if self.__class__.attempts == 1:
                            raise OSError("broker temporarily unavailable")
                        super().__init__(**kwargs)

                RecoverOnCloseProducer.instances.clear()
                module = load_sender_module(
                    sender_path,
                    f"progress_sender_close_retry_{index}",
                    RecoverOnCloseProducer,
                )
                sender = module.ProgressMessageSender("kafka:9092", "progress", "task-close")
                self.assertFalse(sender.send({"progress": 0, "runningStatus": "running"}))
                self.assertFalse(sender.send({"progress": 100, "runningStatus": "completed"}))

                sender.close()

                producer = RecoverOnCloseProducer.instances[-1]
                self.assertTrue(producer.closed)
                self.assertEqual(len(producer.messages), 1)
                payload = json.loads(producer.messages[0][1].decode("utf-8"))["messageContent"]
                self.assertEqual(payload["progress"], 100)
                self.assertEqual(payload["runningStatus"], "completed")


class EntryProgressTests(unittest.TestCase):
    def test_start_progress_precedes_algorithm_execution(self):
        for source_path in [
            ROOT / "change" / "change_detection_core.py",
            ROOT / "fenlei" / "classification_core.py",
        ]:
            with self.subTest(source=str(source_path)):
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                entry = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "entry"
                )
                try_index = next(
                    index for index, node in enumerate(entry.body) if isinstance(node, ast.Try)
                )
                startup_nodes = entry.body[:try_index]
                startup_text = ast.dump(ast.Module(body=startup_nodes, type_ignores=[]))
                self.assertIn("任务已接收，正在初始化", startup_text)

    def test_entry_rethrows_after_sending_failed_status(self):
        for source_path in [
            ROOT / "change" / "change_detection_core.py",
            ROOT / "fenlei" / "classification_core.py",
        ]:
            with self.subTest(source=str(source_path)):
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                entry = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "entry"
                )
                handlers = [handler for node in entry.body if isinstance(node, ast.Try) for handler in node.handlers]
                handler_text = ast.dump(ast.Module(body=handlers[0].body, type_ignores=[]))
                self.assertIn("failed", handler_text)
                self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(handlers[0])))

    def test_tiff_and_missing_output_warnings_do_not_fail_the_step(self):
        for source_path in [
            ROOT / "change" / "change_detection_core.py",
            ROOT / "fenlei" / "classification_core.py",
        ]:
            with self.subTest(source=str(source_path)):
                source = source_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                warning_class = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "NonFatalTaskWarning"
                )
                warning_function = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_is_nonfatal_warning"
                )
                namespace = {}
                exec(
                    compile(
                        ast.Module(body=[warning_class, warning_function], type_ignores=[]),
                        str(source_path),
                        "exec",
                    ),
                    namespace,
                )
                is_warning = namespace["_is_nonfatal_warning"]
                self.assertTrue(is_warning(RuntimeError("Maximum TIFF file size exceeded")))
                self.assertFalse(is_warning(ValueError("参数格式错误")))
                self.assertIn("'runningStatus': 'completed'", source)

    def test_batch_partial_failures_are_reported_as_warnings(self):
        for source_path in [
            ROOT / "change" / "change_detection_core.py",
            ROOT / "fenlei" / "classification_core.py",
        ]:
            with self.subTest(source=str(source_path)):
                source = source_path.read_text(encoding="utf-8")
                self.assertIn("swap_write('warning_list', failed_list)", source)
                self.assertIn("出现告警并已跳过", source)
                self.assertNotIn("raise RuntimeError(f'{result_msg}", source)

    def test_batch_zero_success_is_a_real_failure(self):
        expected_messages = {
            ROOT / "change" / "change_detection_core.py": "批量变化检测全部失败",
            ROOT / "fenlei" / "classification_core.py": "批量类别识别全部失败",
        }
        for source_path, expected_message in expected_messages.items():
            with self.subTest(source=str(source_path)):
                source = source_path.read_text(encoding="utf-8")
                self.assertIn("if not output_shp_list:", source)
                self.assertIn(expected_message, source)

    def test_change_detection_does_not_report_95_before_zero_success_check(self):
        source = (ROOT / "change" / "change_detection_core.py").read_text(encoding="utf-8")
        zero_success_check = source.index("if not output_shp_list:")
        report_95 = source.index("'progress': 95", zero_success_check)
        self.assertLess(zero_success_check, report_95)
        self.assertIn("'progress': 97", source)
        self.assertIn("'progress': 99", source)
        self.assertIn("正在生成变化检测输出数据集", source)

    def test_spatial_validation_errors_are_not_silently_returned(self):
        source = (
            ROOT / "change" / "test_lib_batch_memeff_single_image_nomp.py"
        ).read_text(encoding="utf-8")
        self.assertIn('raise RuntimeError("两幅图像无空间交集，无法执行变化检测")', source)
        self.assertIn("raise RuntimeError(error_message) from e", source)

    def test_stale_shapefile_sidecars_are_removed(self):
        source_path = ROOT / "change" / "change_detection_core.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        selected_nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "_SHAPEFILE_SIDECARS"
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name == "_remove_shapefile_dataset")
        ]
        namespace = {"os": os}
        exec(
            compile(ast.Module(body=selected_nodes, type_ignores=[]), str(source_path), "exec"),
            namespace,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            shp_path = Path(temp_dir) / "result.shp"
            sidecars = [shp_path, shp_path.with_suffix(".dbf"), shp_path.with_suffix(".prj")]
            unrelated = Path(temp_dir) / "keep.txt"
            for path in [*sidecars, unrelated]:
                path.write_text("old", encoding="utf-8")
            namespace["_remove_shapefile_dataset"](shp_path)
            self.assertTrue(all(not path.exists() for path in sidecars))
            self.assertTrue(unrelated.exists())

    def test_local_change_results_copy_all_shapefile_sidecars_atomically(self):
        source_path = ROOT / "change" / "change_detection_core.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        helper_names = {
            "_remove_shapefile_dataset",
            "_remove_file_if_exists",
            "_copy_file_atomically",
            "_copy_shapefile_dataset",
        }
        selected_nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "_SHAPEFILE_SIDECARS"
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name in helper_names)
        ]
        namespace = {"os": os, "shutil": __import__("shutil"), "tempfile": tempfile}
        exec(
            compile(ast.Module(body=selected_nodes, type_ignores=[]), str(source_path), "exec"),
            namespace,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_shp = Path(temp_dir) / "local" / "result.shp"
            destination_shp = Path(temp_dir) / "shared" / "result.shp"
            source_shp.parent.mkdir()
            destination_shp.parent.mkdir()
            for extension in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                source_shp.with_suffix(extension).write_text(extension, encoding="utf-8")
            destination_shp.with_suffix(".shx").write_text("stale", encoding="utf-8")

            namespace["_copy_shapefile_dataset"](source_shp, destination_shp)

            for extension in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                self.assertEqual(
                    destination_shp.with_suffix(extension).read_text(encoding="utf-8"),
                    extension,
                )
            self.assertFalse(any(destination_shp.parent.glob("*.copying")))

    def test_change_progress_reports_tiff_size_and_eta_and_cleans_scratch(self):
        source = (ROOT / "change" / "change_detection_core.py").read_text(encoding="utf-8")
        self.assertIn("os.path.getsize(_tif_path)", source)
        self.assertIn("TIFF已写", source)
        self.assertIn("任务剩余约", source)
        self.assertIn("finally:\n            if pair_scratch_dir is not None:", source)
        self.assertIn("shutil.rmtree(pair_scratch_dir, ignore_errors=True)", source)

    def test_change_fft_tiles_are_small_enough_for_parallel_preprocessing(self):
        source = (
            ROOT / "change" / "test_lib_batch_memeff_single_image_nomp.py"
        ).read_text(encoding="utf-8")
        self.assertIn("'TEST_IMG_SIZE': 1280", source)
        self.assertIn("'TEST_BATCHES': 1", source)
        self.assertIn("'TEST_PREFETCH_FACTOR': 1", source)

    def test_classification_temp_dirs_are_cleaned_on_entry_exit(self):
        source_path = ROOT / "fenlei" / "classification_core.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        helper_names = {
            "_create_classification_temp_dir",
            "_cleanup_classification_temp_dir",
            "_cleanup_all_classification_temp_dirs",
        }
        selected_nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "_ACTIVE_CLASSIFICATION_TEMP_DIRS"
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name in helper_names)
        ]
        namespace = {"tempfile": tempfile, "shutil": __import__("shutil")}
        exec(
            compile(ast.Module(body=selected_nodes, type_ignores=[]), str(source_path), "exec"),
            namespace,
        )
        temp_dir = namespace["_create_classification_temp_dir"]()
        self.assertTrue(Path(temp_dir).is_dir())
        namespace["_cleanup_all_classification_temp_dirs"]()
        self.assertFalse(Path(temp_dir).exists())

    def test_empty_zonal_stats_map_to_unknown(self):
        source = (ROOT / "fenlei" / "classification_core.py").read_text(encoding="utf-8")
        self.assertIn("major_classes.append(0)", source)
        self.assertNotIn("major_classes.append(5)", source)


class ShapefileMergeTests(unittest.TestCase):
    def test_merge_handles_empty_inputs_crs_and_utf8(self):
        source = (ROOT / "change" / "merge.py").read_text(encoding="utf-8")
        self.assertIn("options=['ENCODING=UTF-8']", source)
        self.assertIn("CoordinateTransformation", source)
        self.assertIn("return str(output_path)", source)
        self.assertNotIn("if not valid_shp_files", source)


class PreviewTests(unittest.TestCase):
    def test_tile_thumbnails_are_disabled_but_progress_remains(self):
        source = (
            ROOT / "fenlei" / "clie_new" / "CLInferEngine" / "clie_lib" / "run_lib.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("cv2.imwrite(temp_preview_filename", source)
        self.assertNotIn("temp_preview_root", source)
        self.assertIn("_last_subprocess_progress_key", source)
        self.assertIn("message_dict['runningStatus'] = 'running'", source)


class BigTiffWriterTests(unittest.TestCase):
    class FakeBand:
        def __init__(self, gdal_module):
            self.gdal_module = gdal_module
            self.fail_write = False

        def WriteArray(self, *args, **kwargs):
            if self.fail_write:
                self.gdal_module.error_type = self.gdal_module.CE_Failure
                self.gdal_module.error_message = "TIFFAppendToStrip:Maximum TIFF file size exceeded"
                return 1
            return 0

        def FlushCache(self):
            return 0

    class FakeDataset:
        def __init__(self, gdal_module):
            self.gdal_module = gdal_module
            self.band = BigTiffWriterTests.FakeBand(gdal_module)
            self.fail_flush = False

        def GetRasterBand(self, index):
            return self.band

        def FlushCache(self):
            if self.fail_flush:
                self.gdal_module.error_type = self.gdal_module.CE_Failure
                self.gdal_module.error_message = "TIFFAppendToStrip:Maximum TIFF file size exceeded"
                return 1
            return 0

    class FakeDriver:
        def __init__(self, gdal_module):
            self.gdal_module = gdal_module
            self.options = None

        def Create(self, filename, width, height, nbands, dtype, options=None):
            self.options = options
            self.dataset = BigTiffWriterTests.FakeDataset(self.gdal_module)
            return self.dataset

    class FakeGdal(types.ModuleType):
        GDT_Byte = 1
        GDT_UInt16 = 2
        GDT_Int16 = 3
        GDT_UInt32 = 4
        GDT_Int32 = 5
        GDT_Float32 = 6
        GDT_Float64 = 7
        CE_Failure = 3

        def __init__(self):
            super().__init__("gdal")
            self.error_type = 0
            self.error_message = ""
            self.driver = BigTiffWriterTests.FakeDriver(self)

        def GetDriverByName(self, name):
            return self.driver

        def ErrorReset(self):
            self.error_type = 0
            self.error_message = ""

        def GetLastErrorType(self):
            return self.error_type

        def GetLastErrorMsg(self):
            return self.error_message

    class FakeImage:
        def __getitem__(self, item):
            return object()

    def test_writer_forces_bigtiff_and_surfaces_write_errors(self):
        fake_gdal = self.FakeGdal()
        imageio = load_imageio_module(fake_gdal)
        writer = imageio.ImageWriter("large.tif", 100, 100, 1, compress="LZW")

        self.assertIn("BIGTIFF=YES", fake_gdal.driver.options)
        self.assertIn("TILED=YES", fake_gdal.driver.options)
        fake_gdal.driver.dataset.band.fail_write = True
        with self.assertRaisesRegex(RuntimeError, "Maximum TIFF file size exceeded"):
            writer.write_image(self.FakeImage())

    def test_active_change_detection_writer_uses_bigtiff(self):
        source = (ROOT / "change" / "test_lib_batch_memeff_single_image_nomp.py").read_text(encoding="utf-8")
        self.assertIn("options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']", source)

    def test_writer_surfaces_deferred_final_flush_errors(self):
        fake_gdal = self.FakeGdal()
        imageio = load_imageio_module(fake_gdal)
        writer = imageio.ImageWriter("large.tif", 100, 100, 1, compress="LZW")
        fake_gdal.driver.dataset.fail_flush = True

        with self.assertRaisesRegex(RuntimeError, "Maximum TIFF file size exceeded"):
            writer.close()

    def test_classification_does_not_swallow_writer_close_errors(self):
        source = (
            ROOT / "fenlei" / "clie_new" / "CLInferEngine" / "clie_lib" / "run_lib.py"
        ).read_text(encoding="utf-8")
        self.assertIn("raise RuntimeError(f'分类结果写盘失败:", source)
        self.assertIn("raise RuntimeError(f'分类结果写盘或后处理失败:", source)


if __name__ == "__main__":
    unittest.main()
