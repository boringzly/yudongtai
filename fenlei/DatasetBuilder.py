import json
import os


class DatasetBuilder:

    def __init__(self, output_dataset_path=None, version=3):
        self._path = output_dataset_path
        self._version = version
        self._datasets = {}
        self._render = []

    def add(self, name, path, type, file_list=None):
        if file_list is None:
            file_list = sorted(os.listdir(path)) if os.path.isdir(path) else []
        self._datasets[name] = {
            "path": path,
            "file": file_list,
            "type": type
        }

    def add_value(self, name, value):
        self._datasets[name] = {
            "value": value,
            "type": "swap"
        }

    def set_render(self, render_list):
        self._render = render_list

    def save(self):
        if self._path is None:
            return
        data = {
            "dataset": self._datasets,
            "render": self._render,
            "version": self._version
        }
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
