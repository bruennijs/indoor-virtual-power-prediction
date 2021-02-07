import os
from io import open
from xmltodict import parse as xml_parse

def read_tcx(file_path: str) -> dict:
    project_root_dir = os.path.abspath('.')
    abs_file_path = os.path.join(project_root_dir, file_path)
    with open(abs_file_path, mode='r', encoding='utf-8') as f:
        content = f.read()
        return xml_parse(content)

