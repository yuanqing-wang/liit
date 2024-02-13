import re
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

def parse_patent(path):
    with open(path, "r") as f:
        xml = f.read()
    xml = re.sub(r" xmlns=\".*?\"", "", xml)
    xml = re.sub(r"dl:", "", xml)
    root = ET.fromstring(xml)
    for reaction in root:
        for child in reaction:
            print(child)
    return root


    