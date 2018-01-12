# -*- coding: utf-8 -*-
import os
import sys
current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))
from configs.globals import PROJECT_FOLDER

DATA_FOLDER = PROJECT_FOLDER + "/data/knowledge/"