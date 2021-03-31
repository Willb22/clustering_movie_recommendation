#!/usr/bin/python3

import pandas as pd
import numpy as np
import datetime
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from variables import input_dir, output_dir
