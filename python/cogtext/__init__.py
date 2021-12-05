from .datasets.pubmed import *
from .co_occurrence import co_occurrence_matrix

from .svd_embedding import svd_embedding

from datetime import datetime

__version__ = '0.1.' + datetime.now().strftime('%Y%m%d%H')
