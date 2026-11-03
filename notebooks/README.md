# Notebooks

**Naming:** `NN_<initials>_<topic>.ipynb`, e.g. `01_AB_eda.ipynb`, `03_CD_bilstm_tuning.ipynb`.

**Import the shared code** by putting this in the first cell:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))   # repo root

from src.data.loaders import load_splits
from src.evaluation.metrics import compute_metrics, log_run
```

**Tips**
- Notebooks are good for exploration and for the video code walkthrough. Move reusable code into `src/`.
- Before committing, clear huge outputs (e.g. full training logs) to keep diffs readable.
- Save figures for slides to `results/figures/`.
