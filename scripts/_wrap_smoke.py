import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ['ENABLE_QWEN_LLM'] = '1'
print('before-main')
import scripts.run_qwen_smoke_test as smoke
print(smoke.OUTPUT_DIR)
smoke.main()
print('after-main')
