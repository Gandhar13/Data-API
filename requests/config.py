from pathlib import Path
from models import Frequency


DATA_FILE    = Path(__file__).parent.parent / "data" / "testdata.parquet"
RAM_LIMIT    = 500 * 1024 * 1024  # 500MB per request

# Mapping from our Frequency enum to NSEResampler's target_tf labels
FREQUENCY_TO_TF = {
    Frequency.MIN_30: "30min",
    Frequency.HOUR_1: "1H",
    Frequency.HOUR_2: "2H",
    Frequency.HOUR_4: "4H",
    Frequency.DAY_1:  "1D",
    Frequency.WEEK_1: "1W",
    Frequency.MONTH_1: "1M"
}