import ray
import modin.pandas as pd
import sys

ray.init()

input_path = sys.argv[0]
df = pd.read_parquet(input_path)
res = df.count()
print(res)





