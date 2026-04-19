import os
import pandas as pd

results_dir = "/home/daizhirui/results/OREN/replica"
filename = "sdf_and_grad_metrics.csv"

df = pd.read_csv(os.path.join(results_dir, "room0/latest/eval", filename))
# change the row label to "room0"
df["scene"] = "room0"
for scene in ["room1", "room2", "office0", "office1", "office2", "office3", "office4"]:
    df_scene = pd.read_csv(os.path.join(results_dir, scene, "latest/eval", filename))
    df_scene["scene"] = scene
    df = pd.concat([df, df_scene], ignore_index=True)

print(df)
df.to_csv(os.path.join(results_dir, "all_scenes_metrics.csv"), index=False)