import pandas as pd
import dtale
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(19, 12))
df = pd.read_csv("sub_classifiers/AP.csv")
sns.scatterplot(data=df, x="sub_classifier",y="AP",hue="main_classifier", ax=ax, s=200,edgecolor="k" )
ax.set_title("Average Precision Results")
ax.set_xlabel("Text files corresponding to Sub-Classifiers")
plt.tight_layout()
plt.savefig("./sub_classifiers/AP.png", dpi=400)
plt.show()
