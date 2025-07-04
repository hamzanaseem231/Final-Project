import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create sample dataset
df = sns.load_dataset("iris")

# Create a seaborn plot
sns.pairplot(df, hue="species")
plt.show()
