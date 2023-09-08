import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


df = pd.read_csv('./data/raw/train.csv', low_memory=False)

# Count the occurrences of each unique value in the 'yr' column
yr_counts = df['yr'].value_counts()

unique_names = df['yr'].unique()



# Extract the names (unique values) from the index of yr_counts
names = yr_counts.index

# Create the bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=names, y=yr_counts.values)
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Categories')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('reports/figures/FD_categories.png')



# Filter out all warnings
warnings.filterwarnings("ignore")

# Select the columns of interest
columns_to_plot = [
    'GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per',
    'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPA', 'twoP_per', 'TPM', 'TPA', 'TP_per', 'blk_per',
    'stl_per', 'ftr', 'porpag', 'adjoe', 'pfr', 'ast_tov', 'rimmade', 'rimmade_rimmiss',
    'midmade', 'midmade_midmiss', 'dunksmade', 'dunksmiss_dunksmade', 'drtg', 'adrtg', 'dporpag',
    'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb',
    'ast', 'stl', 'blk', 'pts'
]

# Plot the distributions
plt.figure(figsize=(15, 25))
for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(10, 5, i)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(col)
    plt.tight_layout()

plt.savefig('reports/figures/kurtosis.png')

warnings.resetwarnings()