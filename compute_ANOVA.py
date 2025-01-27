import numpy as np
import pandas as pd
from scipy.stats import f_oneway
def compute_OneWayANOVA():
  male_df = pd.read_csv(f"male_wer.csv")
  female_df = pd.read_csv(f"female_wer.csv")
  male_list = male_df.values.tolist()
  female_list = female_df.values.tolist()
  anova_results = f_oneway(male_list, female_list)
  sig_value = ''
  if anova_results[1][0] < 0.05:
    sig_value = 'True'
  else:
    sig_value = 'False'
  print("\nOne-Way ANOVA:")
  print("F Statistic:",anova_results[0])
  print("P value:",anova_results[1], ":Significant:",sig_value)

if __name__ == "__main__":
  compute_OneWayANOVA()