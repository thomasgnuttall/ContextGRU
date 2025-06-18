import pandas as pd
import numpy as np
all_experiments = [f"increasing_context/{round(v,2)}_svara" for v in np.arange(0.1,4.2,0.1)]
#	"experiment_1_svara",
#	"experiment_1_svara_form",
#	"experiment_2_svara",
#	"experiment_2_svara_form",
#	"experiment_3_svara",
#	"experiment_3_svara_form",
#	"experiment_4_svara",
#	"experiment_4_svara_form",
#	"experiment_5_svara",
#	"experiment_5_svara_form",
#	"experiment_6_svara",
#	"experiment_6_svara_form",
#	"experiment_7_svara",
#	"experiment_7_svara_form",
#	"experiment_8_svara",
#	"experiment_8_svara_form"
#]

def clean_dataframe(df):
    """
    Cleans the DataFrame by:
    1. Removing rows where all values exactly match the column names.
    2. Converting all columns to numeric types (non-numeric values become NaN).
    """
    header_values = list(df.columns)
    
    # Remove rows that exactly match the column names
    mask = ~df.apply(lambda row: list(row) == header_values, axis=1)
    df = df[mask].reset_index(drop=True)
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df



for experiment_name in all_experiments:
	path = f"models/{experiment_name}/results.csv"

	try:
		df = pd.read_csv(path)
		df = clean_dataframe(df)
		f1 = df.groupby('fold_idx')['avg_test_f1'].apply(np.max).values
		f1_mean = np.mean(f1)
		f1_std = np.std(f1)

		print(f'{experiment_name}: f1 mean: {round(f1_mean,3)}, f1 std: {round(f1_std,3)}')
	except:
		continue


