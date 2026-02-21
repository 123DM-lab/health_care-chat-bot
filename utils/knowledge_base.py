import pandas as pd

desc_df = pd.read_csv("data/description.csv")
prec_df = pd.read_csv("data/precautions.csv")

print("Description CSV columns:", desc_df.columns)
print("Precautions CSV columns:", prec_df.columns)

def get_disease_info(disease):
    # Strip column names in case of spaces
    desc_df.columns = desc_df.columns.str.strip()
    prec_df.columns = prec_df.columns.str.strip()

    # For description, 'Drug Reaction' is the disease column
    description = desc_df[desc_df["Drug Reaction"] == disease].iloc[:, 1].values

    # For precautions, 'Drug Reaction' is also the key column, rest are precautions
    precautions = prec_df[prec_df["Drug Reaction"] == disease].iloc[:, 1:].values

    desc_text = description[0] if len(description) > 0 else "No description available"
    prec_list = precautions[0].tolist() if len(precautions) > 0 else []

    return desc_text, prec_list



