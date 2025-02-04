from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

def evaluate(gold_standard_path, prediction_path):
    """
    The files should be in .tsv format with 3 ordered columns:
    regulator, target, score
    """
    def load_tsv(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=None)
        df = df.loc[df[0] != df[1]]

        return df

    gold_standard, prediction = load_tsv(gold_standard_path), load_tsv(prediction_path)

    merged_df = pd.merge(
        gold_standard,
        prediction,
        on=[0, 1],
        suffixes=('_gold_standard', '_prediction')
    )

    AUC = roc_auc_score(
        merged_df['2_gold_standard'].astype(int),
        merged_df['2_prediction'].astype(int)
    )
    AUPR = average_precision_score(
        merged_df['2_gold_standard'].astype(int),
        merged_df['2_prediction'].astype(int)
    )

    return AUC, AUPR