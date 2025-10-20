import re
import pandas as pd

REPORT_FILES = [
    '../3_Results/mobilenet/reports/classification_report_mobilenet.txt',
    '../3_Results/convnext/reports/classification_report_convnext.txt',
    '../3_Results/effv2l/reports/classification_report_effv2l.txt',
    '../3_Results/resnet101v2/reports/classification_report_resnet101v2.txt',
    '../3_Results/effv2s/reports/classification_report_effv2s.txt',
    '../3_Results/convnextl/reports/classification_report_convnextl.txt',
    '../3_Results/nasnet/reports/classification_report_nasnet.txt',
    '../3_Results/fossil_classifier_final/reports/classification_report_fossil_classifier_final.txt',
]


def parse_report(filepath):
    """Parse a sklearn classification report text file and extract accuracy, Top-3 and macro avg metrics.

    Returns a dict: {'accuracy','top3','auc','macro_precision','macro_recall','macro_f1'}
    """
    res = {'accuracy': None, 'top3': None, 'auc': None, 'macro_precision': None, 'macro_recall': None, 'macro_f1': None}
    with open(filepath, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]

    # Scan all lines for the metrics we care about
    for ln in lines:
        # Accuracy: either 'Test Accuracy : 0.9182' or the classification table 'accuracy 0.9182'
        if re.search(r'\bTest\s+Accuracy\b|^\s*accuracy\s', ln, re.I):
            parts = re.split(r'[:\s]+', ln.strip())
            for tok in parts[1:]:
                try:
                    val = float(tok)
                    res['accuracy'] = val
                    break
                except ValueError:
                    continue
        # Top-3 accuracy lines like 'Top-3 Accuracy: 0.9954'
        if re.search(r'Top[- ]?3', ln, re.I):
            parts = re.split(r'[:\s]+', ln.strip())
            for tok in parts[1:]:
                try:
                    val = float(tok)
                    res['top3'] = val
                    break
                except ValueError:
                    continue
        # AUC lines like 'Test AUC      : 0.9956'
        if re.search(r'\bAUC\b', ln, re.I):
            parts = re.split(r'[:\s]+', ln.strip())
            for tok in parts[1:]:
                try:
                    val = float(tok)
                    res['auc'] = val
                    break
                except ValueError:
                    continue
        # macro avg within the classification report table
        m = re.match(r'\s*macro avg\s+(\S+)\s+(\S+)\s+(\S+)', ln)
        if m:
            res['macro_precision'] = float(m.group(1))
            res['macro_recall'] = float(m.group(2))
            res['macro_f1'] = float(m.group(3))
            continue

    return res


def main():
    rows = []
    for fp in REPORT_FILES:
        try:
            vals = parse_report(fp)
        except FileNotFoundError:
            print(f"Warning: file not found: {fp}")
            vals = {'accuracy': None, 'macro_precision': None, 'macro_recall': None, 'macro_f1': None ,'top3': None }
        model = fp.split('/')[-3]  # get folder name like 'mobilenet'
        rows.append({'model': model, **vals})

    df = pd.DataFrame(rows).set_index('model')
    df.sort_values(by='accuracy', ascending=False, inplace=True)

    # Round to 3 decimals for presentation
    out_df = df[['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'top3', 'auc']].round(3)

    # Print LaTeX table (3 decimals)
    print(out_df.to_latex(float_format="{:.3f}".format, na_rep='--'))

    # Also save CSV for reference
    # out_df.to_csv('./3_Results/_comparison/macro_accuracy_table.csv')


if __name__ == '__main__':
    main()
