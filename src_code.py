import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.calibration import calibration_curve
from sklearn.metrics import (log_loss, brier_score_loss, roc_auc_score,
                             average_precision_score, precision_recall_curve)
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
import os

# Setup directories relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
results_file = os.path.join(script_dir, "detailed_metrics_report.txt")

np.random.seed(42)


# Synthetic Dataset Generation
def generate_synthetic_email_dataset():
    class_names = ['Spam', 'Promotions', 'Social', 'Updates', 'Forums']
    class_counts = [89, 118, 86, 152, 55]
    total_samples = sum(class_counts)
    labels = np.concatenate([np.full(n, i) for i, n in enumerate(class_counts)])

    logits = np.random.normal(0, 0.3, (total_samples, len(class_names)))
    for i in range(total_samples):
        logits[i, labels[i]] += 3.0  # Bias for true class logits
    logits *= 2.2  # Amplify logits for overconfidence

    predicted_labels = labels.copy()
    mismatch_prob = 0.33
    for i in range(total_samples):
        if np.random.rand() < mismatch_prob:
            true_c = labels[i]
            if true_c == 0:
                predicted_labels[i] = 1 if np.random.rand() < 0.7 else 0
            elif true_c == 1:
                predicted_labels[i] = 0 if np.random.rand() < 0.4 else 2
            elif true_c == 2:
                predicted_labels[i] = 3
            elif true_c == 3:
                predicted_labels[i] = 2 if np.random.rand() < 0.5 else 4
            else:
                predicted_labels[i] = 3

    for i in range(total_samples):
        logits[i] = np.random.normal(0, 0.3, len(class_names))
        logits[i, predicted_labels[i]] += 2.2  # Predicted class boost

    probs = softmax(logits, axis=1)
    return logits, probs, labels, class_names


# Quantitative metrics implementation
def expected_calibration_error(probs, labels, bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(conf[in_bin])
            ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence_in_bin)
    return ece


def maximum_calibration_error(probs, labels, bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    bin_edges = np.linspace(0, 1, bins + 1)
    max_err = 0.0
    for i in range(bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if np.any(in_bin):
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(conf[in_bin])
            err = abs(accuracy_in_bin - avg_confidence_in_bin)
            if err > max_err:
                max_err = err
    return max_err


def negative_log_likelihood(probs, labels):
    return log_loss(labels, probs)


def brier_score(probs, labels):
    lb = LabelBinarizer()
    true_bin = lb.fit_transform(labels)
    return np.mean([brier_score_loss(true_bin[:, i], probs[:, i]) for i in range(probs.shape[1])])


def ranked_probability_score(probs, labels):
    lb = LabelBinarizer()
    Y = lb.fit_transform(labels)
    F = np.cumsum(probs, axis=1)
    O = np.cumsum(Y, axis=1)
    return np.mean(np.sum((F - O) ** 2, axis=1))


def calibration_slope_intercept(probs, labels):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(float)
    X = sm.add_constant(conf)
    model = sm.OLS(correct, X).fit()
    return model.params[1], model.params[0]


def auroc_confidence_correctness(probs, labels):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    try:
        return roc_auc_score(correct, conf)
    except:
        return float('nan')


def auprc_confidence_correctness(probs, labels):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    try:
        return average_precision_score(correct, conf)
    except:
        return float('nan')


def correlation_metrics(probs, labels):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    pearson_corr = pearsonr(conf, correct)[0]
    spearman_corr = spearmanr(conf, correct)[0]
    kendall_corr = kendalltau(conf, correct)[0]
    return pearson_corr, spearman_corr, kendall_corr


# Temperature Scaling calibration class
class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def _nll(self, T, logits, labels):
        scaled_logits = logits / T
        probs = softmax(scaled_logits, axis=1)
        return log_loss(labels, probs)

    def fit(self, logits, labels):
        temps = np.linspace(0.5, 5.0, 50)
        losses = []
        for t in temps:
            losses.append(self._nll(t, logits, labels))
        self.temperature = temps[np.argmin(losses)]

    def transform(self, logits):
        return logits / self.temperature


# Plot functions
def plot_reliability_diagram(probs, labels, title, filepath):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=15)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Predicted Confidence')
    plt.ylabel('True Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()


def plot_confidence_histogram(probs, labels, title, filepath):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels)
    plt.figure(figsize=(8, 5))
    sns.histplot(conf[correct], bins=30, color='green', label='Correct', stat='density', kde=True)
    sns.histplot(conf[~correct], bins=30, color='red', label='Incorrect', stat='density', kde=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()


def plot_risk_coverage_curve(probs, labels, filepath):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels)
    idx = np.argsort(-conf)
    coverage = np.arange(1, len(conf) + 1) / len(conf)
    risk = [1 - np.mean(correct[idx[:i]]) for i in range(1, len(conf) + 1)]
    plt.figure(figsize=(8, 5))
    plt.plot(coverage, risk, label="Risk (Error Rate)")
    plt.xlabel('Coverage')
    plt.ylabel('Error Rate')
    plt.title('Risk-Coverage Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(filepath)
    plt.close()


def plot_temperature_sweep(logits, labels, filepath):
    temps = np.linspace(0.5, 5, 50)
    nlls = []
    eces = []
    for T in temps:
        p = softmax(logits / T, axis=1)
        nlls.append(log_loss(labels, p))
        eces.append(expected_calibration_error(p, labels))
    plt.figure(figsize=(9, 5))
    plt.plot(temps, nlls, label="NLL")
    plt.plot(temps, eces, label="ECE")
    plt.xlabel("Temperature")
    plt.title("Temperature Sweep")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()


def main():
    logits, probs, labels, class_names = generate_synthetic_email_dataset()

    # Evaluate pre-calibration
    results = {}
    results['Negative Log-Likelihood'] = negative_log_likelihood(probs, labels)
    results['Brier Score'] = brier_score(probs, labels)
    results['Expected Calibration Error'] = expected_calibration_error(probs, labels)
    results['Maximum Calibration Error'] = maximum_calibration_error(probs, labels)
    results['Ranked Probability Score'] = ranked_probability_score(probs, labels)
    slope, intercept = calibration_slope_intercept(probs, labels)
    results['Calibration Slope'] = slope
    results['Calibration Intercept'] = intercept
    results['AUROC'] = auroc_confidence_correctness(probs, labels)
    results['AUPRC'] = auprc_confidence_correctness(probs, labels)
    pearson, spearman, kendall = correlation_metrics(probs, labels)
    results['Pearson'] = pearson
    results['Spearman'] = spearman
    results['Kendall Tau'] = kendall

    # Save pre-calibration plots
    plot_reliability_diagram(probs, labels, "Reliability Diagram - Before Calibration", os.path.join(plots_dir, "reliability_before.png"))
    plot_confidence_histogram(probs, labels, "Confidence Histogram - Before Calibration", os.path.join(plots_dir, "conf_hist_before.png"))
    plot_risk_coverage_curve(probs, labels, os.path.join(plots_dir, "risk_coverage_before.png"))
    plot_temperature_sweep(logits, labels, os.path.join(plots_dir, "temp_sweep.png"))

    # Calibrate logits with temperature scaling
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    calibrated_logits = scaler.transform(logits)
    calibrated_probs = softmax(calibrated_logits, axis=1)

    # Evaluate post-calibration
    results['Negative Log-Likelihood (Calibrated)'] = negative_log_likelihood(calibrated_probs, labels)
    results['Brier Score (Calibrated)'] = brier_score(calibrated_probs, labels)
    results['Expected Calibration Error (Calibrated)'] = expected_calibration_error(calibrated_probs, labels)
    results['Maximum Calibration Error (Calibrated)'] = maximum_calibration_error(calibrated_probs, labels)
    results['Ranked Probability Score (Calibrated)'] = ranked_probability_score(calibrated_probs, labels)
    slope_c, intercept_c = calibration_slope_intercept(calibrated_probs, labels)
    results['Calibration Slope (Calibrated)'] = slope_c
    results['Calibration Intercept (Calibrated)'] = intercept_c
    results['AUROC (Calibrated)'] = auroc_confidence_correctness(calibrated_probs, labels)
    results['AUPRC (Calibrated)'] = auprc_confidence_correctness(calibrated_probs, labels)
    pearson_c, spearman_c, kendall_c = correlation_metrics(calibrated_probs, labels)
    results['Pearson (Calibrated)'] = pearson_c
    results['Spearman (Calibrated)'] = spearman_c
    results['Kendall Tau (Calibrated)'] = kendall_c

    # Save post-calibration plots
    plot_reliability_diagram(calibrated_probs, labels, "Reliability Diagram - After Calibration", os.path.join(plots_dir, "reliability_after.png"))
    plot_confidence_histogram(calibrated_probs, labels, "Confidence Histogram - After Calibration", os.path.join(plots_dir, "conf_hist_after.png"))
    plot_risk_coverage_curve(calibrated_probs, labels, os.path.join(plots_dir, "risk_coverage_after.png"))

    # Write all results to text file
    with open(results_file, "w") as f:
        for metric, val in results.items():
            f.write(f"{metric}: {val:.6f}\n")

    # Display results
    for metric, val in results.items():
        print(f"{metric}: {val:.6f}")

    print(f"\nPlots saved under: {plots_dir}")
    print(f"Detailed metric results saved to: {results_file}")


if __name__ == "__main__":
    main()
