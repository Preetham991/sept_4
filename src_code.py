import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import pearsonr, spearmanr, kendalltau
import os

# Ensure folder paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(script_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
results_file = os.path.join(script_dir, "results_full_report.txt")

np.random.seed(42)

def generate_synthetic_email_dataset():
    class_names = ['Spam', 'Promotions', 'Social', 'Updates', 'Forums']
    class_counts = [89, 118, 86, 152, 55]
    total_samples = sum(class_counts)
    labels = np.concatenate([np.full(n, i) for i, n in enumerate(class_counts)])

    logits = np.random.normal(0, 0.3, (total_samples, len(class_names)))
    for i in range(total_samples):
        logits[i, labels[i]] += 3.0  # bias toward true class
    logits *= 2.2  # Amplify for strong overconfidence

    predicted_labels = labels.copy()
    mismatch_prob = 0.33
    for i in range(total_samples):
        if np.random.rand() < mismatch_prob:
            c = labels[i]
            if c == 0:
                predicted_labels[i] = 1 if np.random.rand() < 0.7 else 0
            elif c == 1:
                predicted_labels[i] = 0 if np.random.rand() < 0.4 else 2
            elif c == 2:
                predicted_labels[i] = 3
            elif c == 3:
                predicted_labels[i] = 2 if np.random.rand() < 0.5 else 4
            else:
                predicted_labels[i] = 3

    for i in range(total_samples):
        logits[i] = np.random.normal(0, 0.3, len(class_names))
        logits[i, predicted_labels[i]] += 2.2  # predicted class boost

    probs = softmax(logits, axis=1)
    return logits, probs, labels, class_names

# Quantitative Metrics
def expected_calibration_error(probs, labels, n_bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        idxs = (conf > bins[i]) & (conf <= bins[i+1])
        prop = np.mean(idxs)
        if prop > 0:
            ece += prop * abs(np.mean(correct[idxs]) - np.mean(conf[idxs]))
    return ece

def maximum_calibration_error(probs, labels, n_bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    errors = []
    for i in range(n_bins):
        idxs = (conf > bins[i]) & (conf <= bins[i+1])
        if np.any(idxs):
            errors.append(abs(np.mean(correct[idxs]) - np.mean(conf[idxs])))
    return max(errors) if errors else 0

def negative_log_likelihood(probs, labels):
    return log_loss(labels, probs)

def brier_score(probs, labels):
    lb = LabelBinarizer()
    Y = lb.fit_transform(labels)
    return np.mean([brier_score_loss(Y[:, i], probs[:, i]) for i in range(probs.shape[1])])

def ranked_probability_score(probs, labels):
    lb = LabelBinarizer()
    Y = lb.fit_transform(labels)
    F = np.cumsum(probs, axis=1)
    O = np.cumsum(Y, axis=1)
    return np.mean(np.sum((F - O) ** 2, axis=1))

def calibration_slope_intercept(probs, labels):
    import statsmodels.api as sm
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(float)
    X = sm.add_constant(conf)
    model = sm.OLS(correct, X).fit()
    return model.params[1], model.params[0]

def auroc_confidence_vs_correctness(probs, labels):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    try:
        return roc_auc_score(correct, conf)
    except:
        return np.nan

def auprc_confidence_vs_correctness(probs, labels):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    try:
        return average_precision_score(correct, conf)
    except:
        return np.nan

def correlation_metrics(probs, labels):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    return pearsonr(conf, correct)[0], spearmanr(conf, correct)[0], kendalltau(conf, correct)[0]

# Temperature Scaler Calibration
class TemperatureScaler:
    def __init__(self): self.temperature = 1.0
    def _nll(self, T, logits, labels):
        p = softmax(logits / T, axis=1)
        return log_loss(labels, p)
    def fit(self, logits, labels):
        temps = np.linspace(0.5, 5.0, 50)
        losses = [self._nll(t, logits, labels) for t in temps]
        self.temperature = temps[np.argmin(losses)]
    def transform(self, logits):
        return logits / self.temperature

# Plotting functions (accept full file path as filename)
def plot_reliability_diagram(probs, labels, title, filename):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=15)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, 'o-', label='Model')
    plt.plot([0,1],[0,1], '--', label='Perfect')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_reliability_per_class(probs, labels, class_names, folder):
    n = len(class_names)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, n*2))
    axes = axes.flatten()
    for i, c in enumerate(class_names):
        idx = labels == i
        if np.sum(idx) == 0:
            axes[i].axis('off')
            continue
        conf = np.max(probs[idx], axis=1)
        correct = (np.argmax(probs[idx], axis=1) == labels[idx]).astype(int)
        prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10)
        axes[i].plot(prob_pred, prob_true, marker='o')
        axes[i].plot([0,1],[0,1], '--', color='gray')
        axes[i].set_title(c)
        axes[i].set_xlabel('Confidence')
        axes[i].set_ylabel('Accuracy')
        axes[i].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "reliability_per_class.png"))
    plt.close()

def plot_confidence_histogram(probs, labels, title, filename):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels)
    plt.figure(figsize=(8,5))
    sns.histplot(conf[correct], bins=30, color='green', label='Correct', stat='density', kde=True)
    sns.histplot(conf[~correct], bins=30, color='red', label='Incorrect', stat='density', kde=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_heatmap_confidence_vs_error(probs, labels, folder):
    import matplotlib.colors as mcolors
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    bins = pd.cut(conf, bins=20)
    df = pd.DataFrame({'conf': conf, 'correct': correct, 'bin': bins})
    #grouped = df.groupby('bin')['correct'].agg(['count','mean'])
    grouped = df.groupby('bin', observed=False)['correct'].agg(['count','mean'])

    grouped['error'] = 1 - grouped['mean']
    plt.figure(figsize=(12, 3))
    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=0, vmax=grouped['error'].max())
    plt.bar(range(len(grouped)), grouped['error'], color=cmap(norm(grouped['error'])))
    plt.xticks(range(len(grouped)), grouped.index.astype(str), rotation=90)
    plt.title('Error Rate per Confidence Bin')
    plt.xlabel('Confidence Bin')
    plt.ylabel('Error Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "confidence_error_heatmap.png"))
    plt.close()

def plot_violin_box_plots(probs, labels, folder):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    df = pd.DataFrame({'Confidence': conf, 'Correct': correct.astype(str)})

    plt.figure(figsize=(8,5))
    sns.violinplot(x='Correct', y='Confidence', data=df)
    plt.title('Violin plot of Confidence by Correctness')
    plt.savefig(os.path.join(folder, "violin_plot_confidence.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Correct', y='Confidence', data=df)
    plt.title('Box plot of Confidence by Correctness')
    plt.savefig(os.path.join(folder, "box_plot_confidence.png"))
    plt.close()

def plot_risk_coverage_curve(probs, labels, filename):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels)
    sorted_idx = np.argsort(-conf)
    coverage = np.arange(1, len(conf)+1) / len(conf)
    risk = [1 - np.mean(correct[sorted_idx[:i]]) for i in range(1, len(conf)+1)]
    plt.figure(figsize=(8,5))
    plt.plot(coverage, risk)
    plt.title("Risk-Coverage Curve")
    plt.xlabel("Coverage")
    plt.ylabel("Error Rate")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_temperature_sweep(logits, labels, filename):
    temps = np.linspace(0.5,5,40)
    nlls = []
    eces = []
    for T in temps:
        p = softmax(logits / T, axis=1)
        nlls.append(log_loss(labels, p))
        eces.append(expected_calibration_error(p, labels))
    plt.figure(figsize=(9,5))
    plt.plot(temps, nlls, label="NLL")
    plt.plot(temps, eces, label="ECE")
    plt.xlabel("Temperature")
    plt.title("Temperature Sweep")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_cumulative_gain(probs, labels, folder):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    df = pd.DataFrame({'conf': conf, 'correct': correct}).sort_values(by='conf', ascending=False)
    df['cum_pos'] = df['correct'].cumsum()
    df['fraction'] = np.arange(1, len(df)+1)/len(df)
    plt.figure(figsize=(8,5))
    plt.plot(df['fraction'], df['cum_pos']/df['correct'].sum(), label='Model')
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Random')
    plt.title('Cumulative Gain Chart')
    plt.xlabel('Fraction of Samples')
    plt.ylabel('Fraction of Positives Found')
    plt.legend()
    plt.savefig(os.path.join(folder, "cumulative_gain.png"))
    plt.close()

def plot_lift_chart(probs, labels, folder):
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(int)
    df = pd.DataFrame({'conf': conf, 'correct': correct}).sort_values(by='conf', ascending=False)
    df['cum_hits'] = df['correct'].cumsum()
    df['lift'] = (df['cum_hits'] / np.arange(1,len(df)+1)) / (df['correct'].sum()/len(df))
    df['fraction'] = np.arange(1,len(df)+1)/len(df)
    plt.figure(figsize=(8,5))
    plt.plot(df['fraction'], df['lift'])
    plt.axhline(1, linestyle='--', color='gray')
    plt.title('Lift Chart')
    plt.xlabel('Fraction of Samples')
    plt.ylabel('Lift')
    plt.savefig(os.path.join(folder, "lift_chart.png"))
    plt.close()

def main():
    logits, probs, labels, class_names = generate_synthetic_email_dataset()

    # Quantitative metrics
    results = {}
    results['Negative Log-Likelihood'] = negative_log_likelihood(probs, labels)
    results['Brier Score'] = brier_score(probs, labels)
    results['Expected Calibration Error'] = expected_calibration_error(probs, labels)
    results['Maximum Calibration Error'] = maximum_calibration_error(probs, labels)
    results['Ranked Probability Score'] = ranked_probability_score(probs, labels)
    slope, intercept = calibration_slope_intercept(probs, labels)
    results['Calibration Slope'] = slope
    results['Calibration Intercept'] = intercept
    results['AUROC'] = auroc_confidence_vs_correctness(probs, labels)
    results['AUPRC'] = auprc_confidence_vs_correctness(probs, labels)
    pearson, spearman, kendall = correlation_metrics(probs, labels)
    results['Pearson Correlation'] = pearson
    results['Spearman Correlation'] = spearman
    results['Kendall Tau Correlation'] = kendall

    # Save metrics to file
    with open(results_file, 'w') as f:
        for k,v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    # Generate and save plots
    plot_reliability_diagram(probs, labels, "Overall Reliability Diagram", os.path.join(plot_dir, "reliability_overall.png"))
    plot_reliability_per_class(probs, labels, class_names, plot_dir)
    plot_confidence_histogram(probs, labels, "Confidence Histogram", os.path.join(plot_dir, "confidence_histogram.png"))
    plot_heatmap_confidence_vs_error(probs, labels, plot_dir)
    plot_violin_box_plots(probs, labels, plot_dir)
    plot_risk_coverage_curve(probs, labels, os.path.join(plot_dir, "risk_coverage_curve.png"))
    plot_temperature_sweep(logits, labels, os.path.join(plot_dir, "temperature_sweep.png"))
    plot_cumulative_gain(probs, labels, plot_dir)
    plot_lift_chart(probs, labels, plot_dir)

    # Console output
    print(f"Metrics computed and saved to {results_file}")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
    print(f"Plots saved in folder: {plot_dir}")

if __name__ == "__main__":
    main()
