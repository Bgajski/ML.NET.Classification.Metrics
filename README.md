# ML.NET Classification Metrics: ROC Curve & AUC

## About

ML.NET.Classification.Metrics is an unofficial library for ML.NET that adds detailed ROC curve generation and AUC calculation,
complementing ML.NET’s built-in metrics with richer visualization and clearer threshold analysis.
 
## Key Features

- ROC curve (Exact): Generates ROC points for every unique score threshold, with tied scores grouped, plus endpoints, giving the most accurate ROC curve and AUC.
- ROC curve (Binned): Sweeps thresholds from 1.0 to 0.0 using (buckets + 1) steps for a fast, fixed-size approximation (best when inputs are probabilities in `[0,1]`)
- Custom buckets: Choose bucket count in Binned mode to trade curve detail for speed.
- Consistent AUC: Computes AUC via trapezoidal integration over the same ROC points, so the curve and AUC always match.
- Threshold visibility: Each ROC point includes its threshold, making cutoff analysis (TPR/FPR changes) straightforward.
- Interop-ready: Returns strongly-typed .NET structs, making results easy to visualize in charts and diagnostics tools.

## Library Guide

`EvaluateRoc` builds a ROC curve by sorting samples by score and simulating different decision thresholds, expecting:

- `scores (float[])`: a ranking signal where higher values mean “more likely positive”
- `labels (bool[])`: ground truth labels (true = positive, false = negative)

Important: `scores[]` and `labels[]` must be aligned (same row order, same length).

- Use Probability when available (calibrated in [0,1]). Otherwise use Score and set clampScoresToUnitInterval: false.

### Logistic Regression: prepare scores and labels (ensure the correct label column name)

    public sealed class BinaryPrediction
    {
        public float Probability { get; set; } // Logistic Regression: in [0,1]
    }

    public sealed class LabelRow
    {
        public bool Label { get; set; }
    }

    var predictions = model.Transform(testData);

    float[] scores = mlContext.Data
        .CreateEnumerable<BinaryPrediction>(predictions, reuseRowObject: false)
        .Select(r => r.Probability)
        .ToArray();

    bool[] labels = mlContext.Data
        .CreateEnumerable<LabelRow>(testData, reuseRowObject: false)
        .Select(r => r.Label)
        .ToArray();

Example A: Exact ROC (recommended)

- Use Exact mode for the most accurate ROC curve and AUC (one point per unique threshold + endpoints).

    var rocExact = CoreMetricsEvaluator.EvaluateRoc(
        scores: scores,
        labels: labels,
        mode: RocMode.Exact,
        clampScoresToUnitInterval: true
    );

double auc = rocExact.Score;
var points = rocExact.Points; // ROC points

Example B: Binned ROC (fast approximation)

- Use Binned mode for a fast, fixed-size ROC curve: thresholds sweep from 1.0 → 0.0 using (buckets + 1) steps.
- Binned mode assumes scores are in `[0,1]` (or set `clampScoresToUnitInterval: true`).

    var rocBinned = CoreMetricsEvaluator.EvaluateRoc(
        scores: scores,
        labels: labels,
        mode: RocMode.Binned,
        buckets: 200,
        clampScoresToUnitInterval: true
    );

EvaluateRoc returns a MetricsCurve<RocPoint>:
- `roc.Score` → the AUC
- `roc.Points` → ROC points containing:
  
    - Fpr (False Positive Rate)
    - Tpr (True Positive Rate)
    - Threshold (the cutoff used for that point)

## NuGet package

1. Add the NuGet package: 
- dotnet add package ML.NET.Classification.Metrics --version 0.1.4

2. Import the namespace in your code
- using ML.NET.Metrics.Evaluation.Public;

## Libraries

- .NET 8.0
- Windows x64 native binary
