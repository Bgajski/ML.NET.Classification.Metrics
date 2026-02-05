using ML.NET.Metrics.Evaluation.Internal;

namespace ML.NET.Metrics.Evaluation.Public
{
    public static class CoreMetricsEvaluator
    {
        static CoreMetricsEvaluator() => NativeResolver.EnsureRegistered();

        public static MetricsCurve<RocPoint> EvaluateRoc(
            float[] scores,
            bool[] labels,
            RocMode mode = RocMode.Exact,
            int buckets = 100,
            bool clampScoresToUnitInterval = true
        )
        {
            ArgumentNullException.ThrowIfNull(scores);
            ArgumentNullException.ThrowIfNull(labels);
            if (scores.Length != labels.Length)
                throw new ArgumentException("scores and labels lengths differ");
            if (scores.Length == 0)
                throw new ArgumentException("scores and labels are empty");

            // convert labels bool -> byte
            var labels8 = new byte[labels.Length];
            for (int i = 0; i < labels.Length; i++)
                labels8[i] = labels[i] ? (byte)1 : (byte)0;

            byte clamp01 = clampScoresToUnitInterval ? (byte)1 : (byte)0;

            if (mode == RocMode.Binned)
            {
                if (buckets <= 0) throw new ArgumentOutOfRangeException(nameof(buckets));
                int bins = buckets + 1;
                var roc = new RocPoint[bins];

                int status = CoreMetricsMethods.ComputeRocAuc_Binned(
                    scores,
                    labels8,
                    (nuint)scores.Length,
                    buckets,
                    roc,
                    (nuint)roc.Length,
                    out double auc,
                    clamp01
                );

                if (status != (int)CoreMetricsStatus.Ok)
                    throw new InvalidOperationException($"Native ROC (binned) failed: {(CoreMetricsStatus)status}");

                return new MetricsCurve<RocPoint>(roc, auc);
            }
            else
            {
                // worst case points = n + 2
                var rocBuf = new RocPoint[scores.Length + 2];

                int status = CoreMetricsMethods.ComputeRocAuc_Exact(
                    scores,
                    labels8,
                    (nuint)scores.Length,
                    rocBuf,
                    (nuint)rocBuf.Length,
                    out nuint written,
                    out double auc,
                    clamp01
                );

                if (status != (int)CoreMetricsStatus.Ok)
                    throw new InvalidOperationException($"Native ROC (exact) failed: {(CoreMetricsStatus)status}");

                int w = checked((int)written);
                var roc = new RocPoint[w];
                Array.Copy(rocBuf, roc, w);

                return new MetricsCurve<RocPoint>(roc, auc);
            }
        }
    }
}
