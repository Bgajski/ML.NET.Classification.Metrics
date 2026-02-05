namespace ML.NET.Metrics.Evaluation.Public
{
    public sealed class MetricsCurve<TPoint>
    {
        public MetricsCurve(TPoint[] points, double score)
        {
            Points = points;
            Score = score;
        }

        public TPoint[] Points { get; }
        public double Score { get; }
    }
}
