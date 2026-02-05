using System.Runtime.InteropServices;

namespace ML.NET.Metrics.Evaluation.Public
{
    [StructLayout(LayoutKind.Sequential)]
    public struct RocPoint
    {
        public double Fpr;
        public double Tpr;
        public float Threshold;
    }

    public enum RocMode
    {
        Exact,
        Binned
    }

    public enum CoreMetricsStatus : int
    {
        Ok = 0,
        ErrNullPtr = 1,
        ErrInvalidN = 2,
        ErrInvalidBuckets = 3,
        ErrInvalidOutLen = 4
    }
}
