#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define CM_EXPORT __declspec(dllexport)
#else
#define CM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

    // error codes
    enum CoreMetricsStatus : int32_t {
        CM_OK = 0,
        CM_ERR_NULLPTR = 1,
        CM_ERR_INVALID_N = 2,
        CM_ERR_INVALID_BUCKETS = 3,
        CM_ERR_INVALID_OUTLEN = 4
    };

    // ROC point with threshold
    struct RocPoint {
        double Fpr;
        double Tpr;
        float  Threshold;
    };

    // Binned ROC thresholds are swept from 1.0 down to 0.0 with (buckets+1) points
    CM_EXPORT int32_t ComputeRocAuc_Binned(
        const float* scores,
        const uint8_t* labels,   
        size_t n,
        int32_t buckets,
        RocPoint* out_roc,
        size_t out_roc_len,
        double* out_auc,
        uint8_t clamp_scores_to_unit_interval // 1=clamp to [0,1], 0=don’t clamp
    );

    // Exact ROC one point per unique threshold (tie handled) + endpoints
    // out_roc_len >= (unique_scores_count + 2) because of ((0,0, +inf) and (1,1, -inf))
    CM_EXPORT int32_t ComputeRocAuc_Exact(
        const float* scores,
        const uint8_t* labels,
        size_t n,
        RocPoint* out_roc,
        size_t out_roc_len,
        size_t* out_points_written,
        double* out_auc,
        uint8_t clamp_scores_to_unit_interval
    );

} // extern "C"
