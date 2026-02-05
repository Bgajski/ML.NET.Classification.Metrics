#include "coremetrics.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

static inline float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static inline void add_trap_area(double& auc, const RocPoint& prev, const RocPoint& cur) {
    // ensure non negative dx
    double dx = cur.Fpr - prev.Fpr;
    if (dx < 0.0) dx = 0.0;
    auc += dx * (cur.Tpr + prev.Tpr) * 0.5;
}

static inline void compute_totals(const uint8_t* labels, size_t n, uint64_t& total_pos, uint64_t& total_neg) {
    total_pos = 0;
    for (size_t i = 0; i < n; i++) total_pos += (labels[i] != 0) ? 1ull : 0ull;
    total_neg = static_cast<uint64_t>(n) - total_pos;
}

extern "C" int32_t ComputeRocAuc_Binned(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    int32_t buckets,
    RocPoint* out_roc,
    size_t out_roc_len,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) {
    if (!scores || !labels || !out_roc || !out_auc) return CM_ERR_NULLPTR;
    if (n == 0) return CM_ERR_INVALID_N;
    if (buckets <= 0) return CM_ERR_INVALID_BUCKETS;

    const size_t bins = static_cast<size_t>(buckets) + 1;
    if (out_roc_len < bins) return CM_ERR_INVALID_OUTLEN;

    // build sorted indices by descending score
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;

    auto score_at = [&](size_t i) -> float {
        float s = scores[i];
        return clamp_scores_to_unit_interval ? clamp01(s) : s;
        };

    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        float sa = score_at(a);
        float sb = score_at(b);
        return sa > sb;
        });

    uint64_t total_pos = 0, total_neg = 0;
    compute_totals(labels, n, total_pos, total_neg);

    uint64_t tp = 0, fp = 0;
    size_t cursor = 0;

    double auc = 0.0;
    RocPoint prev{ 0.0, 0.0, std::numeric_limits<float>::infinity() };
    bool wrote_any = false;

    // threshold from 1.0 -> 0.0
    for (size_t b = 0; b < bins; b++) {
        double t = 1.0 - (static_cast<double>(b) / static_cast<double>(buckets));
        float cut = static_cast<float>(t);

        // include all scores >= cut
        while (cursor < n && score_at(idx[cursor]) >= cut) {
            if (labels[idx[cursor]] != 0) tp++; else fp++;
            cursor++;
        }

        double tpr = (total_pos > 0) ? (static_cast<double>(tp) / static_cast<double>(total_pos)) : 0.0;
        double fpr = (total_neg > 0) ? (static_cast<double>(fp) / static_cast<double>(total_neg)) : 0.0;

        RocPoint cur{ fpr, tpr, cut };
        out_roc[b] = cur;

        if (wrote_any) add_trap_area(auc, prev, cur);
        prev = cur;
        wrote_any = true;
    }

    *out_auc = auc;
    return CM_OK;
}

extern "C" int32_t ComputeRocAuc_Exact(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    RocPoint* out_roc,
    size_t out_roc_len,
    size_t* out_points_written,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) {
    if (!scores || !labels || !out_roc || !out_points_written || !out_auc) return CM_ERR_NULLPTR;
    if (n == 0) return CM_ERR_INVALID_N;

    auto score_at = [&](size_t i) -> float {
        float s = scores[i];
        return clamp_scores_to_unit_interval ? clamp01(s) : s;
        };

    // sort indices by descending score
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        float sa = score_at(a);
        float sb = score_at(b);
        if (sa == sb) return a < b;
        return sa > sb;
        });

    uint64_t total_pos = 0, total_neg = 0;
    compute_totals(labels, n, total_pos, total_neg);

    // 1) start point (0,0, +inf)
    // 2) one point after each tie group is added (threshold = that score)
    // 3) end point (1,1, -inf)
    
    // maximum points n + 2
    if (out_roc_len < (n + 2)) {
        return CM_ERR_INVALID_OUTLEN;
    }

    size_t w = 0;
    // start point threshold above max => predict none positive
    out_roc[w++] = RocPoint{ 0.0, 0.0, std::numeric_limits<float>::infinity() };

    uint64_t tp = 0, fp = 0;

    size_t cursor = 0;
    while (cursor < n) {
        float current_score = score_at(idx[cursor]);

        // handle tied scores as a single threshold step
        uint64_t tp_add = 0, fp_add = 0;
        size_t j = cursor;
        while (j < n && score_at(idx[j]) == current_score) {
            if (labels[idx[j]] != 0) tp_add++; else fp_add++;
            j++;
        }

        tp += tp_add;
        fp += fp_add;
        cursor = j;

        double tpr = (total_pos > 0) ? (static_cast<double>(tp) / static_cast<double>(total_pos)) : 0.0;
        double fpr = (total_neg > 0) ? (static_cast<double>(fp) / static_cast<double>(total_neg)) : 0.0;

        // threshold = current_score (meaning predict positive if score >= current_score)
        out_roc[w++] = RocPoint{ fpr, tpr, current_score };
    }

    // end point threshold below min => predict all positive (guaranteed (1,1) for finite totals)
    out_roc[w++] = RocPoint{ 1.0, 1.0, -std::numeric_limits<float>::infinity() };

    // compute AUC from written points 
    double auc = 0.0;
    for (size_t i = 1; i < w; i++) {
        add_trap_area(auc, out_roc[i - 1], out_roc[i]);
    }

    *out_points_written = w;
    *out_auc = auc;
    return CM_OK;
}
