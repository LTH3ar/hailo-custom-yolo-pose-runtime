/**
 * Generic YOLOv8-pose post-process for Hailo `hailofilter`.
 * See yolov8_pose_postprocess.hpp for the overview.
 *
 * Expected HEF outputs (the "9 raw head convs" export used by the Hailo Model
 * Zoo and by hailo-custom-yolo-pose/scripts/04-pytorch-to-hailo.py):
 *   per scale: boxes  H x W x 4*reg_max   (DFL distribution, ltrb)
 *              scores H x W x num_classes (logits, or sigmoid if the .alls
 *                                          used change_output_activation)
 *              kpts   H x W x 3*K         (x, y, visibility per keypoint)
 * Tensor order inside the HEF does not matter: tensors are grouped by
 * spatial size and assigned a role by channel count.
 **/
#include "yolov8_pose_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/istreamwrapper.h"

namespace
{
const char *LOG_TAG = "[hcpose] ";

enum class Activation
{
    AUTO,
    SIGMOID, // apply sigmoid in the post-process (HEF outputs logits)
    NONE     // HEF already outputs probabilities
};

Activation parse_activation(const std::string &s)
{
    if (s == "sigmoid" || s == "logits")
        return Activation::SIGMOID;
    if (s == "none" || s == "identity" || s == "probabilities")
        return Activation::NONE;
    return Activation::AUTO;
}

const char *activation_name(Activation a)
{
    switch (a)
    {
    case Activation::SIGMOID:
        return "logits -> sigmoid applied here";
    case Activation::NONE:
        return "already sigmoid in HEF";
    default:
        return "auto";
    }
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---------------------------------------------------------------------------
// Tensor access that works for uint8, uint16 and float32 outputs.
// ---------------------------------------------------------------------------
struct TensorView
{
    const uint8_t *data = nullptr;
    int type = 1; // 1=uint8, 2=uint16, 3=float32 (HailoTensorFormatType values)
    float zp = 0.f;
    float scale = 1.f;
    int h = 0, w = 0, f = 0;

    explicit TensorView(const HailoTensorPtr &t)
    {
        data = t->data();
        type = static_cast<int>(t->format().type);
        if (type == 0)
            type = 1; // AUTO means device default -> uint8
        zp = t->quant_info().qp_zp;
        scale = t->quant_info().qp_scale;
        h = static_cast<int>(t->height());
        w = static_cast<int>(t->width());
        f = static_cast<int>(t->features());
    }

    inline float at(size_t idx) const
    {
        switch (type)
        {
        case 2:
            return (static_cast<float>(reinterpret_cast<const uint16_t *>(data)[idx]) - zp) * scale;
        case 3:
            return reinterpret_cast<const float *>(data)[idx];
        default:
            return (static_cast<float>(data[idx]) - zp) * scale;
        }
    }
};

const char *type_name(int type)
{
    switch (type)
    {
    case 2:
        return "uint16";
    case 3:
        return "float32";
    default:
        return "uint8";
    }
}

// ---------------------------------------------------------------------------
// Parameters (from JSON) + layout resolved from the first frame.
// ---------------------------------------------------------------------------
struct Scale
{
    std::string box_name, score_name, kpt_name;
    int h = 0, w = 0;
    float stride_x = 0.f, stride_y = 0.f;
};

struct PoseParams
{
    // user settings
    float score_threshold = 0.5f;
    float iou_threshold = 0.7f;
    float keypoint_threshold = 0.5f;
    int max_detections = 100;
    int num_keypoints = -1; // -1 = auto
    int num_classes = -1;   // -1 = auto
    int reg_max = -1;       // -1 = auto (16 for standard YOLOv8)
    int input_width = -1;   // -1 = auto (largest grid * 8)
    int input_height = -1;
    bool class_agnostic_nms = true;
    Activation score_activation = Activation::AUTO;
    Activation keypoint_activation = Activation::AUTO;
    std::vector<std::string> labels;
    std::vector<std::pair<int, int>> skeleton;
    std::string landmarks_type = "centerpose"; // name hailooverlay / hailo-apps callbacks expect
    bool verbose = true;

    // resolved at runtime
    bool resolved = false;
    bool failed = false;
    std::vector<Scale> scales;
    int K = 0, NC = 0, REG = 0;
    int in_w = 0, in_h = 0;
    std::mutex mutex;
};

template <typename T>
void read_number(const rapidjson::Value &doc, const char *key, T &out)
{
    if (doc.HasMember(key) && doc[key].IsNumber())
        out = static_cast<T>(doc[key].GetDouble());
}

void load_config(PoseParams &p, const std::string &path)
{
    if (path.empty() || path == "NULL")
        return;
    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
        std::cerr << LOG_TAG << "cannot open config " << path << ", using defaults" << std::endl;
        return;
    }
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);
    if (doc.HasParseError() || !doc.IsObject())
    {
        std::cerr << LOG_TAG << "invalid JSON in " << path << ": "
                  << rapidjson::GetParseError_En(doc.GetParseError()) << ", using defaults" << std::endl;
        return;
    }
    read_number(doc, "score_threshold", p.score_threshold);
    read_number(doc, "iou_threshold", p.iou_threshold);
    read_number(doc, "keypoint_threshold", p.keypoint_threshold);
    read_number(doc, "max_detections", p.max_detections);
    read_number(doc, "num_keypoints", p.num_keypoints);
    read_number(doc, "num_classes", p.num_classes);
    read_number(doc, "reg_max", p.reg_max);
    read_number(doc, "input_width", p.input_width);
    read_number(doc, "input_height", p.input_height);
    if (doc.HasMember("class_agnostic_nms") && doc["class_agnostic_nms"].IsBool())
        p.class_agnostic_nms = doc["class_agnostic_nms"].GetBool();
    if (doc.HasMember("verbose") && doc["verbose"].IsBool())
        p.verbose = doc["verbose"].GetBool();
    if (doc.HasMember("score_activation") && doc["score_activation"].IsString())
        p.score_activation = parse_activation(doc["score_activation"].GetString());
    if (doc.HasMember("keypoint_activation") && doc["keypoint_activation"].IsString())
        p.keypoint_activation = parse_activation(doc["keypoint_activation"].GetString());
    if (doc.HasMember("landmarks_type") && doc["landmarks_type"].IsString())
        p.landmarks_type = doc["landmarks_type"].GetString();
    if (doc.HasMember("labels") && doc["labels"].IsArray())
    {
        for (auto &v : doc["labels"].GetArray())
            if (v.IsString())
                p.labels.emplace_back(v.GetString());
    }
    if (doc.HasMember("skeleton") && doc["skeleton"].IsArray())
    {
        for (auto &pair : doc["skeleton"].GetArray())
            if (pair.IsArray() && pair.Size() == 2 && pair[0].IsInt() && pair[1].IsInt())
                p.skeleton.emplace_back(pair[0].GetInt(), pair[1].GetInt());
    }
}

// ---------------------------------------------------------------------------
// Layout resolution
// ---------------------------------------------------------------------------
std::string describe_tensors(std::vector<HailoTensorPtr> &tensors)
{
    std::ostringstream os;
    for (auto &t : tensors)
        os << "\n    " << t->name() << "  " << t->height() << "x" << t->width() << "x" << t->features()
           << "  " << type_name(static_cast<int>(t->format().type));
    return os.str();
}

bool resolve_layout(PoseParams &p, std::vector<HailoTensorPtr> &tensors, std::string &err)
{
    if (tensors.empty() || tensors.size() % 3 != 0)
    {
        err = "expected 3 output tensors per scale (boxes, scores, keypoints), got " +
              std::to_string(tensors.size()) + ":" + describe_tensors(tensors) +
              "\n  If the HEF has on-chip NMS or a single concatenated output, it was not "
              "exported with the 9 raw head convs as end nodes.";
        return false;
    }
    // group by spatial size
    std::map<std::pair<int, int>, std::vector<HailoTensorPtr>> groups;
    for (auto &t : tensors)
        groups[{static_cast<int>(t->height()), static_cast<int>(t->width())}].push_back(t);

    const int want_kp_f = p.num_keypoints > 0 ? p.num_keypoints * 3 : -1;
    const int want_nc_f = p.num_classes > 0 ? p.num_classes : -1;
    const int want_box_f = p.reg_max > 0 ? p.reg_max * 4 : 64;

    int box_f = -1, score_f = -1, kp_f = -1;
    std::vector<Scale> scales;
    for (auto &g : groups)
    {
        auto &ts = g.second;
        if (ts.size() != 3)
        {
            err = "grid " + std::to_string(g.first.first) + "x" + std::to_string(g.first.second) + " has " +
                  std::to_string(ts.size()) + " tensors, expected 3:" + describe_tensors(tensors);
            return false;
        }
        HailoTensorPtr box, score, kpt;
        std::vector<HailoTensorPtr> rest;
        for (auto &t : ts)
        {
            if (!box && static_cast<int>(t->features()) == want_box_f &&
                static_cast<int>(t->features()) != want_kp_f && static_cast<int>(t->features()) != want_nc_f)
                box = t;
            else
                rest.push_back(t);
        }
        if (!box || rest.size() != 2)
        {
            err = "could not find the box (DFL) tensor with " + std::to_string(want_box_f) +
                  " channels in grid " + std::to_string(g.first.first) + "x" + std::to_string(g.first.second) +
                  ". Set reg_max in the model sidecar if your head uses a non-default value." +
                  describe_tensors(tensors);
            return false;
        }
        int f0 = rest[0]->features(), f1 = rest[1]->features();
        int kp_idx = -1;
        if (want_kp_f > 0)
            kp_idx = (f0 == want_kp_f) ? 0 : (f1 == want_kp_f ? 1 : -1);
        else if (want_nc_f > 0)
            kp_idx = (f0 == want_nc_f) ? 1 : (f1 == want_nc_f ? 0 : -1);
        else if ((f0 % 3 == 0) != (f1 % 3 == 0))
            kp_idx = (f0 % 3 == 0) ? 0 : 1;
        else
            kp_idx = (f0 >= f1) ? 0 : 1;
        if (kp_idx < 0)
        {
            err = "channel counts in grid " + std::to_string(g.first.first) + "x" + std::to_string(g.first.second) +
                  " do not match num_keypoints/num_classes from the sidecar:" + describe_tensors(tensors);
            return false;
        }
        kpt = rest[kp_idx];
        score = rest[1 - kp_idx];
        if (kpt->features() % 3 != 0)
        {
            err = "keypoint tensor " + kpt->name() + " has " + std::to_string(kpt->features()) +
                  " channels, not a multiple of 3 (x, y, visibility)";
            return false;
        }
        int bf = box->features(), sf = score->features(), kf = kpt->features();
        if ((box_f > 0 && bf != box_f) || (score_f > 0 && sf != score_f) || (kp_f > 0 && kf != kp_f))
        {
            err = "channel counts differ between scales:" + describe_tensors(tensors);
            return false;
        }
        box_f = bf;
        score_f = sf;
        kp_f = kf;
        Scale s;
        s.box_name = box->name();
        s.score_name = score->name();
        s.kpt_name = kpt->name();
        s.h = g.first.first;
        s.w = g.first.second;
        scales.push_back(s);
    }
    // finest grid first (stride 8, 16, 32 ...)
    std::sort(scales.begin(), scales.end(), [](const Scale &a, const Scale &b) { return a.w > b.w; });
    p.in_w = p.input_width > 0 ? p.input_width : scales.front().w * 8;
    p.in_h = p.input_height > 0 ? p.input_height : scales.front().h * 8;
    for (auto &s : scales)
    {
        s.stride_x = static_cast<float>(p.in_w) / s.w;
        s.stride_y = static_cast<float>(p.in_h) / s.h;
    }
    p.scales = scales;
    p.REG = box_f / 4;
    p.NC = score_f;
    p.K = kp_f / 3;
    return true;
}

// Decide sigmoid-or-not from the actual value range of the tensors.
// Logit outputs always contain strongly negative background values;
// probability outputs are confined to [0, 1].
Activation detect_activation(const std::vector<TensorView> &views, int channel_stride, int channel_offset)
{
    float mn = 1e30f, mx = -1e30f;
    for (auto &v : views)
    {
        size_t n = static_cast<size_t>(v.h) * v.w * v.f;
        for (size_t i = channel_offset; i < n; i += channel_stride)
        {
            float x = v.at(i);
            mn = std::min(mn, x);
            mx = std::max(mx, x);
        }
    }
    return (mn < -0.01f || mx > 1.01f) ? Activation::SIGMOID : Activation::NONE;
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------
struct Candidate
{
    float conf;
    int cls;
    int scale;
    int cell;
    float x1, y1, x2, y2; // normalized to network input
};

float iou(const Candidate &a, const Candidate &b)
{
    float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

struct PoseResult
{
    Candidate c;
    std::vector<float> kx, ky, kv; // normalized to network input
};

std::vector<PoseResult> decode(PoseParams &p, std::vector<HailoTensorPtr> &tensors)
{
    std::map<std::string, HailoTensorPtr> by_name;
    for (auto &t : tensors)
        by_name[t->name()] = t;

    std::vector<TensorView> boxes, scores, kpts;
    for (auto &s : p.scales)
    {
        boxes.emplace_back(by_name.at(s.box_name));
        scores.emplace_back(by_name.at(s.score_name));
        kpts.emplace_back(by_name.at(s.kpt_name));
    }

    if (p.score_activation == Activation::AUTO)
    {
        p.score_activation = detect_activation(scores, 1, 0);
        if (p.verbose)
            std::cerr << LOG_TAG << "class scores: " << activation_name(p.score_activation) << std::endl;
    }
    if (p.keypoint_activation == Activation::AUTO)
    {
        p.keypoint_activation = detect_activation(kpts, 3, 2);
        if (p.verbose)
            std::cerr << LOG_TAG << "keypoint visibility: " << activation_name(p.keypoint_activation) << std::endl;
    }
    const bool score_sig = p.score_activation == Activation::SIGMOID;
    const bool kpt_sig = p.keypoint_activation == Activation::SIGMOID;
    // compare in the raw domain to avoid one exp() per anchor
    const float thr = p.score_threshold;
    const float raw_thr = score_sig ? std::log(thr / (1.f - std::min(thr, 0.999999f))) : thr;

    std::vector<Candidate> cands;
    for (size_t si = 0; si < p.scales.size(); si++)
    {
        const Scale &s = p.scales[si];
        const TensorView &sv = scores[si];
        const int cells = s.h * s.w;
        for (int cell = 0; cell < cells; cell++)
        {
            int best = 0;
            float best_raw = sv.at(static_cast<size_t>(cell) * p.NC);
            for (int c = 1; c < p.NC; c++)
            {
                float v = sv.at(static_cast<size_t>(cell) * p.NC + c);
                if (v > best_raw)
                {
                    best_raw = v;
                    best = c;
                }
            }
            if (best_raw < raw_thr)
                continue;
            Candidate cand;
            cand.conf = score_sig ? sigmoid(best_raw) : best_raw;
            cand.cls = best;
            cand.scale = static_cast<int>(si);
            cand.cell = cell;
            cands.push_back(cand);
        }
    }

    // DFL box decode for surviving anchors
    std::vector<float> bins(p.REG);
    for (auto &cand : cands)
    {
        const Scale &s = p.scales[cand.scale];
        const TensorView &bv = boxes[cand.scale];
        const int row = cand.cell / s.w, col = cand.cell % s.w;
        float dist[4];
        for (int side = 0; side < 4; side++)
        {
            size_t base = static_cast<size_t>(cand.cell) * bv.f + side * p.REG;
            float mx = -1e30f;
            for (int b = 0; b < p.REG; b++)
            {
                bins[b] = bv.at(base + b);
                mx = std::max(mx, bins[b]);
            }
            float sum = 0.f, acc = 0.f;
            for (int b = 0; b < p.REG; b++)
            {
                float e = std::exp(bins[b] - mx);
                sum += e;
                acc += e * b;
            }
            dist[side] = acc / sum;
        }
        const float cx = (col + 0.5f) * s.stride_x, cy = (row + 0.5f) * s.stride_y;
        cand.x1 = (cx - dist[0] * s.stride_x) / p.in_w;
        cand.y1 = (cy - dist[1] * s.stride_y) / p.in_h;
        cand.x2 = (cx + dist[2] * s.stride_x) / p.in_w;
        cand.y2 = (cy + dist[3] * s.stride_y) / p.in_h;
    }

    // NMS (highest score first). stable_sort keeps ties (common with 8-bit
    // quantized scores) in anchor order, matching the NumPy decoder exactly.
    std::stable_sort(cands.begin(), cands.end(), [](const Candidate &a, const Candidate &b) { return a.conf > b.conf; });
    std::vector<Candidate> kept;
    std::vector<bool> removed(cands.size(), false);
    for (size_t i = 0; i < cands.size() && static_cast<int>(kept.size()) < p.max_detections; i++)
    {
        if (removed[i])
            continue;
        kept.push_back(cands[i]);
        for (size_t j = i + 1; j < cands.size(); j++)
        {
            if (removed[j])
                continue;
            if (!p.class_agnostic_nms && cands[i].cls != cands[j].cls)
                continue;
            if (iou(cands[i], cands[j]) > p.iou_threshold)
                removed[j] = true;
        }
    }

    // keypoints only for the final detections
    std::vector<PoseResult> results;
    results.reserve(kept.size());
    for (auto &c : kept)
    {
        const Scale &s = p.scales[c.scale];
        const TensorView &kv = kpts[c.scale];
        const int row = c.cell / s.w, col = c.cell % s.w;
        PoseResult r;
        r.c = c;
        r.kx.resize(p.K);
        r.ky.resize(p.K);
        r.kv.resize(p.K);
        size_t base = static_cast<size_t>(c.cell) * kv.f;
        for (int k = 0; k < p.K; k++)
        {
            float x = kv.at(base + 3 * k), y = kv.at(base + 3 * k + 1), v = kv.at(base + 3 * k + 2);
            r.kx[k] = (x * 2.f + col) * s.stride_x / p.in_w;
            r.ky[k] = (y * 2.f + row) * s.stride_y / p.in_h;
            r.kv[k] = kpt_sig ? sigmoid(v) : v;
        }
        results.push_back(std::move(r));
    }
    return results;
}

inline float clamp01(float v) { return std::min(1.f, std::max(0.f, v)); }

void run(HailoROIPtr roi, PoseParams &p, bool letterbox)
{
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.failed)
        return;
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    if (tensors.empty())
        return;
    if (!p.resolved)
    {
        std::string err;
        if (!resolve_layout(p, tensors, err))
        {
            std::cerr << LOG_TAG << "ERROR: " << err << std::endl;
            p.failed = true;
            return;
        }
        p.resolved = true;
        if (p.verbose)
        {
            std::cerr << LOG_TAG << "model: " << p.K << " keypoints, " << p.NC << " class(es), reg_max " << p.REG
                      << ", input " << p.in_w << "x" << p.in_h << std::endl;
            for (auto &s : p.scales)
                std::cerr << LOG_TAG << "  stride " << s.stride_x << ": boxes=" << s.box_name
                          << " scores=" << s.score_name << " kpts=" << s.kpt_name << std::endl;
            for (auto &pr : p.skeleton)
                if (pr.first >= p.K || pr.second >= p.K)
                    std::cerr << LOG_TAG << "WARNING: skeleton pair (" << pr.first << "," << pr.second
                              << ") out of range for " << p.K << " keypoints" << std::endl;
        }
        std::vector<std::pair<int, int>> valid;
        for (auto &pr : p.skeleton)
            if (pr.first >= 0 && pr.second >= 0 && pr.first < p.K && pr.second < p.K)
                valid.push_back(pr);
        p.skeleton = valid;
    }

    std::vector<PoseResult> results = decode(p, tensors);

    // network-normalized -> frame-normalized
    HailoBBox map = letterbox ? hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox())
                              : HailoBBox(0.f, 0.f, 1.f, 1.f);
    auto mx = [&](float x) { return x * map.width() + map.xmin(); };
    auto my = [&](float y) { return y * map.height() + map.ymin(); };

    for (auto &r : results)
    {
        float x1 = clamp01(mx(r.c.x1)), y1 = clamp01(my(r.c.y1));
        float x2 = clamp01(mx(r.c.x2)), y2 = clamp01(my(r.c.y2));
        float bw = x2 - x1, bh = y2 - y1;
        if (bw <= 1e-6f || bh <= 1e-6f)
            continue;
        std::string label = (r.c.cls < static_cast<int>(p.labels.size())) ? p.labels[r.c.cls]
                            : (p.NC == 1 ? std::string("person") : "class_" + std::to_string(r.c.cls));
        auto det = std::make_shared<HailoDetection>(HailoBBox(x1, y1, bw, bh), r.c.cls, label, r.c.conf);

        std::vector<HailoPoint> points;
        points.reserve(p.K);
        for (int k = 0; k < p.K; k++)
        {
            // landmarks are stored relative to their detection box
            float px = (clamp01(mx(r.kx[k])) - x1) / bw;
            float py = (clamp01(my(r.ky[k])) - y1) / bh;
            points.emplace_back(px, py, clamp01(r.kv[k]));
        }
        det->add_object(std::make_shared<HailoLandmarks>(p.landmarks_type, points, p.keypoint_threshold, p.skeleton));
        hailo_common::add_object(roi, det);
    }
    if (letterbox)
        roi->clear_scaling_bbox();
}
} // namespace

// ---------------------------------------------------------------------------
// hailofilter entry points
// ---------------------------------------------------------------------------
void *init(const std::string config_path, const std::string function_name)
{
    auto *p = new PoseParams();
    load_config(*p, config_path);
    if (p->verbose)
        std::cerr << LOG_TAG << "post-process loaded (" << function_name << "), score>=" << p->score_threshold
                  << " iou=" << p->iou_threshold << " kpt>=" << p->keypoint_threshold << std::endl;
    return p;
}

void free_resources(void *params_void_ptr)
{
    delete reinterpret_cast<PoseParams *>(params_void_ptr);
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    try
    {
        run(roi, *reinterpret_cast<PoseParams *>(params_void_ptr), false);
    }
    catch (const std::exception &e)
    {
        std::cerr << LOG_TAG << "exception in filter: " << e.what() << std::endl;
    }
}

void filter_letterbox(HailoROIPtr roi, void *params_void_ptr)
{
    try
    {
        run(roi, *reinterpret_cast<PoseParams *>(params_void_ptr), true);
    }
    catch (const std::exception &e)
    {
        std::cerr << LOG_TAG << "exception in filter_letterbox: " << e.what() << std::endl;
    }
}
