/**
 * Tiling for the GStreamer engine: extra zoomed-in passes over the far rows.
 *
 *   hailocropper so-path=<this .so> function-name=create_tile_crops
 *       -> one crop for the whole frame plus one per tile (tiles come from the
 *          HCPOSE_TILES environment variable, set by hcpose: "x,y,w,h;x,y,w,h;..."
 *          normalized to the frame). Tile ROIs are attached to the frame ROI, so
 *          the pose post-process on each tile adds its detections (tile-relative)
 *          to an object that is still reachable after hailoaggregator.
 *   hailofilter so-path=<this .so> function-name=merge_tiles   (after hailoaggregator)
 *       -> maps tile detections to frame coordinates, drops tile detections cut
 *          by an inner tile edge, merges duplicates across passes, removes the
 *          tile ROIs. Same rules as hcpose/tiling.py (python engine).
 *
 * Part of hcpose, licensed under AGPL-3.0-or-later (see LICENSE).
 **/
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hcpose_params.hpp"
#include "yolov8_pose_postprocess.hpp"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

namespace
{
struct MergeParams : HcposeParams
{
    float frame_w = 1280.f, frame_h = 720.f; // only ratios matter, but keypoint distances use pixels
    float iou = 0.5f;
    float ios = 0.7f;
    float edge_margin = 0.02f;
    float kpt_thr = 0.5f;
};

struct Box
{
    float x1, y1, x2, y2; // pixels
};

struct Cand
{
    HailoDetectionPtr det;
    int src; // -1 = full frame, else tile index
    Box box;
    std::vector<float> kx, ky, kv; // pixels
};

std::vector<HailoBBox> parse_tiles(const char *spec)
{
    std::vector<HailoBBox> tiles;
    if (spec == nullptr)
        return tiles;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ';'))
    {
        float v[4];
        int n = 0;
        std::stringstream is(item);
        std::string num;
        while (n < 4 && std::getline(is, num, ','))
            v[n++] = std::strtof(num.c_str(), nullptr);
        if (n == 4 && v[2] > 0.f && v[3] > 0.f)
            tiles.emplace_back(v[0], v[1], v[2], v[3]);
    }
    return tiles;
}

inline float area(const Box &b) { return std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1); }

void overlap(const Box &a, const Box &b, float &iou, float &ios)
{
    float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float inter = iw * ih;
    float aa = area(a), ab = area(b);
    iou = inter / std::max(aa + ab - inter, 1e-9f);
    ios = inter / std::max(std::min(aa, ab), 1e-9f);
}

bool same_person(const Cand &a, const Cand &b, const MergeParams &p)
{
    float o_iou, o_ios;
    overlap(a.box, b.box, o_iou, o_ios);
    if (o_iou >= p.iou)
        return true;
    if (o_ios < p.ios)
        return false;
    // one box inside the other: only the same person if the visible keypoints agree,
    // because a back-row student's box often lies inside the front student's box
    size_t n = std::min(a.kv.size(), b.kv.size());
    double sum = 0.0;
    int common = 0;
    for (size_t k = 0; k < n; k++)
    {
        if (a.kv[k] >= p.kpt_thr && b.kv[k] >= p.kpt_thr)
        {
            sum += std::hypot(a.kx[k] - b.kx[k], a.ky[k] - b.ky[k]);
            common++;
        }
    }
    if (common < 3)
        return false;
    float sa = std::sqrt(std::min(area(a.box), area(b.box)));
    return sum / common < 0.2 * sa;
}

bool cut_by_inner_edge(const HailoBBox &d, const HailoBBox &t, float margin)
{
    const float e = 1e-6f;
    float mx = margin * t.width(), my = margin * t.height();
    float tx2 = t.xmin() + t.width(), ty2 = t.ymin() + t.height();
    return (t.xmin() > e && d.xmin() <= t.xmin() + mx) || (tx2 < 1.f - e && d.xmax() >= tx2 - mx) ||
           (t.ymin() > e && d.ymin() <= t.ymin() + my) || (ty2 < 1.f - e && d.ymax() >= ty2 - my);
}

Cand make_cand(const HailoDetectionPtr &det, const HailoBBox &frame_box, int src, const MergeParams &p)
{
    Cand c;
    c.det = det;
    c.src = src;
    c.box = {frame_box.xmin() * p.frame_w, frame_box.ymin() * p.frame_h, frame_box.xmax() * p.frame_w,
             frame_box.ymax() * p.frame_h};
    auto lms = det->get_objects_typed(HAILO_LANDMARKS);
    if (!lms.empty())
    {
        auto lm = std::dynamic_pointer_cast<HailoLandmarks>(lms[0]);
        for (auto &pt : lm->get_points())
        {
            c.kx.push_back((frame_box.xmin() + pt.x() * frame_box.width()) * p.frame_w);
            c.ky.push_back((frame_box.ymin() + pt.y() * frame_box.height()) * p.frame_h);
            c.kv.push_back(pt.confidence());
        }
    }
    return c;
}

template <typename T>
void read_num(const rapidjson::Value &v, const char *key, T &out)
{
    if (v.HasMember(key) && v[key].IsNumber())
        out = static_cast<T>(v[key].GetDouble());
}
} // namespace

HcposeParams *tiling_make_merge_params(const std::string &config_path)
{
    auto *p = new MergeParams();
    std::ifstream ifs(config_path);
    if (ifs.is_open())
    {
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);
        if (!doc.HasParseError() && doc.IsObject() && doc.HasMember("tile_merge") && doc["tile_merge"].IsObject())
        {
            const auto &m = doc["tile_merge"];
            read_num(m, "frame_width", p->frame_w);
            read_num(m, "frame_height", p->frame_h);
            read_num(m, "iou", p->iou);
            read_num(m, "ios", p->ios);
            read_num(m, "edge_margin", p->edge_margin);
            read_num(m, "kpt_thr", p->kpt_thr);
        }
    }
    return p;
}

std::vector<HailoROIPtr> create_tile_crops(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    (void)image;
    std::vector<HailoROIPtr> crops;
    crops.emplace_back(roi); // the whole frame, as libwhole_buffer.so does
    for (auto &t : parse_tiles(std::getenv("HCPOSE_TILES")))
    {
        auto tile = std::make_shared<HailoROI>(t);
        roi->add_object(tile);
        crops.emplace_back(tile);
    }
    return crops;
}

void merge_tiles(HailoROIPtr roi, void *params_void_ptr)
{
    const MergeParams &p = *static_cast<MergeParams *>(static_cast<HcposeParams *>(params_void_ptr));
    try
    {
        std::vector<Cand> cands;
        std::vector<HailoROIPtr> tiles;
        for (auto &obj : roi->get_objects())
        {
            if (obj->get_type() == HAILO_DETECTION)
            {
                auto det = std::dynamic_pointer_cast<HailoDetection>(obj);
                cands.push_back(make_cand(det, det->get_bbox(), -1, p));
            }
            else if (obj->get_type() == HAILO_ROI)
            {
                tiles.push_back(std::dynamic_pointer_cast<HailoROI>(obj));
            }
        }
        if (tiles.empty())
            return;
        for (size_t ti = 0; ti < tiles.size(); ti++)
        {
            HailoBBox tb = tiles[ti]->get_bbox();
            for (auto &obj : tiles[ti]->get_objects_typed(HAILO_DETECTION))
            {
                auto det = std::dynamic_pointer_cast<HailoDetection>(obj);
                HailoBBox fb = hailo_common::create_flattened_bbox(det->get_bbox(), tb);
                if (cut_by_inner_edge(fb, tb, p.edge_margin))
                    continue;
                det->set_bbox(fb);
                det->clear_scaling_bbox();
                cands.push_back(make_cand(det, fb, static_cast<int>(ti), p));
            }
        }
        std::stable_sort(cands.begin(), cands.end(),
                         [](const Cand &a, const Cand &b) { return a.det->get_confidence() > b.det->get_confidence(); });
        std::vector<const Cand *> kept;
        for (auto &c : cands)
        {
            bool dup = false;
            for (auto *k : kept)
            {
                if (k->src != c.src && same_person(c, *k, p))
                {
                    dup = true;
                    break;
                }
            }
            if (!dup)
                kept.push_back(&c);
        }
        roi->remove_objects_typed(HAILO_DETECTION);
        roi->remove_objects_typed(HAILO_ROI);
        for (auto *k : kept)
            roi->add_object(k->det);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[hcpose] exception in merge_tiles: " << e.what() << std::endl;
    }
}
