// Runs create_tile_crops + merge_tiles exactly as the GStreamer pipeline does and prints
// the merged detections, so tests/test_tiling.py can compare them with hcpose/tiling.py.
//
// usage: tile_harness <scenario.json> <config.json>
// scenario: {"W":..,"H":.., "tiles":"x,y,w,h;..." (normalized, also exported as HCPOSE_TILES),
//            "full": [det...], "tiled": [[det...] per tile]}
// det: {"box":[x1,y1,x2,y2] (pixels: frame for "full", tile for "tiled"), "score":s,
//       "kpts":[[x,y,v],...] (same pixel space as box)}
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "hailo_common.hpp"
#include "hailo_objects.hpp"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

class HailoMat;
extern "C" {
void *init(const std::string config_path, const std::string function_name);
void free_resources(void *p);
std::vector<HailoROIPtr> create_tile_crops(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
void merge_tiles(HailoROIPtr roi, void *p);
}

static HailoDetectionPtr make_det(const rapidjson::Value &d, float W, float H)
{
    const auto &b = d["box"];
    float x1 = b[0].GetFloat() / W, y1 = b[1].GetFloat() / H, x2 = b[2].GetFloat() / W, y2 = b[3].GetFloat() / H;
    float bw = x2 - x1, bh = y2 - y1;
    auto det = std::make_shared<HailoDetection>(HailoBBox(x1, y1, bw, bh), 0, "person", d["score"].GetFloat());
    std::vector<HailoPoint> pts;
    for (auto &k : d["kpts"].GetArray())
        pts.emplace_back((k[0].GetFloat() / W - x1) / bw, (k[1].GetFloat() / H - y1) / bh, k[2].GetFloat());
    det->add_object(std::make_shared<HailoLandmarks>("centerpose", pts, 0.5f));
    return det;
}

int main(int argc, char **argv)
{
    if (argc < 3)
        return 2;
    std::ifstream ifs(argv[1]);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document sc;
    sc.ParseStream(isw);
    float W = sc["W"].GetFloat(), H = sc["H"].GetFloat();
    setenv("HCPOSE_TILES", sc["tiles"].GetString(), 1);

    auto roi = std::make_shared<HailoROI>(HailoBBox(0.f, 0.f, 1.f, 1.f));
    auto crops = create_tile_crops(nullptr, roi);
    if (crops.size() != 1 + sc["tiled"].Size() || crops[0] != roi)
    {
        std::cerr << "unexpected crops: " << crops.size() << std::endl;
        return 3;
    }
    // what the pose post-process would add to each crop (relative to that crop)
    for (auto &d : sc["full"].GetArray())
        hailo_common::add_object(roi, make_det(d, W, H));
    for (size_t t = 1; t < crops.size(); t++)
    {
        HailoBBox tb = crops[t]->get_bbox();
        for (auto &d : sc["tiled"][t - 1].GetArray())
            hailo_common::add_object(crops[t], make_det(d, tb.width() * W, tb.height() * H));
    }
    void *params = init(argv[2], "merge_tiles");
    merge_tiles(roi, params);
    free_resources(params);

    std::cout << "[";
    bool first = true;
    for (auto &obj : roi->get_objects())
    {
        if (obj->get_type() == HAILO_ROI)
        {
            std::cerr << "tile ROI left behind" << std::endl;
            return 4;
        }
        auto det = std::dynamic_pointer_cast<HailoDetection>(obj);
        if (!det)
            continue;
        auto b = det->get_bbox();
        std::cout << (first ? "" : ",") << "{\"box\":[" << b.xmin() * W << "," << b.ymin() * H << "," << b.xmax() * W
                  << "," << b.ymax() * H << "],\"score\":" << det->get_confidence() << ",\"kpts\":[";
        first = false;
        auto lm = std::dynamic_pointer_cast<HailoLandmarks>(det->get_objects_typed(HAILO_LANDMARKS)[0]);
        bool f2 = true;
        for (auto &p : lm->get_points())
        {
            std::cout << (f2 ? "" : ",") << "[" << (b.xmin() + p.x() * b.width()) * W << ","
                      << (b.ymin() + p.y() * b.height()) * H << "," << p.confidence() << "]";
            f2 = false;
        }
        std::cout << "]}";
    }
    std::cout << "]" << std::endl;
    return 0;
}
