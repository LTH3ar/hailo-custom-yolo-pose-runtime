// Feeds saved output tensors through the post-process .so exactly as hailofilter
// would (init -> filter/filter_letterbox -> free_resources) and prints the
// resulting detections as JSON. Used by tests/test_cpp_vs_numpy.py.
//
// usage: cpp_harness <manifest.txt> <config.json> <letterbox 0|1> [xmin ymin w h scaling bbox [roi bbox]]
// manifest line: name h w f type(1=u8,2=u16,3=f32) zp scale path
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "hailo_common.hpp"
#include "hailo_objects.hpp"

extern "C" {
void *init(const std::string config_path, const std::string function_name);
void free_resources(void *p);
void filter(HailoROIPtr roi, void *p);
void filter_letterbox(HailoROIPtr roi, void *p);
}

int main(int argc, char **argv)
{
    if (argc < 4)
        return 2;
    std::ifstream man(argv[1]);
    std::vector<std::vector<uint8_t>> storage;
    std::vector<HailoTensorPtr> tensors;
    std::string line;
    while (std::getline(man, line))
    {
        std::istringstream is(line);
        std::string name, path;
        uint32_t h, w, f;
        int type;
        float zp, scale;
        if (!(is >> name >> h >> w >> f >> type >> zp >> scale >> path))
            continue;
        std::ifstream bin(path, std::ios::binary);
        storage.emplace_back((std::istreambuf_iterator<char>(bin)), std::istreambuf_iterator<char>());
        hailo_tensor_metadata_t meta{};
        std::strncpy(meta.name, name.c_str(), sizeof(meta.name) - 1);
        meta.format.type = static_cast<HailoTensorFormatType>(type);
        meta.format.is_nms = false;
        meta.shape.height = h;
        meta.shape.width = w;
        meta.shape.features = f;
        meta.quant_info.qp_zp = zp;
        meta.quant_info.qp_scale = scale;
        tensors.push_back(std::make_shared<HailoTensor>(storage.back().data(), meta));
    }
    bool letterbox = std::string(argv[3]) == "1";
    void *params = init(argv[2], letterbox ? "filter_letterbox" : "filter");
    for (int frame = 0; frame < 2; frame++) // second frame checks cached layout path
    {
        // optional argv[8..11]: the ROI's own bbox (a tile); detections stay relative to it
        HailoBBox roi_box = argc >= 12 ? HailoBBox(std::stof(argv[8]), std::stof(argv[9]), std::stof(argv[10]), std::stof(argv[11]))
                                       : HailoBBox(0.f, 0.f, 1.f, 1.f);
        auto roi = std::make_shared<HailoROI>(roi_box);
        if (argc >= 8)
            roi->set_scaling_bbox(HailoBBox(std::stof(argv[4]), std::stof(argv[5]), std::stof(argv[6]), std::stof(argv[7])));
        for (auto &t : tensors)
            roi->add_tensor(t);
        if (letterbox)
            filter_letterbox(roi, params);
        else
            filter(roi, params);
        if (frame == 0)
            continue;
        std::cout << "[";
        bool first = true;
        for (auto &det : hailo_common::get_hailo_detections(roi))
        {
            auto b = det->get_bbox();
            std::cout << (first ? "" : ",") << "{\"box\":[" << b.xmin() << "," << b.ymin() << "," << b.xmax() << ","
                      << b.ymax() << "],\"conf\":" << det->get_confidence() << ",\"cls\":" << det->get_class_id()
                      << ",\"label\":\"" << det->get_label() << "\",\"kpts\":[";
            first = false;
            auto lms = det->get_objects_typed(HAILO_LANDMARKS);
            if (!lms.empty())
            {
                auto lm = std::dynamic_pointer_cast<HailoLandmarks>(lms[0]);
                bool f2 = true;
                for (auto &p : lm->get_points())
                {
                    std::cout << (f2 ? "" : ",") << "[" << p.x() * b.width() + b.xmin() << "," << p.y() * b.height() + b.ymin()
                              << "," << p.confidence() << "]";
                    f2 = false;
                }
                std::cout << "],\"pairs\":" << lm->get_pairs().size() << ",\"kthr\":" << lm->get_threshold();
            }
            else
                std::cout << "]";
            std::cout << "}";
        }
        std::cout << "]" << std::endl;
    }
    free_resources(params);
    return 0;
}
