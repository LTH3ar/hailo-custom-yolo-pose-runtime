/**
 * Generic YOLOv8-pose post-process for Hailo `hailofilter`.
 *
 * Works for any keypoint count / class count. Everything that can be read
 * from the output tensors (keypoint count, class count, DFL bins, strides,
 * tensor order, uint8/uint16 format, whether sigmoid is already applied
 * inside the HEF) is detected at runtime on the first frame.
 *
 * Optional settings (thresholds, skeleton, labels) come from a JSON file
 * passed through `hailofilter config-path=...`.
 *
 * Part of hcpose, licensed under AGPL-3.0-or-later (see LICENSE).
 **/
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

class HailoMat; // only passed through; no OpenCV dependency needed

__BEGIN_DECLS
void *init(const std::string config_path, const std::string function_name);
void free_resources(void *params_void_ptr);
// hailofilter: pose post-process for the whole frame or one crop/tile
void filter(HailoROIPtr roi, void *params_void_ptr);
void filter_letterbox(HailoROIPtr roi, void *params_void_ptr);
// tiling (tiling.cpp): hailocropper crop function and the hailofilter that merges tiles
std::vector<HailoROIPtr> create_tile_crops(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
void merge_tiles(HailoROIPtr roi, void *params_void_ptr);
__END_DECLS
