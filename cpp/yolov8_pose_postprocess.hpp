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
#include <string>
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS
void *init(const std::string config_path, const std::string function_name);
void free_resources(void *params_void_ptr);
void filter(HailoROIPtr roi, void *params_void_ptr);
void filter_letterbox(HailoROIPtr roi, void *params_void_ptr);
__END_DECLS
