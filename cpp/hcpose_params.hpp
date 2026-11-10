/**
 * Common base for the parameter objects that hcpose's hailofilter functions get from init().
 * One .so exports a single init()/free_resources() pair, so every function's params derive
 * from this and are deleted through the virtual destructor.
 *
 * Part of hcpose, licensed under AGPL-3.0-or-later (see LICENSE).
 **/
#pragma once
#include <string>

struct HcposeParams
{
    virtual ~HcposeParams() = default;
};

// tiling.cpp
HcposeParams *tiling_make_merge_params(const std::string &config_path);
