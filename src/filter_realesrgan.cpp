#include "filter_realesrgan.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>

#include <spdlog/spdlog.h>

extern "C" {
#include <libavutil/hash.h>
}

#include "conversions.h"
#include "fsutils.h"
#include "logger_manager.h"

namespace video2x {
namespace processors {

FilterRealesrgan::FilterRealesrgan(
    int gpuid,
    bool tta_mode,
    int scaling_factor,
    int noise_level,
    const fsutils::StringType model_name
)
    : realesrgan_(nullptr),
      gpuid_(gpuid),
      tta_mode_(tta_mode),
      scaling_factor_(scaling_factor),
      noise_level_(noise_level),
      model_name_(std::move(model_name)) {}

FilterRealesrgan::~FilterRealesrgan() {
    delete realesrgan_;
    realesrgan_ = nullptr;
}

int FilterRealesrgan::init(AVCodecContext* dec_ctx, AVCodecContext* enc_ctx, AVBufferRef*) {
    // Construct the model paths using std::filesystem
    std::filesystem::path model_param_path;
    std::filesystem::path model_bin_path;

    fsutils::StringType param_file_name = model_name_;
    fsutils::StringType bin_file_name = model_name_;

    if (model_name_ == STR("realesr-generalv3") && noise_level_ > 0) {
        param_file_name += STR("-wdn");
        bin_file_name += STR("-wdn");
    }

    param_file_name += STR("-x") + fsutils::to_string_type(scaling_factor_) + STR(".param");
    bin_file_name += STR("-x") + fsutils::to_string_type(scaling_factor_) + STR(".bin");

    // Find the model paths by model name if provided
    model_param_path = std::filesystem::path(STR("models")) / STR("realesrgan") / param_file_name;
    model_bin_path = std::filesystem::path(STR("models")) / STR("realesrgan") / bin_file_name;

    // Get the full paths using a function that possibly modifies or validates the path
    std::optional<std::filesystem::path> model_param_full_path =
        fsutils::find_resource(model_param_path);
    std::optional<std::filesystem::path> model_bin_full_path =
        fsutils::find_resource(model_bin_path);

    // Check if the model files exist
    if (!model_param_full_path.has_value()) {
        logger()->error("Real-ESRGAN model param file not found: {}", model_param_path.u8string());
        return -1;
    }
    if (!model_bin_full_path.has_value()) {
        logger()->error("Real-ESRGAN model bin file not found: {}", model_bin_path.u8string());
        return -1;
    }

    // Create a new Real-ESRGAN instance
    realesrgan_ = new RealESRGAN(gpuid_, tta_mode_);

    // Store the time bases
    in_time_base_ = dec_ctx->time_base;
    out_time_base_ = enc_ctx->time_base;
    out_pix_fmt_ = enc_ctx->pix_fmt;

    // Load the model
    const auto load_start = std::chrono::steady_clock::now();
    if (realesrgan_->load(model_param_full_path.value(), model_bin_full_path.value()) != 0) {
        logger()->error("Failed to load Real-ESRGAN model");
        return -1;
    }
    logger()->debug("[art-baseline] model_load_ms={:.3f}",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count());

    // Set Real-ESRGAN parameters
    realesrgan_->scale = scaling_factor_;
    realesrgan_->prepadding = 10;

    // Calculate tilesize based on GPU heap budget
    uint32_t heap_budget = ncnn::get_gpu_device(gpuid_)->get_heap_budget();
    if (heap_budget > 1900) {
        realesrgan_->tilesize = 200;
    } else if (heap_budget > 550) {
        realesrgan_->tilesize = 100;
    } else if (heap_budget > 190) {
        realesrgan_->tilesize = 64;
    } else {
        realesrgan_->tilesize = 32;
    }
    logger()->debug("[art-baseline] gpu={} heap_budget_mb={} tile={} scale={} prepadding={} tta={}",
        gpuid_, heap_budget, realesrgan_->tilesize, realesrgan_->scale,
        realesrgan_->prepadding, tta_mode_);

    return 0;
}

int FilterRealesrgan::filter(AVFrame* in_frame, AVFrame** out_frame) {
    int ret;

    // Convert the input frame to BGR24
    ncnn::Mat in_mat = conversions::avframe_to_ncnn_mat(in_frame);
    if (in_mat.empty()) {
        logger()->error("Failed to convert AVFrame to ncnn::Mat");
        return -1;
    }

    // Allocate space for output ncnn::Mat
    int output_width = in_mat.w * realesrgan_->scale;
    int output_height = in_mat.h * realesrgan_->scale;
    ncnn::Mat out_mat = ncnn::Mat(output_width, output_height, static_cast<size_t>(3), 3);

    const auto inference_start = std::chrono::steady_clock::now();
    ret = realesrgan_->process(in_mat, out_mat);
    logger()->debug("[art-baseline] pts={} inference_ms={:.3f} result={}", in_frame->pts,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inference_start).count(), ret);
    if (ret != 0) {
        logger()->error("Real-ESRGAN processing failed");
        return ret;
    }
    if (logger()->should_log(spdlog::level::debug)) {
        AVHashContext* hash = nullptr;
        if (av_hash_alloc(&hash, "sha256") < 0) {
            return AVERROR(ENOMEM);
        }
        av_hash_init(hash);
        av_hash_update(hash, static_cast<const uint8_t*>(out_mat.data),
            static_cast<size_t>(output_width) * output_height * 3);
        uint8_t digest[65];
        av_hash_final_hex(hash, digest, sizeof(digest));
        av_hash_freep(&hash);
        logger()->debug("[art-baseline] pts={} pixel_sha256={}", in_frame->pts,
            reinterpret_cast<const char*>(digest));
    }

    // Convert ncnn::Mat to AVFrame
    *out_frame = conversions::ncnn_mat_to_avframe(out_mat, out_pix_fmt_);
    if (*out_frame == nullptr) {
        logger()->error("Failed to convert ncnn::Mat to AVFrame");
        return AVERROR(ENOMEM);
    }

    // Rescale PTS to encoder's time base
    (*out_frame)->pts = av_rescale_q(in_frame->pts, in_time_base_, out_time_base_);

    // Return the processed frame to the caller
    return ret;
}

void FilterRealesrgan::get_output_dimensions(
    const ProcessorConfig&,
    int in_width,
    int in_height,
    int& out_width,
    int& out_height
) const {
    out_width = in_width * scaling_factor_;
    out_height = in_height * scaling_factor_;
}

}  // namespace processors
}  // namespace video2x
