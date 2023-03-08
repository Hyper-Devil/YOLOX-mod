#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

#define DEVICE 0 // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

using namespace nvinfer1;

class Detect
{
public:
    Detect();
    ~Detect();

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    struct bbox_t
    {
        unsigned int x, y, w, h;     // (x,y) - top-left corner, (w, h) - width & height of bounded box
        float prob;                  // confidence - probability that the object was found correctly
        unsigned int obj_id;         // class of object - from range [0, classes-1]
        unsigned int track_id;       // tracking id for video (0 - untracked, 1 - inf - tracked object)
        unsigned int frames_counter; // counter of frames on which the object was detected
        float x_3d, y_3d, z_3d;      // center of object (in Meters) if ZED 3D Camera is used
    };

    cv::Mat static_resize(cv::Mat &img);
    static void generate_grids_and_stride(std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);
    static inline float intersection_area(const Object &a, const Object &b);
    static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right);
    static void qsort_descent_inplace(std::vector<Object> &objects);
    static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold);
    static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float *feat_blob, float prob_threshold, std::vector<Object> &objects);
    float *blobFromImage(cv::Mat &img);
    static void decode_outputs(float *prob, std::vector<Object> &objects, float scale, const int img_w, const int img_h);
    static void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects, std::string f);
    void doInference(IExecutionContext &context, float *input, float *output, const int output_size, cv::Size input_shape);
    static const char *class_names[];
    const static float color_list[80][3];
    
    void detect_forward(cv::Mat &img);
    void convert_object2bbox(std::vector<Object> &objects, std::vector<bbox_t> &bbox_detected);

    ICudaEngine *engine;
    IRuntime *runtime;
    IExecutionContext *context;
    float *blob;
    static float *prob;
    int output_size;
    float scale;
    nvinfer1::Dims out_dims;
    int img_w,img_h;
    std::vector<Object> objects; //得到的直接结果
    std::vector<bbox_t> bbox_detected; //需要转换为darknet格式

};