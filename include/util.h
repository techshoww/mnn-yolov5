#ifndef UTIL_H
#define UTIL_H 1


//#include "opencv2/core/types.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
//using namespace cv;

typedef struct TrackingBox
{
	int frame;
	int id;
	int label;
	float score;
	cv::Rect_<float> box;
}TrackingBox;



double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);


#if 1

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}
#else 
static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}
#endif

#endif