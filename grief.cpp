#include "filesystem.h"
#include <Eigen/Dense>
#include <algorithm>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <regex>
#include <streambuf>
#include <string>
#include <vector>

Eigen::Matrix3d read_homography_from_file(const std::string& path) {
    using namespace std;
    ifstream t(path);
    string str((istreambuf_iterator<char>(t)), (istreambuf_iterator<char>()));
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());

    std::istringstream iss(str);
    std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                     std::istream_iterator<std::string>());
    assert(results.size() >= 9);

    Eigen::Matrix3d mat;
    for (int i = 0; i < 9; ++i) {
        mat(i / 3, i % 3) = stod(results[i]);
    }
    return mat;
}

std::vector<Eigen::Matrix3d> load_homographies(const std::string& path) {
    namespace fs = std::filesystem;
    using namespace std;

    regex re("H1to[0-9]+p");
    vector<string> homography_paths;

    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry) &&
            regex_match(entry.path().filename().string(), re)) {
            homography_paths.push_back(entry.path().string());
        }
    }

    std::sort(begin(homography_paths), end(homography_paths), less<string>());

    std::vector<Eigen::Matrix3d> homographies;
    homographies.push_back(Eigen::Matrix3d::Identity());

    Eigen::Matrix3d h;
    for (const auto& path : homography_paths) {
        h = read_homography_from_file(path);
        homographies.push_back(h);
    }
    return homographies;
}

std::vector<cv::Mat> load_images(const std::string& path) {
    namespace fs = std::filesystem;
    using namespace std;

    vector<string> image_paths;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry) &&
            entry.path().extension().string() == ".ppm") {
            image_paths.push_back(entry.path().string());
        }
    }

    std::sort(begin(image_paths), end(image_paths), less<string>());
    std::vector<cv::Mat> images;
    for (const auto& ip : image_paths) {
        images.push_back(cv::imread(ip));
    }
    return images;
}

std::vector<cv::Mat> blur_images(const std::vector<cv::Mat>& images, int ksize,
                                 double sigma) {
    std::vector<cv::Mat> blurred_images;
    for (const auto& img : images) {
        cv::Mat blur_img;
        cv::cvtColor(img, blur_img, cv::COLOR_RGBA2GRAY);
        cv::GaussianBlur(blur_img, blur_img, {ksize, ksize}, sigma, sigma);
        blurred_images.push_back(blur_img);
    }
    return blurred_images;
}

std::vector<cv::KeyPoint> FAST(const cv::Mat& image) {
    auto feature_detector = cv::FastFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    feature_detector->detect(image, keypoints);
    return keypoints;
}

template <int N> class GRIEF {
    Eigen::Matrix<int, N, 4> pattern;

  public:
    GRIEF(const std::string& pattern_type) {
        fmt::print("GRIEF-{} PATTERN TYPE: {}\n", N / 8, pattern_type);

        //std::random_device rd();
        //std::mt19937 gen(rd());
        std::mt19937 gen(0);    // Set random seed for repeatability.

        if (pattern_type == "GI") {
            std::uniform_real_distribution<double> dis(-16, 16);
            pattern = Eigen::Matrix<int, N, 4>::NullaryExpr(
                [&]() { return dis(gen); });
        } else if (pattern_type == "GII") {
            std::normal_distribution<double> dis(0.0, 32.0 / 5.0);
            pattern = Eigen::Matrix<int, N, 4>::NullaryExpr(
                [&]() { return dis(gen); });
        } else if (pattern_type == "GIII") {
            std::normal_distribution<double> disX(0.0, 32.0 / 5.0);
            for (int i = 0; i < pattern.rows(); ++i) {
                pattern(i, 0) = disX(gen);
                pattern(i, 1) = disX(gen);
                std::normal_distribution<double> disYx(pattern(i, 0),
                                                       32.0 / 10.0);
                std::normal_distribution<double> disYy(pattern(i, 1),
                                                       32.0 / 10.0);
                pattern(i, 2) = disYx(gen);
                pattern(i, 3) = disYy(gen);
            }
        } else if (pattern_type == "SYMMETRIC") {
            throw std::runtime_error("SYMMETRIC GRIEF NOT IMPLEMENTED YET.");
        } else {
            throw std::runtime_error("GRIEF PATTERN TYPE NOT RECOGNIZED.");
        }
    }

    void display_pattern(int scale = 8) const {
        int max = pattern.maxCoeff();
        int min = pattern.minCoeff();
        int width = max - min;

        cv::Mat pattern_img(scale * width, scale * width, CV_8UC3,
                            cv::Scalar(0, 0, 0));

        for (int i = 0; i < pattern.rows(); ++i) {
            cv::Point pt0 = {scale * (width / 2 + pattern(i, 0)),
                             scale * (width / 2 + pattern(i, 1))};
            cv::Point pt1 = {scale * (width / 2 + pattern(i, 2)),
                             scale * (width / 2 + pattern(i, 3))};
            cv::line(pattern_img, pt0, pt1, cv::Scalar(255, 255, 255));
        }
        cv::imshow("Pattern", pattern_img);
        cv::waitKey(0);
        cv::destroyWindow("Pattern");
    }

    std::vector<std::bitset<N>>
    describe(const std::vector<cv::KeyPoint>& keypoints,
             const cv::Mat& img) const {
        std::vector<std::bitset<N>> descriptors;

        for (size_t k = 0; k < keypoints.size(); ++k) {
            const auto& kp = keypoints[k];

            std::bitset<N> descriptor(0x0);

            for (int i = 0; i < pattern.rows(); ++i) {
                int x0 = pattern(i, 0) + kp.pt.x;
                int y0 = pattern(i, 1) + kp.pt.y;
                int x1 = pattern(i, 2) + kp.pt.x;
                int y1 = pattern(i, 3) + kp.pt.y;

                x0 = std::clamp<int>(x0, 0, img.cols - 1);
                y0 = std::clamp<int>(y0, 0, img.rows - 1);
                x1 = std::clamp<int>(x1, 0, img.cols - 1);
                y1 = std::clamp<int>(y1, 0, img.rows - 1);

                if (img.at<uchar>(cv::Point(x0, y0)) <
                    img.at<uchar>(cv::Point(x1, y1))) {
                    descriptor.set(i);
                }
            }
            descriptors.push_back(descriptor);
        }
        return descriptors;
    }
};

template <int N>
int hamming_distance(const std::bitset<N>& d0, const std::bitset<N>& d1) {
    return (d0 ^ d1).count();
}

template <int N>
size_t idx_of_closest(const std::bitset<N>& d,
                      const std::vector<std::bitset<N>>& vec) {
    size_t min_idx = 0;
    int min_dist = std::numeric_limits<int>::max();
    for (size_t i = 0; i < vec.size(); ++i) {
        int dist = hamming_distance<N>(d, vec[i]);
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
            if (min_dist == 0) {
                break;
            }
        }
    }
    return min_idx;
}

std::vector<cv::KeyPoint> apply_homography(const std::vector<cv::KeyPoint>& kps,
                                           const Eigen::Matrix3d H) {
    std::vector<cv::KeyPoint> Hkps;
    for (const auto& kp : kps) {
        Eigen::Vector3d xyw = {kp.pt.x, kp.pt.y, 1.0};
        xyw = H * xyw;
        xyw /= xyw[2];
        cv::KeyPoint kp_copy = kp;
        kp_copy.pt.x = xyw[0];
        kp_copy.pt.y = xyw[1];
        Hkps.push_back(kp_copy);
    }
    return Hkps;
}

int main(int argc, char** argv) {
    using namespace std;

    // Select the number of bits in a descriptor.
    const int N = 512;

    if (argc != 2) {
        fmt::print("usage: ./grief <dataset>\n");
        return -1;
    }

    std::string dataset_dir(argv[1]);

    // Load images and homographies from a data directory.
    auto homographies = load_homographies(dataset_dir);
    vector<cv::Mat> images = load_images(dataset_dir);

    // Blur images using parameters from original paper.
    vector<cv::Mat> blurred_images = blur_images(images, 9, 2.0);

    // Construct the GRIEF object using N-bit descriptors
    // and the GI pattern type from the original BRIEF paper.
    GRIEF<N> grief("GI");
    grief.display_pattern();

    // Extract keypoints from the first image.
    const auto kp0 = FAST(images[0]);
    // Generate descriptors for those keypoints.
    const auto descr0 = grief.describe(kp0, blurred_images[0]);

    // For each remaining image...
    for (size_t i = 1; i < images.size(); ++i) {
        // Transform image 0 keypoints into this image.
        auto kp1 = apply_homography(kp0, homographies[i]);
        // Generate descriptors for those keypoints.
        auto descr1 = grief.describe(kp1, blurred_images[i]);

        // Compute the recognition rate, the percentage of
        // descriptors that are the same in both images.
        double recog_rate = 0.0;
        for (size_t d = 0; d < descr0.size(); ++d) {
            int d0 = hamming_distance<N>(descr0[d], descr1[d]);
            int idx = idx_of_closest<N>(descr0[d], descr1);
            int d1 = hamming_distance<N>(descr0[d], descr1[idx]);
            recog_rate += (d0 == d1);
        }
        recog_rate /= descr0.size();

        fmt::print("1|{} RECOGNITION RATE: {}\n", i+1, recog_rate);
    }

    return 0;
}
