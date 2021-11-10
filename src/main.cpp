#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <fstream>
#include <chrono>

double blockSigma(cv::Mat & m, int i, int j, int block_size);
double blockCovariance(cv::Mat & m1, cv::Mat & m2, int i, int j, int block_size);

// DISPARITY IMAGES

cv::Mat matchStereoSSDNaive(cv::Mat img1, cv::Mat img2, int patch_size = 2, int disparity = 2) {
    // https://github.com/davechristian/Simple-SSD-Stereo/blob/main/stereomatch_SSD.py
    if ( (img1.rows == img2.rows) || (img1.cols == img2.cols) ) {
        cv::Mat SSD(img1.rows, img1.cols, CV_8U);
        int patch_half = 1;
        if(patch_size > 1){
            patch_half = patch_size / 2;
        }
        unsigned char disparity_adjust = 255 / disparity;
        for(int i = patch_half+1; i < img1.rows-patch_half; i++){
#pragma omp parallel for
            for(int j = patch_half+1; j < img1.cols-patch_half; j++){
                unsigned char best_offset = 0;
                int prev_ssd = 999999999;
                for(int k = 0; k < disparity; k++){
                    int curr_ssd = 0;
                    int temp_ssd;
                    for(int v = -patch_half; v<patch_half; v++){
                        for(int u = -patch_half; u<patch_half; u++) {
                            temp_ssd = img1.at<uchar>(i+v,j+u) - img2.at<uchar>(i+v,j+u - k);
                            curr_ssd += temp_ssd*temp_ssd;
                        }
                    }
                    if(curr_ssd < prev_ssd){
                        prev_ssd = curr_ssd;
                        best_offset = k;
                    }
                }
                SSD.at<uchar>(i,j) = best_offset * disparity_adjust;
            }
        }
        return SSD;
    }
    else{
        cv::Mat SSD(0, 0, CV_16U);
        return SSD;
    }
}

std::tuple<cv::Mat, cv::Mat> dynamicStereoMatching(cv::Mat img1, cv::Mat img2, int occlusion){
    // http://www.cs.umd.edu/~djacobs/CMSC426/Slides/stereo_algo.pdf
    cv::Mat disparityImageLeft(img1.rows, img1.cols, CV_8U, cv::Scalar(0));
    cv::Mat disparityImageRight(img2.rows, img2.cols, CV_8U, cv::Scalar(0));

    std::vector<std::vector<int>> cost(img1.cols,std::vector<int>(img1.cols, 0));
    std::vector<std::vector<int>> path(img1.cols,std::vector<int>(img1.cols, 0));

    for(int k = 0; k < img1.rows; k++){
#pragma omp parallel for
        for(int i = 0; i < img1.cols; i++) {
            cost[i][0] = i*occlusion;
            cost[0][i] = i*occlusion;
        }
#pragma omp parallel for
        for(int i = 1; i < img1.cols; i++){
            for(int j = 1; j < img1.cols; j++){
                int t1 = img1.at<uchar>(k, i);
                int t2 = img2.at<uchar>(k, j);
                unsigned c = std::abs(t1-t2);
                unsigned min_1 = cost[i-1][j-1] + c;
                unsigned min_2 = cost[i-1][j]+occlusion;
                unsigned min_3 = cost[i][j-1]+occlusion;
                unsigned cmin = std::min({min_1, min_2, min_3});
                cost[i][j] = (int)cmin;
                if(cmin == min_1){
                    path[i][j] = 1;
                }
                else if(cmin == min_2){
                    path[i][j] = 2;
                }
                else if(cmin == min_3){
                    path[i][j] = 3;
                }
            }
        }
        int p = img1.cols - 1;
        int q = img1.cols - 1;
        while((p!=0) && (q!=0)){
            if(path[p][q] == 1){
                disparityImageLeft.at<uchar>(k,p) = unsigned(std::abs(p-q));
                disparityImageRight.at<uchar>(k,q) = unsigned(std::abs(p-q));
                p--;
                q--;
            }
            else if(path[p][q] == 2){
                disparityImageLeft.at<uchar>(k,p) = unsigned(NAN);
                p--;
            }
            else if(path[p][q] == 3){
                disparityImageRight.at<uchar>(k,q) = unsigned(NAN);
                q--;
            }
        }
    }
    return {disparityImageLeft, disparityImageRight};
}

std::tuple<cv::Mat, cv::Mat> openCvMethodBM(cv::Mat img1, cv::Mat img2, int numDisparities, int blockSize){
    cv::Mat disparityImageLeft(img1.rows, img1.cols, CV_8U, cv::Scalar(254));
    cv::Mat disparityImageRight(img2.rows, img2.cols, CV_8U, cv::Scalar(254));
    cv::Mat convertedImageLeft(img2.rows, img2.cols, CV_8U);
    cv::Mat convertedImageRight(img2.rows, img2.cols, CV_8U);
    cv::resize(img1, convertedImageLeft, cv::Size(), 0.5, 0.5);
    cv::resize(img2, convertedImageRight, cv::Size(), 0.5, 0.5);
    cv::Ptr<cv::StereoBM> stereoMatchLeft = cv::StereoBM::create(numDisparities, blockSize);
    cv::Ptr<cv::StereoBM> stereoMatchRight = cv::StereoBM::create(numDisparities, blockSize);

    stereoMatchLeft->compute(convertedImageLeft, convertedImageRight, disparityImageLeft);
    stereoMatchRight->compute(convertedImageRight, convertedImageLeft, disparityImageRight);
    return {disparityImageLeft, disparityImageRight};
}

// ----------------------------------------------------------------------------------------------------------

// COMPARISON TECHNIQUES

// Structural Similarity Index Measure
double compareImagesSSIM(cv::Mat img1, cv::Mat img2, int block_size, float c1 = 6.5025, float c2 = 58.5225){
    double ssim = 0;

    for(int i = 0; i < (int)(img1.rows / block_size); i++){
        for(int j = 0; j < (int)(img2.rows / block_size); j++){
            int k = i * block_size;
            int l = j * block_size;

            double avg_left = cv::mean(img1(cv::Range(i, i + block_size), cv::Range(j, j + block_size)))[0];
            double avg_right = cv::mean(img2(cv::Range(i, i + block_size), cv::Range(j, j + block_size)))[0];
            double sigma_left = blockSigma(img1, k, l, block_size);
            double sigma_right = blockSigma(img2, k, l, block_size);
            double sigma_both = blockCovariance(img1, img2, k, l, block_size);

            ssim += unsigned(((2 * avg_left * avg_right + c1) * (2 * sigma_both + c2))
                    / ((avg_left * avg_left + avg_right * avg_right + c1) * (sigma_left * sigma_left + sigma_right * sigma_right + c2)));
        }
    }

    return ssim;
}

double meanSquareError(cv::Mat img1, cv::Mat img2){
    // https://stackoverflow.com/questions/29973957/mse-for-two-vec3b-images-in-opencv
    double MSE = 0;
    for(int i = 0; i < img1.rows; i++)
        for(int j = 0; j < img2.cols; j++)
            MSE += std::sqrt(std::pow(img1.at<uchar>(i,j), 2) - std::pow(img2.at<uchar>(i,j), 2));
    MSE /= img1.rows * img2.cols;
    return MSE;
}

double peakSignalToNoiseRatio(cv::Mat img1, cv::Mat img2, int block_size, int img_max=255){
    return (10 * std::log10((img_max*img_max) / meanSquareError(img1, img2)));
}

// ----------------------------------------------------------------------------------------------------------------


std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> analyseStereoPair(cv::Mat img1, cv::Mat img2,
                                                                 cv::Mat ground, std::string path_to_save,
                                                                 int patch_size, int disparity,
                                                                 int occlusion, int peak_block_size,
                                                                 int ssim_block_size,
                                                                 std::string name){
    std::cout<<"Time stats for " + name<<std::endl;
    cv::Mat SSD;
    auto start = std::chrono::high_resolution_clock::now();
    SSD = matchStereoSSDNaive(img1, img2, patch_size, disparity);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Time of Stereo Naive for " << std::to_string(patch_size) << " patch size and " << std::to_string(disparity) << " disparity" << std::endl;
    std::cout << duration.count() << " seconds" << std::endl;
    std::cout << "Peak Signal To Noise Ratio: " << std::to_string(peakSignalToNoiseRatio(SSD, ground, peak_block_size, 255)) << std::endl;
    std::cout << "Structural Similarity Index Measure: " << std::to_string(compareImagesSSIM(SSD, ground, ssim_block_size)) << std::endl;
//    cv::imshow("Stereo Naive", SSD);
//    cv::waitKey();

    cv::Mat SSD_2;
    std::cout<<"Time stats for (+10) " + name<<std::endl;
    start = std::chrono::high_resolution_clock::now();
    SSD_2 = matchStereoSSDNaive(img1, img2, patch_size+10, disparity+10);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Time of Stereo Naive for " << std::to_string(patch_size+10) << " patch size and " << std::to_string(disparity+10) << " disparity" << std::endl;
    std::cout << duration.count() << " seconds" << std::endl;
    std::cout << "Peak Signal To Noise Ratio: " << std::to_string(peakSignalToNoiseRatio(SSD_2, ground, peak_block_size, 255)) << std::endl;
    std::cout << "Structural Similarity Index Measure: " << std::to_string(compareImagesSSIM(SSD_2, ground, ssim_block_size)) << std::endl;
//    cv::imshow("Stereo Naive 2", SSD_2);
//    cv::waitKey();

    cv::Mat stereoBM;
    std::cout<<"Time stats for OpenCV StereoBM " + name<<std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto [disparityImageLeftBM, disparityImageRightBM] = openCvMethodBM(img1, img2, 16, patch_size+3);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << duration.count() << " seconds" << std::endl;
    std::cout << "Peak Signal To Noise Ratio: " << std::to_string(peakSignalToNoiseRatio(disparityImageLeftBM, ground, peak_block_size, 255)) << std::endl;
//    cv::imshow("OpenCV Stereo BM", disparityImageLeftBM);
//    cv::waitKey();

    start = std::chrono::high_resolution_clock::now();
    auto [disparityImageLeft, disparityImageRight] = dynamicStereoMatching(std::move(img1), std::move(img2), occlusion);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Time of Dynamic Programming for " << std::to_string(occlusion) << " occlusion" << std::endl;
    std::cout << duration.count() << " seconds" << std::endl;
    std::cout << "Peak Signal To Noise Ratio: " << std::to_string(peakSignalToNoiseRatio(disparityImageLeft, ground, peak_block_size, 255)) << std::endl;
    std::cout << "Structural Similarity Index Measure: " << std::to_string(compareImagesSSIM(disparityImageLeft, ground, ssim_block_size)) << std::endl;
//    cv::imshow("Disparity Image Left", disparityImageLeft);
//    cv::waitKey();
//    cv::imshow("Disparity Image Right", disparityImageRight);
//    cv::waitKey();

    cv::imwrite(path_to_save + "SSD.png", SSD);
    cv::imwrite(path_to_save + "SSD_2.png", SSD_2);
    cv::imwrite(path_to_save + "dynamic_programing_left.png", disparityImageLeft);
    cv::imwrite(path_to_save + "dynamic_programing_right.png", disparityImageRight);
    cv::imwrite(path_to_save + "stereoBM_left.png", disparityImageLeftBM);
    cv::imwrite(path_to_save + "stereoBM_right.png", disparityImageRightBM);

    return {SSD, SSD_2, disparityImageLeft, disparityImageRight};
}


std::list<std::array<double, 6>> convertToPointcloud(cv::Mat disparityImage, cv::Mat rgbImage,
                                                     double camera_focal_length, double camera_baseline,
                                                     double scaling_factor, double fovy){
    std::list<std::array<double, 6>> pointcloud;
    auto img_center_w = rgbImage.cols;
    auto img_center_h = rgbImage.rows;
    auto ascept = rgbImage.rows / rgbImage.cols;
    auto fovx = 2 * std::atan(img_center_w * 0.5 / (img_center_h * 0.5 / std::tan(fovy * 3.1415 / 360 / 2))) / 3.1415 * 360;
    auto fx = img_center_w / 2 / (std::tan(fovx * 3.1415 / 360 ));
    auto fy = img_center_h / 2 / (std::tan(fovy * 3.1415 / 360 ));

    for(int i = 0; i  < rgbImage.rows; i++){
        for(int j = 0; j < rgbImage.cols; j++){
            if(disparityImage.at<uchar>(i,j) > 0){
                double Z = disparityImage.at<uchar>(i,j) / scaling_factor;
                double X = (j - img_center_w / 2) * Z / fx;
                double Y = (i - img_center_h / 2) * Z / fy;
                std::array<double, 6> point = {
                        X, Y, Z,
                        (double)rgbImage.at<cv::Vec3b>(i, j)[0],
                        (double)rgbImage.at<cv::Vec3b>(i, j)[1],
                        (double)rgbImage.at<cv::Vec3b>(i, j)[2]};
                pointcloud.push_back(point);
            }
        }
    }
    return pointcloud;
}

void savePointcloudToFile(const std::list<std::array<double, 6>>& pointcloud, const std::string& file_name){
    std::ofstream file(file_name);
    int countVertex = static_cast<int>(pointcloud.size());
    std::string header =
            "ply\n"
            "format ascii 1.0\n"
            "element vertex " + std::to_string(countVertex-1) + "\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uint8 red\n"
            "property uint8 green\n"
            "property uint8 blue\n"
            "end_header";
    file << header << std::endl;
    if (file.is_open()){
        for(auto const &i: pointcloud){
            file << std::setprecision(5)
                 << (float)i[0] << " "
                 << (float)i[1] << " "
                 << (float)i[2] << " "
                 << (int)i[3] << " "
                 << (int)i[4] << " "
                 << (int)i[5] << std::endl;
        }
    }
}

void loadAnalyseAndSave(const std::string& path_to_img1,
                        const std::string& path_to_img2,
                        const std::string& path_to_disp,
                        const std::string& path_to_save,
                        int patch_size,
                        int disparity,
                        int occlusion,
                        int peak_block_size,
                        int ssim_block_size,
                        int camera_focal_length,
                        int camera_baseline,
                        int scaling_factor,
                        int fovy,
                        std::string info_name){
    cv::Mat img1 = cv::imread(path_to_img1);
    cv::Mat img2 = cv::imread(path_to_img2);
    cv::Mat ground = cv::imread(path_to_disp);
    cv::Mat img1_gray(img1.rows, img1.cols, CV_8U);
    cv::Mat img2_gray(img1.rows, img1.cols, CV_8U);
    cv::Mat ground_gray(img1.rows, img1.cols, CV_8U);
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    // Get disparity images
    auto [SSD, SSD_2, disparityImageLeft, disparityImageRight] = analyseStereoPair(img1_gray,img2_gray,ground_gray,
                                                                                   path_to_save, patch_size,
                                                                                   disparity, occlusion, peak_block_size,
                                                                                   ssim_block_size, std::move(info_name));

    std::list<std::array<double, 6>> toPointcloudSSD = convertToPointcloud(SSD, img1, camera_focal_length, camera_baseline, scaling_factor, fovy);
    std::list<std::array<double, 6>> toPointcloudSSD_2 = convertToPointcloud(SSD_2, img1, camera_focal_length, camera_baseline, scaling_factor, fovy);
    std::list<std::array<double, 6>> toPointcloudDP = convertToPointcloud(disparityImageLeft, img1, camera_focal_length, camera_baseline, scaling_factor, fovy);
    savePointcloudToFile(toPointcloudSSD, path_to_save + "SSD.ply");
    savePointcloudToFile(toPointcloudSSD_2, path_to_save + "SSD_2.ply");
    savePointcloudToFile(toPointcloudDP, path_to_save + "DP.ply");
}

int main() {
    auto total = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    loadAnalyseAndSave("data/set_1/left.png", "data/set_1/right.png", "data/set_1/disp.png", "data/set_1/",
                       2, 4, 18, 2, 2, 3740, 160, 5, 60, "Art - Set 1");
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Art Set 1 " << duration.count() << " seconds" << std::endl << std::endl;

    start = std::chrono::high_resolution_clock::now();
    loadAnalyseAndSave("data/set_2/left.png", "data/set_2/right.png", "data/set_2/disp.png", "data/set_2/",
                       2, 4, 18, 2, 2, 3740, 160, 5, 60, "Masks - Set 2");
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Masks Set 2 " << duration.count() << " seconds" << std::endl << std::endl;

    start = std::chrono::high_resolution_clock::now();
    loadAnalyseAndSave("data/set_3/left.png", "data/set_3/right.png", "data/set_1/set_3.png", "data/set_3/",
                       2, 4, 18, 2, 2, 3740, 160, 5, 60, "Toys - Set 3");
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Toys Set 3 " << duration.count() << " seconds" << std::endl << std::endl;

    start = std::chrono::high_resolution_clock::now();
    loadAnalyseAndSave("data/set_4/left.png", "data/set_4/right.png", "data/set_4/disp.png", "data/set_4/",
                       2, 4, 18, 2, 2, 3740, 160, 5, 60, "Plant - Set 4");
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Plant Set 4: " << duration.count() << " seconds" << std::endl << std::endl;

    start = std::chrono::high_resolution_clock::now();
    loadAnalyseAndSave("data/set_5/left.png", "data/set_5/right.png", "data/set_5/disp.png", "data/set_5/",
                       2, 4, 18, 2, 2, 3740, 160, 5, 60, "Bowl - Set 5");
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Bowl Set 5: " << duration.count() << " seconds" << std::endl << std::endl;

    start = std::chrono::high_resolution_clock::now();
    loadAnalyseAndSave("data/set_6/left.png", "data/set_6/right.png", "data/set_6/disp.png", "data/set_6/",
                       2, 4, 18, 2, 2, 3740, 160, 5, 60, "Pots - Set 6");
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
    std::cout << "Pots Set 6: " << duration.count() << " seconds" << std::endl << std::endl;

    auto duration_total = std::chrono::duration_cast<std::chrono::seconds>(stop-total);
    std::cout << "Total time: " << duration_total.count() << " seconds" << std::endl;
}




// blockCovariance and blockSigma functions were used from these github repo
// https://gist.github.com/Bibimaw/8873663
double blockSigma(cv::Mat & m, int i, int j, int block_size) {
    double sd = 0;

    cv::Mat m_tmp = m(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m_squared(block_size, block_size, CV_8U);

    cv::multiply(m_tmp, m_tmp, m_squared);

    double avg = mean(m_tmp)[0];
    double avg_2 = mean(m_squared)[0];


    sd = sqrt(avg_2 - avg * avg);

    return sd;
}

double blockCovariance(cv::Mat & m1, cv::Mat & m2, int i, int j, int block_size) {
    cv::Mat m3 = cv::Mat::zeros(block_size, block_size, m1.depth());
    cv::Mat m1_tmp = m1(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m2_tmp = m2(cv::Range(i, i + block_size), cv::Range(j, j + block_size));

    multiply(m1_tmp, m2_tmp, m3);
    double avg_ro 	= mean(m3)[0];
    double avg_r 	= mean(m1_tmp)[0];
    double avg_o 	= mean(m2_tmp)[0];

    double sd_ro = avg_ro - avg_o * avg_r;

    return sd_ro;
}
