#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
	// Check for the arguments
	if(argc < 2) {
		std::cout
			<< "Usage: ./dence_tracking /path/to/dataset"
			<< std::endl;

		return 0;
	}

	auto dataset_path = std::string(argv[1]) + "%d.png";

	cv::Mat image0 = cv::imread(cv::format(dataset_path.c_str(), 0));
	cv::Mat image1 = cv::imread(cv::format(dataset_path.c_str(), 1));

	return 0;
}
