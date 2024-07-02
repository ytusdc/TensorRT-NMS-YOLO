#pragma once
#include <string>
#include "yolo.hpp"

int main(int argc, char** argv) {
    std::string engine_file_path = "";
    if (argc == 4 && std::string(argv[2]) == "-i") 
    {
        engine_file_path = argv[1];
    } 
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolo ../model_trt.engine -i ../*.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }

    auto start = std::chrono::system_clock::now();
    const std::string input_image_path {argv[3]};
    YOLO yolo(engine_file_path);
    yolo.detect_img(input_image_path);

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}