#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

const std::string path = "C:/Users/Brouse/Code/IdeaProjects/IA/py";

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    std::cout << "Cuda: " << torch::cuda::is_available() << std::endl;

    std::cout << "Loading model...\n";

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(path + "/resnet18_cifar10_scripted.pt", device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Model loaded\n";

    int total_predictions = 0;
    int correct_predictions = 0;

    for (const auto& entry : fs::directory_iterator(path + "/preprocessed")) {
        auto* matrix = new int[10 * 10];

        std::cout << "Reading " << entry.path().string() << "\n";

        std::ifstream input(entry.path(), std::ios::binary | std::ios::ate);
        if (!input) {
            std::cerr << "Error opening file: " << entry.path() << "\n";
            continue;
        }

        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!input.read(buffer.data(), size)) {
            std::cerr << "Error reading file: " << entry.path() << "\n";
            continue;
        }

        torch::IValue ivalue;
        try {
            ivalue = torch::pickle_load(buffer);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading pickle data: " << e.what() << "\n";
            continue;
        }

        auto dict = ivalue.toGenericDict();
        torch::Tensor images = dict.at("images").toTensor().to(device);
        torch::Tensor labels = dict.at("labels").toTensor().to(device);

        // Make sure dimensions match
        if (images.size(0) != labels.size(0)) {
            std::cerr << "Mismatch in batch size for images and labels.\n";
            continue;
        }

        int correct = 0;
        int total = 0;
        torch::NoGradGuard no_grad;

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < images.size(0); ++i) {
            torch::Tensor single_image = images[i].unsqueeze(0); // Add batch dimension
            torch::Tensor single_label = labels[i];

            torch::Tensor output = model.forward({single_image}).toTensor();
            int prediction = output.argmax(1).item<int>();
            int label = single_label.item<int>();

            bool correct_prediction = (prediction == label);
            correct += correct_prediction ? 1 : 0;
            total += 1;

            matrix[10 * label + prediction]++;
            //std::cout << "Image " << i << ": predicted = " << prediction << ", actual = " << label
            //          << (correct_prediction ? " [OK]" : " [WRONG]") << "\n";
        }

        std::cout << "\nConfusion Matrix (rows = actual, columns = predicted):\n\n";

        // Print header
        std::cout << "     ";
        for (int col = 0; col < 10; ++col)
            std::cout << std::setw(4) << col;
        std::cout << "\n";

        // Print separator
        std::cout << "     " << std::string(4 * 10, '-') << "\n";

        // Print each row
        for (int row = 0; row < 10; ++row) {
            std::cout << std::setw(2) << row << " |";
            for (int col = 0; col < 10; ++col) {
                std::cout << std::setw(4) << matrix[10 * row + col];
            }
            std::cout << "\n";
        }

        auto end = std::chrono::system_clock::now();
        std::cout << "Batch accuracy: " << static_cast<float>(correct) / total * 100.0f << "%\n";

        std::chrono::duration<float,std::milli> duration = end - start;
        std::cout << duration.count() << "ms\n";

        correct_predictions += correct;
        total_predictions += total;
    }

    std::cout << "==== FINAL ACCURACY ====\n";
    std::cout << "Correct: " << correct_predictions << " / Total: " << total_predictions << "\n";
    std::cout << "Accuracy: " << static_cast<float>(correct_predictions) / total_predictions * 100.0f << "%\n";

    return 0;
}
