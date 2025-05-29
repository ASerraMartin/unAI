cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cpp-torch-deployment)

# Requiere C++17 (recomendado para LibTorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Si quieres usar Release mode
set(CMAKE_BUILD_TYPE Release)

# Encontrar LibTorch
find_package(Torch REQUIRED)

# Crear ejecutable
add_executable(${PROJECT_NAME} main.cpp)

# Linkear contra las librerías de LibTorch
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Para que pueda encontrar correctamente los headers
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

# Evitar advertencias comunes de símbolo en Windows
if (MSVC)
    add_definitions(-D_SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING)
endif()