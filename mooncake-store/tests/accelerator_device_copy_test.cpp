// Copies between CUDA device memory and host memory through the store's
// accelerator device. ctest runs this binary twice: with MC_GPU_COPY unset
// (the synchronous cudaMemcpy on the legacy default stream) and with
// MC_GPU_COPY=pinned (a non-blocking stream with pinned staging of pageable
// host memory). Each test skips without a CUDA device.

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "device/accelerator_registry.h"

namespace mooncake {
namespace {

using device::CopyDirection;

bool PinnedModeRequested() {
    const char* env = std::getenv("MC_GPU_COPY");
    return env != nullptr && std::strcmp(env, "pinned") == 0;
}

class AcceleratorDeviceCopyTest : public ::testing::Test {
   protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            cudaGetLastError();
            GTEST_SKIP() << "no CUDA device";
        }
        device_ = device::GetAcceleratorRegistry().GetDevice(
            device::AcceleratorVendor::kNvidia);
        ASSERT_NE(device_, nullptr);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    static std::vector<char> Pattern(size_t size, int seed) {
        std::vector<char> bytes(size);
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = static_cast<char>((i * 131 + seed * 7) & 0xff);
        }
        return bytes;
    }

    const device::AcceleratorDevice* device_ = nullptr;
};

TEST_F(AcceleratorDeviceCopyTest, CopiesMoveEveryByte) {
    // Sizes below, at and across the 4 MiB pinned staging chunk.
    for (size_t size : {size_t{1}, size_t{4 << 20}, size_t{(9 << 20) + 7}}) {
        void* gpu = nullptr;
        ASSERT_EQ(cudaMalloc(&gpu, size), cudaSuccess);
        const std::vector<char> pageable_src = Pattern(size, 1);
        std::vector<char> pageable_dst(size, 0);
        ASSERT_TRUE(device_->Copy(gpu, pageable_src.data(), size,
                                  CopyDirection::kHostToDevice));
        ASSERT_TRUE(device_->Copy(pageable_dst.data(), gpu, size,
                                  CopyDirection::kDeviceToHost));
        EXPECT_EQ(pageable_dst, pageable_src) << size << " bytes, pageable";

        char* pinned = nullptr;
        ASSERT_EQ(cudaMallocHost(&pinned, size), cudaSuccess);
        const std::vector<char> pattern = Pattern(size, 2);
        std::memcpy(pinned, pattern.data(), size);
        ASSERT_TRUE(
            device_->Copy(gpu, pinned, size, CopyDirection::kHostToDevice));
        std::memset(pinned, 0, size);
        ASSERT_TRUE(
            device_->Copy(pinned, gpu, size, CopyDirection::kDeviceToHost));
        EXPECT_EQ(std::memcmp(pinned, pattern.data(), size), 0)
            << size << " bytes, pinned";
        cudaFreeHost(pinned);
        cudaFree(gpu);
    }
}

void CUDART_CB SleepOnStream(void* seconds) {
    std::this_thread::sleep_for(
        std::chrono::duration<double>(*static_cast<double*>(seconds)));
}

// The synchronous cudaMemcpy waits for work queued on the legacy default
// stream; the pinned mode's copy does not.
TEST_F(AcceleratorDeviceCopyTest, OnlyThePinnedModeIgnoresTheLegacyStream) {
    const size_t size = 8 << 20;
    void* gpu = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu, size), cudaSuccess);
    std::vector<char> host = Pattern(size, 3);
    // Allocate the staging chunk and the copy stream before the timing.
    ASSERT_TRUE(
        device_->Copy(gpu, host.data(), size, CopyDirection::kHostToDevice));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    static double kBlockSeconds = 1.5;
    ASSERT_EQ(
        cudaLaunchHostFunc(cudaStreamLegacy, SleepOnStream, &kBlockSeconds),
        cudaSuccess);
    const auto start = std::chrono::steady_clock::now();
    ASSERT_TRUE(
        device_->Copy(host.data(), gpu, size, CopyDirection::kDeviceToHost));
    const double copy_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(host, Pattern(size, 3));
    if (PinnedModeRequested()) {
        EXPECT_LT(copy_seconds, kBlockSeconds / 2);
    } else {
        EXPECT_GE(copy_seconds, kBlockSeconds * 0.9);
    }
    cudaFree(gpu);
}

}  // namespace
}  // namespace mooncake

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
