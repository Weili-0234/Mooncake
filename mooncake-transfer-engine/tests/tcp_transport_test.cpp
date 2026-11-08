// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cuda_alike.h"
#if defined(USE_CUDA) && defined(USE_NVMEOF)
#include <cufile.h>
#endif

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) || \
    defined(USE_MACA) || defined(USE_HYGON) || defined(USE_COREX)
#include <cassert>
#include "common/base/status.h"

static void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#include "transfer_engine.h"
#include "transport/transport.h"

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) || \
    defined(USE_MACA) || defined(USE_HYGON) || defined(USE_COREX)
DEFINE_int32(gpu_id, 0, "GPU ID to use");
#endif

using namespace mooncake;

namespace mooncake {

class TCPTransportTest : public ::testing::Test {
   public:
   protected:
    void SetUp() override {
        google::InitGoogleLogging("TCPTransportTest");
        FLAGS_logtostderr = 1;

        const char *env = std::getenv("MC_METADATA_SERVER");
        if (env)
            metadata_server = env;
        else
            metadata_server = metadata_server;
        LOG(INFO) << "metadata_server: " << metadata_server;

        env = std::getenv("MC_LOCAL_SERVER_NAME");
        if (env)
            local_server_name = env;
        else
            local_server_name = "127.0.0.2:12345";
        LOG(INFO) << "local_server_name: " << local_server_name;
    }

    void TearDown() override {
        // 清理 glog
        google::ShutdownGoogleLogging();
    }

    std::string metadata_server;
    std::string local_server_name;
};

static void *allocateMemoryPool(size_t size, int socket_id,
                                bool from_vram = false) {
    return numa_alloc_onnode(size, socket_id);
}

TEST_F(TCPTransportTest, GetTcpTest) {
    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(local_server_name);
    auto rc = engine->init(metadata_server, local_server_name,
                           hostname_port.first.c_str(), hostname_port.second);
    LOG_ASSERT(rc == 0);
    Transport *xport = nullptr;
    xport = engine->installTransport("tcp", nullptr);
    LOG_ASSERT(xport != nullptr);
}

TEST_F(TCPTransportTest, Writetest) {
    const size_t kDataLength = 4096000;
    void *addr = nullptr;
    const size_t ram_buffer_size = 1ull << 30;
    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(local_server_name);
    auto rc = engine->init(metadata_server, local_server_name,
                           hostname_port.first.c_str(), hostname_port.second);
    LOG_ASSERT(rc == 0);
    Transport *xport = nullptr;
    xport = engine->installTransport("tcp", nullptr);
    LOG_ASSERT(xport != nullptr);

    addr = allocateMemoryPool(ram_buffer_size, 0, false);
    rc = engine->registerLocalMemory(addr, ram_buffer_size, "cpu:0");
    LOG_ASSERT(!rc);

    for (size_t offset = 0; offset < kDataLength; ++offset)
        *((char *)(addr) + offset) = 'a' + lrand48() % 26;
    auto batch_id = engine->allocateBatchID(1);
    Status s;
    auto segment_id = engine->openSegment(local_server_name);
    TransferRequest entry;
    auto segment_desc = engine->getMetadata()->getSegmentDescByID(segment_id);
    uint64_t remote_base = (uint64_t)segment_desc->buffers[0].addr;
    entry.opcode = TransferRequest::WRITE;
    entry.length = kDataLength;
    entry.source = (uint8_t *)(addr);
    entry.target_id = segment_id;
    entry.target_offset = remote_base;
    s = engine->submitTransfer(batch_id, {entry});
    LOG_ASSERT(s.ok());
    bool completed = false;
    TransferStatus status;
    while (!completed) {
        Status s = engine->getTransferStatus(batch_id, 0, status);
        ASSERT_EQ(s, Status::OK());
        LOG_ASSERT(status.s != TransferStatusEnum::FAILED);
        if (status.s == TransferStatusEnum::COMPLETED) completed = true;
    }
    s = engine->freeBatchID(batch_id);
    ASSERT_EQ(s, Status::OK());
}

TEST_F(TCPTransportTest, WriteAndReadtest) {
    const size_t kDataLength = 4096000;
    void *addr = nullptr;
    const size_t ram_buffer_size = 1ull << 30;
    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(local_server_name);
    engine->init(metadata_server, local_server_name,
                 hostname_port.first.c_str(), hostname_port.second);
    Transport *xport = nullptr;
    xport = engine->installTransport("tcp", nullptr);
    LOG_ASSERT(xport != nullptr);

    addr = allocateMemoryPool(ram_buffer_size, 0, false);
    int rc = engine->registerLocalMemory(addr, ram_buffer_size, "cpu:0");
    LOG_ASSERT(!rc);
    for (size_t offset = 0; offset < kDataLength; ++offset)
        *((char *)(addr) + offset) = 'a' + lrand48() % 26;

    auto segment_id = engine->openSegment(local_server_name);
    auto segment_desc = engine->getMetadata()->getSegmentDescByID(segment_id);
    uint64_t remote_base = (uint64_t)segment_desc->buffers[0].addr;
    {
        auto batch_id = engine->allocateBatchID(1);
        Status s;
        TransferRequest entry;
        entry.opcode = TransferRequest::WRITE;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr);
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            ASSERT_EQ(s, Status::OK());
            LOG_ASSERT(status.s != TransferStatusEnum::FAILED);
            if (status.s == TransferStatusEnum::COMPLETED) completed = true;
        }
        s = engine->freeBatchID(batch_id);
        ASSERT_EQ(s, Status::OK());
    }

    {
        auto batch_id = engine->allocateBatchID(1);
        Status s;

        TransferRequest entry;
        entry.opcode = TransferRequest::READ;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr) + kDataLength;
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            LOG_ASSERT(s.ok());
            if (status.s == TransferStatusEnum::COMPLETED) completed = true;
            LOG_ASSERT(status.s != TransferStatusEnum::FAILED);
        }
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok());
    }
    LOG_ASSERT(0 == memcmp((uint8_t *)(addr), (uint8_t *)(addr) + kDataLength,
                           kDataLength));
}

TEST_F(TCPTransportTest, WriteAndRead2test) {
    const size_t kDataLength = 4096000;
    void *addr = nullptr;
    const size_t ram_buffer_size = 1ull << 30;
    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(local_server_name);
    engine->init(metadata_server, local_server_name,
                 hostname_port.first.c_str(), hostname_port.second);
    Transport *xport = nullptr;
    xport = engine->installTransport("tcp", nullptr);
    LOG_ASSERT(xport != nullptr);

    addr = allocateMemoryPool(ram_buffer_size, 0, false);
    int rc = engine->registerLocalMemory(addr, ram_buffer_size, "cpu:0");
    LOG_ASSERT(!rc);
    for (size_t offset = 0; offset < kDataLength; ++offset)
        *((char *)(addr) + offset) = 'a' + lrand48() % 26;

    auto segment_id = engine->openSegment(local_server_name);
    auto segment_desc = engine->getMetadata()->getSegmentDescByID(segment_id);
    uint64_t remote_base = (uint64_t)segment_desc->buffers[0].addr;

    {
        auto batch_id = engine->allocateBatchID(1);
        Status s;
        TransferRequest entry;
        entry.opcode = TransferRequest::WRITE;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr);
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            ASSERT_EQ(s, Status::OK());
            LOG_ASSERT(status.s != TransferStatusEnum::FAILED);
            if (status.s == TransferStatusEnum::COMPLETED) completed = true;
        }
        s = engine->freeBatchID(batch_id);
        ASSERT_EQ(s, Status::OK());
    }

    {
        auto batch_id = engine->allocateBatchID(1);
        Status s;
        TransferRequest entry;
        entry.opcode = TransferRequest::READ;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr) + kDataLength;
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            LOG_ASSERT(s.ok());
            if (status.s == TransferStatusEnum::COMPLETED) completed = true;
            LOG_ASSERT(status.s != TransferStatusEnum::FAILED);
        }
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok());
    }
    LOG_ASSERT(0 == memcmp((uint8_t *)(addr), (uint8_t *)(addr) + kDataLength,
                           kDataLength));

    for (size_t offset = 0; offset < kDataLength; ++offset)
        *((char *)(addr) + offset) = 'a' + lrand48() % 26;
    {
        auto batch_id = engine->allocateBatchID(1);
        Status s;
        TransferRequest entry;
        entry.opcode = TransferRequest::WRITE;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr);
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            ASSERT_EQ(s, Status::OK());
            LOG_ASSERT(status.s != TransferStatusEnum::FAILED);
            if (status.s == TransferStatusEnum::COMPLETED) completed = true;
        }
        s = engine->freeBatchID(batch_id);
        ASSERT_EQ(s, Status::OK());
    }

    {
        auto batch_id = engine->allocateBatchID(1);
        Status s;
        TransferRequest entry;
        entry.opcode = TransferRequest::READ;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr) + kDataLength;
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            LOG_ASSERT(s.ok());
            if (status.s == TransferStatusEnum::COMPLETED) completed = true;
            LOG_ASSERT(status.s != TransferStatusEnum::FAILED);
        }
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok());
    }
    LOG_ASSERT(0 == memcmp((uint8_t *)(addr), (uint8_t *)(addr) + kDataLength,
                           kDataLength));
}

#if defined(USE_CUDA) && !defined(USE_MACA)
// Transfers whose local buffer or target buffer is CUDA device memory, which
// the TCP transport stages through host buffers. ctest runs them with
// MC_GPU_COPY unset (synchronous cudaMemcpy) and with MC_GPU_COPY=pinned
// (pinned staging on a non-blocking stream).
class TCPTransportGpuTest : public TCPTransportTest {
   protected:
    void SetUp() override {
        TCPTransportTest::SetUp();
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            cudaGetLastError();
            GTEST_SKIP() << "no CUDA device";
        }
        checkCudaError(cudaSetDevice(FLAGS_gpu_id), "cudaSetDevice");
        engine_ = std::make_unique<TransferEngine>(false);
        auto hostname_port = parseHostNameWithPort(local_server_name);
        ASSERT_EQ(
            engine_->init(metadata_server, local_server_name,
                          hostname_port.first.c_str(), hostname_port.second),
            0);
        ASSERT_NE(engine_->installTransport("tcp", nullptr), nullptr);
    }

    // Runs one transfer of `length` bytes between `local` and `target` in
    // this engine's own segment; returns whether it completed.
    bool Transfer(TransferRequest::OpCode opcode, void *local, uint64_t target,
                  size_t length) {
        auto segment_id = engine_->openSegment(local_server_name);
        auto batch_id = engine_->allocateBatchID(1);
        TransferRequest entry;
        entry.opcode = opcode;
        entry.length = length;
        entry.source = static_cast<uint8_t *>(local);
        entry.target_id = segment_id;
        entry.target_offset = target;
        if (!engine_->submitTransfer(batch_id, {entry}).ok()) return false;
        TransferStatus status;
        do {
            if (!engine_->getTransferStatus(batch_id, 0, status).ok() ||
                status.s == TransferStatusEnum::FAILED) {
                return false;
            }
        } while (status.s != TransferStatusEnum::COMPLETED);
        return engine_->freeBatchID(batch_id).ok();
    }

    static std::vector<char> Pattern(size_t size) {
        std::vector<char> bytes(size);
        for (auto &byte : bytes) byte = 'a' + lrand48() % 26;
        return bytes;
    }

    std::unique_ptr<TransferEngine> engine_;
};

TEST_F(TCPTransportGpuTest, GpuBuffersMoveEveryByte) {
    // Several slices, not a multiple of the slice size.
    const size_t kDataLength = 4096000 + 123;
    void *host = numa_alloc_onnode(kDataLength, 0);
    void *gpu_segment = nullptr;
    void *gpu_src = nullptr;
    void *gpu_dst = nullptr;
    checkCudaError(cudaMalloc(&gpu_segment, kDataLength), "cudaMalloc");
    checkCudaError(cudaMalloc(&gpu_src, kDataLength), "cudaMalloc");
    checkCudaError(cudaMalloc(&gpu_dst, kDataLength), "cudaMalloc");
    ASSERT_EQ(engine_->registerLocalMemory(host, kDataLength, "cpu:0"), 0);
    ASSERT_EQ(
        engine_->registerLocalMemory(gpu_segment, kDataLength,
                                     "cuda:" + std::to_string(FLAGS_gpu_id)),
        0);
    const std::vector<char> pattern = Pattern(kDataLength);
    checkCudaError(cudaMemcpy(gpu_src, pattern.data(), kDataLength,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy");

    // The local side is device memory: GPU to a host buffer and back.
    ASSERT_TRUE(Transfer(TransferRequest::WRITE, gpu_src,
                         reinterpret_cast<uint64_t>(host), kDataLength));
    EXPECT_EQ(memcmp(host, pattern.data(), kDataLength), 0);
    ASSERT_TRUE(Transfer(TransferRequest::READ, gpu_dst,
                         reinterpret_cast<uint64_t>(host), kDataLength));
    std::vector<char> back(kDataLength);
    checkCudaError(
        cudaMemcpy(back.data(), gpu_dst, kDataLength, cudaMemcpyDeviceToHost),
        "cudaMemcpy");
    EXPECT_EQ(back, pattern);

    // Both sides are device memory: the target buffer is staged as well.
    checkCudaError(cudaMemset(gpu_dst, 0, kDataLength), "cudaMemset");
    ASSERT_TRUE(Transfer(TransferRequest::WRITE, gpu_src,
                         reinterpret_cast<uint64_t>(gpu_segment), kDataLength));
    ASSERT_TRUE(Transfer(TransferRequest::READ, gpu_dst,
                         reinterpret_cast<uint64_t>(gpu_segment), kDataLength));
    checkCudaError(
        cudaMemcpy(back.data(), gpu_dst, kDataLength, cudaMemcpyDeviceToHost),
        "cudaMemcpy");
    EXPECT_EQ(back, pattern);

    engine_->unregisterLocalMemory(gpu_segment);
    engine_->unregisterLocalMemory(host);
    cudaFree(gpu_dst);
    cudaFree(gpu_src);
    cudaFree(gpu_segment);
    numa_free(host, kDataLength);
}

static void CUDART_CB SleepOnStream(void *seconds) {
    std::this_thread::sleep_for(
        std::chrono::duration<double>(*static_cast<double *>(seconds)));
}

// The synchronous cudaMemcpy waits for work queued on the legacy default
// stream (an application's kernels); the pinned mode's copy does not.
TEST_F(TCPTransportGpuTest, OnlyThePinnedModeIgnoresTheLegacyStream) {
    const size_t kDataLength = 4096000;
    void *host = numa_alloc_onnode(kDataLength, 0);
    void *gpu_src = nullptr;
    checkCudaError(cudaMalloc(&gpu_src, kDataLength), "cudaMalloc");
    ASSERT_EQ(engine_->registerLocalMemory(host, kDataLength, "cpu:0"), 0);
    const std::vector<char> pattern = Pattern(kDataLength);
    checkCudaError(cudaMemcpy(gpu_src, pattern.data(), kDataLength,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy");
    // Fill the staging pool and create the copy streams before the timing.
    ASSERT_TRUE(Transfer(TransferRequest::WRITE, gpu_src,
                         reinterpret_cast<uint64_t>(host), kDataLength));
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    static double kBlockSeconds = 1.5;
    checkCudaError(
        cudaLaunchHostFunc(cudaStreamLegacy, SleepOnStream, &kBlockSeconds),
        "cudaLaunchHostFunc");
    const auto start = std::chrono::steady_clock::now();
    ASSERT_TRUE(Transfer(TransferRequest::WRITE, gpu_src,
                         reinterpret_cast<uint64_t>(host), kDataLength));
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    EXPECT_EQ(memcmp(host, pattern.data(), kDataLength), 0);
    const char *mode = std::getenv("MC_GPU_COPY");
    if (mode != nullptr && std::string(mode) == "pinned") {
        EXPECT_LT(seconds, kBlockSeconds / 2);
    } else {
        EXPECT_GE(seconds, kBlockSeconds * 0.9);
    }

    engine_->unregisterLocalMemory(host);
    cudaFree(gpu_src);
    numa_free(host, kDataLength);
}
#endif

}  // namespace mooncake

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
