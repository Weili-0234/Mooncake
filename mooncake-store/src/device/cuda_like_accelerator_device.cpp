#include "device/accelerator_registry.h"
#include "pinned_host_buffer.h"

#include "cuda_alike.h"

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MACA) || \
    defined(USE_HYGON) || defined(USE_COREX)

namespace mooncake {
namespace device {

void EnsureCudaLikeAcceleratorDeviceLinked() {}

namespace {

void FreeCudaLikePinnedHostBuffer(void* addr) { cudaFreeHost(addr); }

#if defined(USE_CUDA) && !defined(USE_MACA)
// MC_GPU_COPY=pinned: a copy between a GPU buffer and host memory runs as
// cudaMemcpyAsync on a non-blocking stream of the calling thread, and a
// pageable host side is staged through pooled pinned chunks. The default
// synchronous cudaMemcpy runs on the legacy default stream, so it waits for,
// and delays, every CUDA call the application makes on that stream, and a
// copy from pageable memory runs at a few GB/s. Under this mode the copy
// follows no stream order of the application: callers fence a buffer's data
// before they put it, as they must for RDMA transfers.
bool StoreGpuCopyPinned() {
    static const bool pinned = [] {
        const char* env = std::getenv("MC_GPU_COPY");
        const bool enabled = env != nullptr && std::strcmp(env, "pinned") == 0;
        if (enabled) {
            LOG(INFO) << "Store GPU copy: non-blocking stream, pinned staging "
                         "of pageable host memory (MC_GPU_COPY=pinned)";
        } else if (env != nullptr && *env != '\0' &&
                   std::strcmp(env, "legacy") != 0) {
            LOG(WARNING) << "Ignoring MC_GPU_COPY=" << env
                         << "; expected pinned or legacy";
        }
        return enabled;
    }();
    return pinned;
}

constexpr size_t kStagingChunkBytes = 4ull << 20;

// Pinned chunks of kStagingChunkBytes. A copy holds one for its duration and
// returns it; chunks are never freed (cudaFreeHost can synchronize the
// device), so the pool grows to the peak number of concurrent staged copies.
class StagingChunkPool {
   public:
    static StagingChunkPool& instance() {
        static StagingChunkPool* pool = new StagingChunkPool();
        return *pool;
    }

    char* acquire() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!free_.empty()) {
                char* buf = free_.back();
                free_.pop_back();
                return buf;
            }
        }
        void* buf = nullptr;
        if (cudaHostAlloc(&buf, kStagingChunkBytes, cudaHostAllocPortable) !=
            cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        LOG(INFO) << "Store copy pinned chunk pool: cudaHostAlloc #"
                  << allocated_.fetch_add(1) + 1 << " (" << kStagingChunkBytes
                  << " bytes)";
        return static_cast<char*>(buf);
    }

    void release(char* buf) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_.push_back(buf);
    }

   private:
    std::mutex mutex_;
    std::vector<char*> free_;
    std::atomic<uint64_t> allocated_{0};
};

cudaStream_t ThreadCopyStream() {
    thread_local std::unordered_map<int, cudaStream_t> streams;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    auto it = streams.find(device);
    if (it == streams.end()) {
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        it = streams.emplace(device, stream).first;
    }
    return it->second;
}

bool IsPinnedHost(const void* ptr) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeHost;
}

bool CopyOnStream(void* dst, const void* src, size_t size, cudaMemcpyKind kind,
                  cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, size, kind, stream) == cudaSuccess &&
           cudaStreamSynchronize(stream) == cudaSuccess;
}

bool PinnedNonBlockingCopy(void* dst, const void* src, size_t size,
                           cudaMemcpyKind kind) {
    cudaStream_t stream = ThreadCopyStream();
    if (!stream) return false;
    const bool d2h = kind == cudaMemcpyDeviceToHost;
    const bool h2d = kind == cudaMemcpyHostToDevice;
    if ((!d2h && !h2d) || IsPinnedHost(d2h ? dst : src)) {
        return CopyOnStream(dst, src, size, kind, stream);
    }
    char* chunk = StagingChunkPool::instance().acquire();
    if (!chunk) return false;
    bool ok = true;
    for (size_t off = 0; ok && off < size; off += kStagingChunkBytes) {
        const size_t n = std::min(kStagingChunkBytes, size - off);
        if (d2h) {
            ok = CopyOnStream(chunk, static_cast<const char*>(src) + off, n,
                              kind, stream);
            if (ok) std::memcpy(static_cast<char*>(dst) + off, chunk, n);
        } else {
            std::memcpy(chunk, static_cast<const char*>(src) + off, n);
            ok = CopyOnStream(static_cast<char*>(dst) + off, chunk, n, kind,
                              stream);
        }
    }
    StagingChunkPool::instance().release(chunk);
    return ok;
}
#endif

class CudaLikeAcceleratorDevice final : public ProbeCachedAcceleratorDevice {
   public:
    explicit CudaLikeAcceleratorDevice(AcceleratorVendor vendor)
        : vendor_(vendor) {}

    AcceleratorVendor Vendor() const override { return vendor_; }

    bool ProbeAvailable() const override {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    PointerInfo QueryPointer(const void* ptr) const override {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, ptr) == cudaSuccess &&
            attr.type == cudaMemoryTypeDevice) {
            return PointerInfo{.kind = MemoryKind::kDevice,
                               .device_id = attr.device};
        }
        cudaGetLastError();
        return PointerInfo{.kind = MemoryKind::kHost, .device_id = -1};
    }

    int32_t CurrentDeviceId() const override {
        int device_id = -1;
        return cudaGetDevice(&device_id) == cudaSuccess ? device_id : -1;
    }

    void SetContext(int32_t device_id) const override {
        if (device_id >= 0) cudaSetDevice(device_id);
    }

    bool Copy(void* dst, const void* src, size_t size,
              CopyDirection direction) const override {
        cudaMemcpyKind kind = cudaMemcpyDefault;
        switch (direction) {
            case CopyDirection::kHostToDevice:
                kind = cudaMemcpyHostToDevice;
                break;
            case CopyDirection::kDeviceToHost:
                kind = cudaMemcpyDeviceToHost;
                break;
            case CopyDirection::kDeviceToDevice:
                kind = cudaMemcpyDeviceToDevice;
                break;
            case CopyDirection::kHostToHost:
            case CopyDirection::kAuto:
                kind = cudaMemcpyDefault;
                break;
        }
#if defined(USE_CUDA) && !defined(USE_MACA)
        if (kind != cudaMemcpyDefault && StoreGpuCopyPinned()) {
            return PinnedNonBlockingCopy(dst, src, size, kind);
        }
#endif
        return cudaMemcpy(dst, src, size, kind) == cudaSuccess;
    }

    PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
        void* addr = nullptr;
        if (cudaMallocHost(&addr, size) != cudaSuccess) {
            cudaGetLastError();
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(addr, size, FreeCudaLikePinnedHostBuffer);
    }

   private:
    AcceleratorVendor vendor_;
};

#define REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(name, vendor) \
    const CudaLikeAcceleratorDevice name##_device(vendor);  \
    const AcceleratorDeviceRegistrar name##_registrar(name##_device)

#if defined(USE_CUDA)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(nvidia, AcceleratorVendor::kNvidia);
#endif

#if defined(USE_MUSA)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(musa, AcceleratorVendor::kMusa);
#endif

#if defined(USE_MACA)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(maca, AcceleratorVendor::kMaca);
#endif

#if defined(USE_HYGON)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(hygon, AcceleratorVendor::kHygon);
#endif

#if defined(USE_COREX)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(corex, AcceleratorVendor::kCorex);
#endif

#undef REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE

}  // namespace
}  // namespace device
}  // namespace mooncake

#endif
