#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

static_assert(std::endian::native == std::endian::little, "protocol is little-endian");

namespace rl {

constexpr uint32_t kMagic = 0x4C524447u; // bytes "GDRL"
constexpr uint16_t kProtocolVersion = 1;

constexpr int kNumObj = 16;
constexpr int kObjFeats = 5;
constexpr int kPlayerFeats = 8;
constexpr int kObsDim = kPlayerFeats + kNumObj * kObjFeats; // 88

constexpr size_t kReqLen = 12;
constexpr size_t kHeaderLen = 20;
constexpr size_t kRespLen = kHeaderLen + kObsDim * sizeof(float); // 372

enum Cmd : uint8_t { Hello = 0, Reset = 1, Step = 2, Close = 3, Info = 4 };
enum Flag : uint8_t { FlagDone = 1, FlagSuccess = 2, FlagDead = 4 };
enum Err : uint16_t { ErrNone = 0, ErrVersion = 1, ErrBusy = 2, ErrNoPlayer = 3, ErrBadCmd = 4, ErrNoSnapshots = 5 };

struct Request {
    uint8_t cmd = 0, action = 0, skip = 1;
    float startX = 0.f;
    uint32_t aux = 0;
    uint32_t seq = 0; // server-internal, not on the wire
};

inline void decodeRequest(const uint8_t* b, Request& r) {
    r.cmd = b[0];
    r.action = b[1];
    r.skip = b[2];
    std::memcpy(&r.startX, b + 4, 4);
    std::memcpy(&r.aux, b + 8, 4);
}

inline void encodeResponse(uint8_t* out, uint8_t flags, uint8_t cmd, uint16_t err,
                           uint32_t tick, float dx, const float* obs) {
    uint32_t magic = kMagic;
    uint16_t ver = kProtocolVersion;
    uint16_t dim = kObsDim;
    std::memcpy(out + 0, &magic, 4);
    std::memcpy(out + 4, &ver, 2);
    std::memcpy(out + 6, &dim, 2);
    out[8] = flags;
    out[9] = cmd;
    std::memcpy(out + 10, &err, 2);
    std::memcpy(out + 12, &tick, 4);
    std::memcpy(out + 16, &dx, 4);
    std::memcpy(out + kHeaderLen, obs, kObsDim * sizeof(float));
}

} // namespace rl
