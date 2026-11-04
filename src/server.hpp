#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include "protocol.hpp"

namespace rl {

// One-slot mailbox between the socket thread and the game thread.
// Singleton is heap-allocated and never destroyed (the thread is detached).
class Server {
public:
    static Server& get();

    void ensureStarted(uint16_t port);                       // game thread, idempotent
    bool popRequest(Request& out, std::chrono::microseconds timeout); // game thread
    void respond(uint32_t seq, const uint8_t* bytes);        // game thread, kRespLen bytes
    bool connected() const { return m_connected.load(); }

private:
    void threadMain(uint16_t port);
    void serve(uintptr_t sock);

    std::atomic<bool> m_started{false};
    std::atomic<bool> m_connected{false};

    std::mutex m_mx;
    std::condition_variable m_cvReq, m_cvResp;
    Request m_pending{};
    bool m_hasPending = false;
    bool m_waiting = false;
    bool m_respReady = false;
    uint32_t m_seq = 0, m_waitSeq = 0;
    uint8_t m_resp[kRespLen];
};

} // namespace rl
