// winsock2.h must come before anything that might pull in windows.h
#include <winsock2.h>
#include <ws2tcpip.h>
#include "server.hpp"
#include <Geode/Geode.hpp>
#include <thread>

#pragma comment(lib, "ws2_32.lib")

using namespace geode::prelude;

namespace rl {

static bool recvAll(SOCKET s, uint8_t* buf, int n) {
    int got = 0;
    while (got < n) {
        int r = recv(s, reinterpret_cast<char*>(buf) + got, n - got, 0);
        if (r <= 0) return false;
        got += r;
    }
    return true;
}

static bool sendAll(SOCKET s, const uint8_t* buf, int n) {
    int sent = 0;
    while (sent < n) {
        int r = send(s, reinterpret_cast<const char*>(buf) + sent, n - sent, 0);
        if (r <= 0) return false;
        sent += r;
    }
    return true;
}

Server& Server::get() {
    static Server* s = new Server();
    return *s;
}

void Server::ensureStarted(uint16_t port) {
    bool expected = false;
    if (!m_started.compare_exchange_strong(expected, true)) return;
    std::thread([this, port] { threadMain(port); }).detach();
}

bool Server::popRequest(Request& out, std::chrono::microseconds timeout) {
    std::unique_lock lk(m_mx);
    if (!m_cvReq.wait_for(lk, timeout, [&] { return m_hasPending; })) return false;
    out = m_pending;
    m_hasPending = false;
    return true;
}

void Server::respond(uint32_t seq, const uint8_t* bytes) {
    std::lock_guard lk(m_mx);
    if (!m_waiting || seq != m_waitSeq) return; // stale: requester already gave up
    std::memcpy(m_resp, bytes, kRespLen);
    m_respReady = true;
    m_cvResp.notify_one();
}

void Server::threadMain(uint16_t port) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        log::error("[gdrl] WSAStartup failed");
        return;
    }
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof yes);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == SOCKET_ERROR ||
        listen(ls, 1) == SOCKET_ERROR) {
        log::error("[gdrl] bind/listen on 127.0.0.1:{} failed (WSA error {})", port, WSAGetLastError());
        return;
    }
    log::info("[gdrl] listening on 127.0.0.1:{}", port);

    for (;;) {
        SOCKET c = accept(ls, nullptr, nullptr);
        if (c == INVALID_SOCKET) { Sleep(100); continue; }
        int nd = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nd), sizeof nd);
        log::info("[gdrl] client connected");
        m_connected = true;
        serve(static_cast<uintptr_t>(c));
        m_connected = false;
        closesocket(c);
        log::info("[gdrl] client disconnected");
    }
}

void Server::serve(uintptr_t sockRaw) {
    SOCKET c = static_cast<SOCKET>(sockRaw);
    uint8_t reqBuf[kReqLen];
    uint8_t resp[kRespLen];

    while (recvAll(c, reqBuf, static_cast<int>(kReqLen))) {
        Request r;
        decodeRequest(reqBuf, r);
        {
            std::lock_guard lk(m_mx);
            r.seq = ++m_seq;
            m_pending = r;
            m_hasPending = true;
            m_waitSeq = r.seq;
            m_waiting = true;
            m_respReady = false;
        }
        m_cvReq.notify_one();

        bool ok;
        {
            std::unique_lock lk(m_mx);
            ok = m_cvResp.wait_for(lk, std::chrono::seconds(10), [&] { return m_respReady; });
            if (ok) std::memcpy(resp, m_resp, kRespLen);
            else m_hasPending = false; // withdraw the request nobody served
            m_waiting = false;
            m_respReady = false;
        }
        if (!ok) {
            float zeros[kObsDim] = {};
            encodeResponse(resp, 0, r.cmd, ErrBusy, 0, 0.f, zeros);
        }
        if (!sendAll(c, resp, static_cast<int>(kRespLen))) break;
        if (r.cmd == Close) break;
    }
}

} // namespace rl
