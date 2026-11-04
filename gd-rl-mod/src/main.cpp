#define WIN32_LEAN_AND_MEAN
#define _WINSOCK_DEPRECATED_

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include <Geode/modify/GameManager.hpp>

#pragma comment(lib, "ws2_32.lib")

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <sstream>
#include <cmath>

using namespace geode::prelude;

// ---------------------------------------------------------------------------
// Global socket state
// ---------------------------------------------------------------------------
static SOCKET g_listenSock = INVALID_SOCKET;
static SOCKET g_clientSock = INVALID_SOCKET;
static std::mutex g_sockMutex;
static std::atomic<bool> g_sockReady{false};
static std::atomic<bool> g_running{true};

// Latest incoming command
static std::string g_pendingCmd;
static std::mutex g_cmdMutex;

// Parsed level object X positions (for nearest-obstacle distance)
static std::vector<float> g_obstacleXs;

// ---------------------------------------------------------------------------
// Focus-loss bypass: keep the game running when tabbed out
// ---------------------------------------------------------------------------
class $modify(RLAppDelegate, AppDelegate) {
    void applicationDidEnterBackground() {
        // Intentionally do nothing: prevent GD from pausing on focus loss.
    }
    void applicationWillResignActive() {
        // Intentionally do nothing: keep ticking at full speed.
    }
    void applicationWillEnterForeground() {
        // no-op resume handling; let the game continue
    }
};

class $modify(RLGameManager, GameManager) {
    void applicationDidEnterBackground() {
        // no-op: prevent pause on focus loss
    }
    void applicationWillEnterForeground() {
        // no-op
    }
};

// ---------------------------------------------------------------------------
// Socket server thread
// ---------------------------------------------------------------------------
static void socketServerThread() {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return;

    g_listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_listenSock == INVALID_SOCKET) { WSACleanup(); return; }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(3000);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    // Allow quick restart
    int opt = 1;
    setsockopt(g_listenSock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    if (bind(g_listenSock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(g_listenSock); WSACleanup(); return;
    }
    if (listen(g_listenSock, 1) == SOCKET_ERROR) {
        closesocket(g_listenSock); WSACleanup(); return;
    }

    // Non-blocking accept
    u_long nb = 1;
    ioctlsocket(g_listenSock, FIONBIO, &nb);

    while (g_running.load()) {
        // Accept new client if none connected
        {
            std::lock_guard<std::mutex> lock(g_sockMutex);
            if (g_clientSock == INVALID_SOCKET) {
                SOCKET c = accept(g_listenSock, nullptr, nullptr);
                if (c != INVALID_SOCKET) {
                    u_long cnb = 1;
                    ioctlsocket(c, FIONBIO, &cnb);
                    g_clientSock = c;
                    g_sockReady.store(true);
                }
            }
        }

        // Receive commands from client
        if (g_clientSock != INVALID_SOCKET) {
            char buf[256];
            int r = recv(g_clientSock, buf, sizeof(buf) - 1, 0);
            if (r > 0) {
                buf[r] = '\0';
                std::string data(buf);
                // Parse first token
                std::istringstream iss(data);
                std::string cmd;
                iss >> cmd;
                if (!cmd.empty()) {
                    std::lock_guard<std::mutex> lock(g_cmdMutex);
                    g_pendingCmd = cmd;
                }
            } else if (r == 0 || (r == SOCKET_ERROR && WSAGetLastError() != WSAEWOULDBLOCK)) {
                // Disconnected
                std::lock_guard<std::mutex> lock(g_sockMutex);
                closesocket(g_clientSock);
                g_clientSock = INVALID_SOCKET;
                g_sockReady.store(false);
            }
        }

        Sleep(1);
    }

    if (g_clientSock != INVALID_SOCKET) closesocket(g_clientSock);
    closesocket(g_listenSock);
    WSACleanup();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == delim) { out.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static void parseLevelObjects(PlayLayer* pl) {
    g_obstacleXs.clear();
    // PlayLayer stores the level string in m_level->m_levelString (may be empty in some cases)
    auto* level = pl->m_level;
    if (!level) return;
    std::string lvl = level->m_levelString;
    if (lvl.empty()) return;

    // Objects separated by ';'
    auto objects = split(lvl, ';');
    for (auto& obj : objects) {
        if (obj.empty()) continue;
        auto kv = split(obj, ',');
        // pairs: key,value,key,value...
        float x = -1.f;
        for (size_t i = 0; i + 1 < kv.size(); i += 2) {
            int key = 0;
            try { key = std::stoi(kv[i]); } catch (...) { continue; }
            if (key == 2) {
                try { x = std::stof(kv[i + 1]); } catch (...) {}
                break;
            }
        }
        if (x >= 0.f) g_obstacleXs.push_back(x);
    }
}

static float nearestObstacleDistance(float playerX) {
    float best = 1e9f;
    for (float ox : g_obstacleXs) {
        if (ox > playerX) {
            float d = ox - playerX;
            if (d < best) best = d;
        }
    }
    if (best > 1e8f) best = 0.f;
    return best;
}

// ---------------------------------------------------------------------------
// PlayLayer hooks
// ---------------------------------------------------------------------------
class $modify(RLPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool dontCreateObjects) {
        if (!PlayLayer::init(level, dontCreateObjects)) return false;
        parseLevelObjects(this);
        return true;
    }

    void update(float dt) {
        PlayLayer::update(dt);

        if (!g_sockReady.load()) return;

        auto* player = this->m_player1;
        if (!player) return;

        auto pos = player->getPosition();
        float px = pos.x;
        float py = pos.y;
        float vy = (float)player->m_yVelocity;
        float dist = nearestObstacleDistance(px);
        bool dead = this->m_isDead;

        // Build JSON
        std::ostringstream js;
        js << "{\"x\":" << px
           << ",\"y\":" << py
           << ",\"vy\":" << vy
           << ",\"dist\":" << dist
           << ",\"dead\":" << (dead ? "true" : "false") << "}\n";
        std::string msg = js.str();

        // Send state
        {
            std::lock_guard<std::mutex> lock(g_sockMutex);
            if (g_clientSock != INVALID_SOCKET) {
                send(g_clientSock, msg.c_str(), (int)msg.size(), 0);
            }
        }

        // Process pending command
        std::string cmd;
        {
            std::lock_guard<std::mutex> lock(g_cmdMutex);
            cmd = g_pendingCmd;
            g_pendingCmd.clear();
        }
        if (!cmd.empty()) {
            if (cmd == "JUMP") {
                player->pushButton(PlayerButton::Jump);
            } else if (cmd == "RELEASE") {
                player->releaseButton(PlayerButton::Jump);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Mod entry
// ---------------------------------------------------------------------------
$on_mod(Loaded) {
    g_running.store(true);
    std::thread(socketServerThread).detach();
    log::info("GD RL Bridge loaded. Listening on 127.0.0.1:3000");
}

$on_mod(Unloaded) {
    g_running.store(false);
    Sleep(50);
    std::lock_guard<std::mutex> lock(g_sockMutex);
    if (g_clientSock != INVALID_SOCKET) { closesocket(g_clientSock); g_clientSock = INVALID_SOCKET; }
    if (g_listenSock != INVALID_SOCKET) { closesocket(g_listenSock); g_listenSock = INVALID_SOCKET; }
}