#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include "protocol.hpp"
#include "server.hpp"

#pragma comment(lib, "user32.lib")

using namespace geode::prelude;

namespace {

constexpr float kTickDt = 1.0f / 240.0f; // VERIFY for 2.2091
constexpr float kLookBehind = 30.0f;
constexpr float kFailErr = 5.0f;         // replay position error (units) that counts as diverged
constexpr float kExactErr = 0.001f;

enum Phase { PhaseUninit = -1, PhaseRecord = 1, PhaseReplay = 2, PhaseServe = 3 };

bool g_clickHeld = false;
bool g_ourCall = false;
int64_t g_blockUntilMs = 0; // native button handling is swallowed until this time (record phase)

int64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

bool rlEnabled()   { return Mod::get()->getSettingValue<bool>("rl-enabled"); }
bool recordFirst() { return Mod::get()->getSettingValue<bool>("record-first"); }
bool useReplay()   { return Mod::get()->getSettingValue<bool>("use-replay"); }
bool hideRender()  { return Mod::get()->getSettingValue<bool>("hide-render"); }
float spacing()    { return static_cast<float>(Mod::get()->getSettingValue<double>("snapshot-spacing")); }
float dtScale()    { return static_cast<float>(Mod::get()->getSettingValue<double>("dt-scale")); }
int budgetMs()     { return static_cast<int>(Mod::get()->getSettingValue<int64_t>("frame-budget-ms")); }

uint16_t serverPort() {
    if (const char* e = std::getenv("GDRL_PORT")) {
        int p = std::atoi(e);
        if (p > 1023 && p < 65536) return static_cast<uint16_t>(p);
    }
    return static_cast<uint16_t>(Mod::get()->getSettingValue<int64_t>("port"));
}

std::string refPath() {
    if (const char* e = std::getenv("GDRL_REF")) {
        if (*e) return e;
    }
    return "gdrl_reference.bin"; // relative to the game folder
}

std::string absPath(const std::string& p) {
    std::error_code ec;
    auto a = std::filesystem::absolute(p, ec);
    return ec ? p : a.string();
}

void popup(const std::string& s) {
    FLAlertLayer::create("GD RL", s.c_str(), "OK")->show();
}

int gamemodeId(PlayerObject* p) {
    if (p->m_isShip)   return 1;
    if (p->m_isBall)   return 2;
    if (p->m_isBird)   return 3; // UFO
    if (p->m_isDart)   return 4; // wave
    if (p->m_isRobot)  return 5;
    if (p->m_isSpider) return 6;
    if (p->m_isSwing)  return 7;
    return 0;                    // cube
}

void setClick(GJBaseGameLayer* gl, bool held) {
    if (held == g_clickHeld) return;
    g_clickHeld = held;
    g_ourCall = true;
    gl->handleButton(held, 1, true);
    g_ourCall = false;
}

// Forces a real release even if we think the button is already up.
void forceRelease(GJBaseGameLayer* gl) {
    g_clickHeld = true;
    setClick(gl, false);
}

// Human input, polled once per tick (Windows only; only while the game window is focused).
bool humanHeld() {
    DWORD pid = 0;
    GetWindowThreadProcessId(GetForegroundWindow(), &pid);
    if (pid != GetCurrentProcessId()) return false;
    auto down = [](int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; };
    return down(VK_SPACE) || down(VK_UP) || down(VK_LBUTTON);
}

// ---- reference file: one byte per tick (bit0 = held, bit1 = forced release before the tick)
//      plus the player's position after each tick, used to verify the replay ----
struct RefFile {
    float dtScale = 1.0f;
    std::vector<uint8_t> bits;
    std::vector<float> x, y;
};

constexpr uint32_t kRefMagic = 0x46455247u; // "GREF"
constexpr uint32_t kRefVersion = 1;

bool saveRef(const std::string& path, const RefFile& r) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t hdr[3] = {kRefMagic, kRefVersion, static_cast<uint32_t>(r.bits.size())};
    bool ok = std::fwrite(hdr, 4, 3, f) == 3 &&
              std::fwrite(&r.dtScale, 4, 1, f) == 1 &&
              std::fwrite(r.bits.data(), 1, r.bits.size(), f) == r.bits.size() &&
              std::fwrite(r.x.data(), 4, r.x.size(), f) == r.x.size() &&
              std::fwrite(r.y.data(), 4, r.y.size(), f) == r.y.size();
    std::fclose(f);
    return ok;
}

bool loadRef(const std::string& path, RefFile& r) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t hdr[3] = {0, 0, 0};
    bool ok = std::fread(hdr, 4, 3, f) == 3 && hdr[0] == kRefMagic && hdr[1] == kRefVersion &&
              hdr[2] > 0 && hdr[2] < 50000000u;
    if (ok) {
        size_t n = hdr[2];
        r.bits.resize(n);
        r.x.resize(n);
        r.y.resize(n);
        ok = std::fread(&r.dtScale, 4, 1, f) == 1 &&
             std::fread(r.bits.data(), 1, n, f) == n &&
             std::fread(r.x.data(), 4, n, f) == n &&
             std::fread(r.y.data(), 4, n, f) == n;
    }
    std::fclose(f);
    return ok;
}

struct ObjInfo { float x, y, w, h, type; };
struct Snap { int bucket; float x; uint32_t logLen; Ref<CheckpointObject> cp; };

} // namespace

// While recording, the game's own button handling is dropped so that ONLY our polled input
// (applied through setClick) reaches the player. That makes record and replay identical.
class $modify(RLBaseLayer, GJBaseGameLayer) {
    void handleButton(bool down, int button, bool isPlayer1) {
        if (!g_ourCall && button == 1 && nowMs() < g_blockUntilMs) return;
        GJBaseGameLayer::handleButton(down, button, isPlayer1);
    }
};

class $modify(RLPlayLayer, PlayLayer) {
    struct Fields {
        int phase = PhaseUninit;
        bool completed = false;
        bool objsBuilt = false;
        bool hidden = false;
        bool startCaptured = false;
        bool ownReset = false;
        bool tainted = false;
        bool boundaryPending = false;
        float startX = 0.f, startY = 0.f;
        float acc = 0.f;
        float maxErr = 0.f;
        size_t replayIdx = 0;
        uint32_t tick = 0;
        RefFile ref;
        std::vector<ObjInfo> objs;
        std::vector<Snap> snaps; // ascending x, along the final successful path
    };

    // ------------------------------------------------------------ phases
    void initPhase() {
        m_fields->phase = PhaseServe;
        if (recordFirst()) {
            m_fields->phase = PhaseRecord;
            log::info("[gdrl] RECORD phase: play the level; Space/Up/Click at fixed 240 TPS");
            return;
        }
        if (!useReplay()) return;
        RefFile r;
        std::string path = refPath();
        if (!loadRef(path, r)) {
            log::info("[gdrl] no reference file at {}; serving without snapshots", absPath(path));
            return;
        }
        if (std::fabs(r.dtScale - dtScale()) > 1e-6f) {
            popup("Reference file was recorded with a different dt-scale; replay skipped.");
            return;
        }
        log::info("[gdrl] REPLAY phase: {} ticks from {}", r.bits.size(), absPath(path));
        m_fields->ref = std::move(r);
        m_fields->phase = PhaseReplay;
    }

    void applyVisibility() {
        bool hide = m_fields->phase != PhaseRecord && hideRender();
        if (hide != m_fields->hidden) {
            m_fields->hidden = hide;
            this->setVisible(!hide);
        }
    }

    // Shared start of record and replay: a real restart, button released, nothing captured yet.
    void beginReference() {
        m_fields->ownReset = true;
        this->resetLevel();
        m_fields->ownReset = false;
        forceRelease(this);
        m_fields->startX = m_player1->getPositionX();
        m_fields->startY = m_player1->getPositionY();
        m_fields->startCaptured = true;
        m_fields->snaps.clear();
        m_fields->acc = 0.f;
        m_fields->boundaryPending = false;
        m_fields->completed = false;
        m_fields->replayIdx = 0;
        m_fields->maxErr = 0.f;
        m_fields->tick = 0;
    }

    // ------------------------------------------------------------ snapshots
    void captureSnapshot(uint32_t logLen) {
        auto* p = m_player1;
        if (!p || p->m_isDead || m_fields->completed) return;
        float x = p->getPositionX();
        auto& s = m_fields->snaps;
        while (!s.empty() && s.back().x > x) s.pop_back();
        int b = static_cast<int>(x / spacing());
        if (!s.empty() && s.back().bucket >= b) return;
        CheckpointObject* cp = this->createCheckpoint(); // compiled in step 4
        if (!cp) return;
        s.push_back(Snap{b, x, logLen, Ref<CheckpointObject>(cp)});
    }

    void restoreStart(float startX, uint16_t& err) {
        auto& s = m_fields->snaps;
        if (m_checkpointArray) m_checkpointArray->removeAllObjects();
        if (startX > 0.f && !s.empty()) {
            auto it = std::upper_bound(s.begin(), s.end(), startX,
                [](float v, const Snap& sn) { return v < sn.x; });
            size_t i = (it == s.begin()) ? 0 : static_cast<size_t>(it - s.begin()) - 1;
            if (!m_isPracticeMode) this->togglePracticeMode(true);
            m_checkpointArray->addObject(s[i].cp.data());
        } else if (startX > 0.f) {
            err = rl::ErrNoSnapshots;
        }
        this->resetLevel();
        if (m_checkpointArray) m_checkpointArray->removeAllObjects();
    }

    // ------------------------------------------------------------ record phase
    void truncateLog(size_t L) {
        auto& r = m_fields->ref;
        if (L < r.bits.size()) {
            r.bits.resize(L);
            r.x.resize(L);
            r.y.resize(L);
        }
        auto& s = m_fields->snaps;
        while (!s.empty() && s.back().logLen > L) s.pop_back();
        m_fields->tick = static_cast<uint32_t>(L);
    }

    // The game just restarted or reloaded a checkpoint while a reference run was being recorded:
    // cut the log back to the tick the player is now at.
    void onRecordReset() {
        auto* p = m_player1;
        if (!p || !m_fields->startCaptured) return;
        auto& r = m_fields->ref;
        float x = p->getPositionX(), y = p->getPositionY();
        m_fields->acc = 0.f;

        long found = -1;
        if (x == m_fields->startX && y == m_fields->startY) {
            found = 0;
        } else {
            for (size_t i = r.x.size(); i-- > 0;) {
                if (r.x[i] == x && r.y[i] == y) { found = static_cast<long>(i) + 1; break; }
            }
        }
        if (found < 0) {
            m_fields->tainted = true;
            log::warn("[gdrl] reset to ({:.3f},{:.3f}) matches no recorded tick: reference run is invalid", x, y);
            return;
        }
        truncateLog(static_cast<size_t>(found));
        forceRelease(this);
        m_fields->boundaryPending = found > 0;
    }

    void recordStep(float tickDt) {
        bool h = humanHeld();
        setClick(this, h);
        PlayLayer::update(tickDt);
        auto* p = m_player1;
        if (p->m_isDead) return; // dead ticks belong to no valid path
        auto& r = m_fields->ref;
        uint8_t b = static_cast<uint8_t>((h ? 1 : 0) | (m_fields->boundaryPending ? 2 : 0));
        m_fields->boundaryPending = false;
        r.bits.push_back(b);
        r.x.push_back(p->getPositionX());
        r.y.push_back(p->getPositionY());
        m_fields->tick = static_cast<uint32_t>(r.bits.size());
        this->captureSnapshot(static_cast<uint32_t>(r.bits.size()));
    }

    void runRecord(float dt, float tickDt) {
        g_blockUntilMs = nowMs() + 300;
        if (!m_fields->startCaptured) {
            this->beginReference();
            m_fields->ref = RefFile{};
            m_fields->ref.dtScale = dtScale();
            return;
        }
        m_fields->acc += std::min(dt, 0.1f);
        int n = 0;
        while (m_fields->acc >= tickDt && n < 16 && !m_fields->completed) {
            this->recordStep(tickDt);
            m_fields->acc -= tickDt;
            ++n;
        }
        if (m_fields->acc > 4.f * tickDt) m_fields->acc = 4.f * tickDt;
        if (m_fields->completed) this->finishRecording();
    }

    void finishRecording() {
        g_blockUntilMs = 0;
        m_fields->phase = PhaseServe;
        auto& r = m_fields->ref;
        std::string path = refPath();
        std::string msg;
        if (m_fields->tainted) {
            msg = "Reference run NOT saved: a checkpoint reload could not be matched to the recording.\n"
                  "Redo it without placing manual practice checkpoints.\n"
                  "(Snapshots stay usable in this session.)";
        } else if (saveRef(path, r)) {
            msg = fmt::format("Recorded {} ticks ({:.1f} s), {} snapshots.\nSaved to {}\nRL server is now listening.",
                              r.bits.size(), static_cast<double>(r.bits.size()) / 240.0,
                              m_fields->snaps.size(), absPath(path));
        } else {
            msg = fmt::format("Could not write {}", absPath(path));
        }
        log::info("[gdrl] {}", msg);
        popup(msg);
        r = RefFile{};
    }

    // ------------------------------------------------------------ replay phase
    void runReplay(float tickDt) {
        if (!m_fields->startCaptured) {
            this->beginReference();
            return;
        }
        auto& r = m_fields->ref;
        const size_t n = r.bits.size();
        const auto t0 = std::chrono::steady_clock::now();
        bool failed = false;

        while (m_fields->replayIdx < n && !m_fields->completed && !m_player1->m_isDead) {
            size_t i = m_fields->replayIdx;
            if (r.bits[i] & 2) forceRelease(this);
            setClick(this, (r.bits[i] & 1) != 0);
            PlayLayer::update(tickDt);

            auto* p = m_player1;
            float err = std::max(std::fabs(p->getPositionX() - r.x[i]),
                                 std::fabs(p->getPositionY() - r.y[i]));
            m_fields->maxErr = std::max(m_fields->maxErr, err);
            m_fields->replayIdx = i + 1;
            m_fields->tick = static_cast<uint32_t>(i + 1);
            this->captureSnapshot(static_cast<uint32_t>(i + 1));

            if (err > kFailErr) { failed = true; break; }
            if (((i + 1) & 63) == 0 &&
                std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(8)) {
                return; // yield so the window stays responsive; resume next frame
            }
        }
        this->finishReplay(failed);
    }

    void finishReplay(bool failed) {
        bool dead = m_player1->m_isDead;
        bool ok = !failed && !dead && m_fields->completed;
        size_t n = m_fields->ref.bits.size();
        std::string msg;
        if (ok) {
            msg = fmt::format("Replay OK: {} ticks, {} snapshots.\nMax position error vs recording: {:.6f} ({}).\nRL server is now listening.",
                              n, m_fields->snaps.size(), m_fields->maxErr,
                              m_fields->maxErr <= kExactErr ? "bit-exact" : "NOT bit-exact");
        } else {
            m_fields->snaps.clear();
            msg = fmt::format("Replay FAILED at tick {}/{} (dead={}, completed={}, max error {:.3f}).\nSnapshots discarded.",
                              m_fields->replayIdx, n, dead, m_fields->completed, m_fields->maxErr);
        }
        log::info("[gdrl] {}", msg);
        popup(msg);
        m_fields->completed = false;
        m_fields->phase = PhaseServe;
        m_fields->ref = RefFile{};
    }

    // ------------------------------------------------------------ observation
    void buildObjectCache() {
        auto& v = m_fields->objs;
        v.clear();
        m_fields->objsBuilt = true;
        if (!m_objects) return;
        for (auto* obj : CCArrayExt<GameObject*>(m_objects)) {
            if (!obj) continue;
            if (obj->m_objectType == GameObjectType::Decoration) continue;
            auto r = obj->getObjectRect();
            v.push_back({obj->getPositionX(), obj->getPositionY(),
                         r.size.width, r.size.height,
                         static_cast<float>(static_cast<int>(obj->m_objectType))});
        }
        std::sort(v.begin(), v.end(), [](const ObjInfo& a, const ObjInfo& b) {
            return a.x != b.x ? a.x < b.x : a.y < b.y;
        });
        log::info("[gdrl] cached {} non-decoration objects", v.size());
    }

    void fillObs(float* obs) {
        std::fill(obs, obs + rl::kObsDim, 0.f);
        auto* p = m_player1;
        float px = p->getPositionX(), py = p->getPositionY();
        obs[0] = px;
        obs[1] = py;
        obs[2] = p->m_yVelocity;
        obs[3] = static_cast<float>(gamemodeId(p));
        obs[4] = p->m_playerSpeed;
        obs[5] = p->m_isUpsideDown ? -1.f : 1.f;
        obs[6] = p->m_isOnGround ? 1.f : 0.f;
        obs[7] = 0.f;

        if (!m_fields->objsBuilt) buildObjectCache();
        auto& objs = m_fields->objs;
        auto it = std::lower_bound(objs.begin(), objs.end(), px - kLookBehind,
            [](const ObjInfo& o, float v) { return o.x < v; });

        float* o = obs + rl::kPlayerFeats;
        int n = 0;
        for (; it != objs.end() && n < rl::kNumObj; ++it, ++n, o += rl::kObjFeats) {
            o[0] = it->type;
            o[1] = it->x - px;
            o[2] = it->y - py;
            o[3] = it->w;
            o[4] = it->h;
        }
        for (; n < rl::kNumObj; ++n, o += rl::kObjFeats) o[0] = -1.f;
    }

    // ------------------------------------------------------------ serve phase
    void runServe(float tickDt) {
        auto& srv = rl::Server::get();
        srv.ensureStarted(serverPort());
        if (!srv.connected()) return; // frozen until a client connects

        const auto t0 = std::chrono::steady_clock::now();
        const auto budget = std::chrono::milliseconds(budgetMs());

        rl::Request r;
        while (srv.popRequest(r, std::chrono::microseconds(1000))) {
            uint16_t err = rl::ErrNone;
            float dx = 0.f;

            switch (r.cmd) {
                case rl::Hello:
                    if (r.aux != rl::kProtocolVersion) err = rl::ErrVersion;
                    break;
                case rl::Reset:
                    this->restoreStart(r.startX, err);
                    m_fields->completed = false;
                    m_fields->tick = 0;
                    forceRelease(this);
                    break;
                case rl::Step: {
                    auto* p = m_player1;
                    if (!p) { err = rl::ErrNoPlayer; break; }
                    int skip = std::clamp<int>(r.skip, 1, 16);
                    setClick(this, r.action != 0);
                    float x0 = p->getPositionX();
                    for (int i = 0; i < skip; ++i) {
                        if (p->m_isDead || m_fields->completed) break;
                        PlayLayer::update(tickDt);
                        ++m_fields->tick;
                    }
                    dx = p->getPositionX() - x0;
                    break;
                }
                case rl::Info:
                case rl::Close:
                    break;
                default:
                    err = rl::ErrBadCmd;
            }

            uint8_t flags = 0;
            float obs[rl::kObsDim];
            if (m_player1) {
                bool dead = m_player1->m_isDead;
                if (dead) flags |= rl::FlagDead;
                if (m_fields->completed) flags |= rl::FlagSuccess;
                if (dead || m_fields->completed) flags |= rl::FlagDone;
                fillObs(obs);
            } else {
                std::fill(obs, obs + rl::kObsDim, 0.f);
                err = rl::ErrNoPlayer;
            }

            uint32_t tickOut = m_fields->tick;
            if (r.cmd == rl::Info) {
                auto& s = m_fields->snaps;
                size_t base = static_cast<size_t>(r.aux) * rl::kObsDim;
                std::fill(obs, obs + rl::kObsDim, 0.f);
                for (size_t i = 0; i < static_cast<size_t>(rl::kObsDim) && base + i < s.size(); ++i)
                    obs[i] = s[base + i].x;
                tickOut = static_cast<uint32_t>(s.size());
                dx = spacing();
            }

            uint8_t out[rl::kRespLen];
            rl::encodeResponse(out, flags, r.cmd, err, tickOut, dx, obs);
            srv.respond(r.seq, out);

            if (std::chrono::steady_clock::now() - t0 > budget) break;
        }
    }

    // ------------------------------------------------------------ hooks
    void update(float dt) {
        if (!rlEnabled() || !m_player1) {
            if (m_fields->hidden) {
                m_fields->hidden = false;
                this->setVisible(true);
            }
            PlayLayer::update(dt);
            return;
        }
        if (m_fields->phase == PhaseUninit) this->initPhase();
        this->applyVisibility();

        const float tickDt = kTickDt * dtScale();
        if (m_fields->phase == PhaseRecord) { this->runRecord(dt, tickDt); return; }
        if (m_fields->phase == PhaseReplay) { this->runReplay(tickDt); return; }
        this->runServe(tickDt);
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        if (m_fields->ownReset || m_fields->phase != PhaseRecord || !rlEnabled()) return;
        this->onRecordReset();
    }

    // Vanilla auto-respawn must not fire while the trainer or the replay is in control.
    void delayedResetLevel() {
        if (rlEnabled() && (m_fields->phase == PhaseReplay || m_fields->phase == PhaseServe)) return;
        PlayLayer::delayedResetLevel();
    }

    void levelComplete() {
        if (!rlEnabled()) { PlayLayer::levelComplete(); return; }
        m_fields->completed = true; // swallow the end sequence in every RL phase
    }
};
