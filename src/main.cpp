#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <algorithm>
#include <chrono>
#include <vector>
#include "protocol.hpp"
#include "server.hpp"

using namespace geode::prelude;

namespace {

constexpr float kTickDt = 1.0f / 240.0f; // VERIFY for 2.2091 (use dt-scale to compensate)
constexpr float kLookBehind = 30.0f;     // include objects slightly behind the player
bool g_clickHeld = false;

bool rlEnabled() { return Mod::get()->getSettingValue<bool>("rl-enabled"); }

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
    gl->handleButton(held, 1, true); // compiled in step 2
}

struct ObjInfo { float x, y, w, h, type; };

} // namespace

class $modify(RLPlayLayer, PlayLayer) {
    struct Fields {
        bool completed = false;
        bool objsBuilt = false;
        uint32_t tick = 0;
        std::vector<ObjInfo> objs; // non-decoration objects sorted by x
    };

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
        obs[7] = 0.f; // reserved (size/mirror in step 4)

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
        for (; n < rl::kNumObj; ++n, o += rl::kObjFeats) o[0] = -1.f; // empty slot
    }

    void update(float dt) {
        if (!rlEnabled()) {
            PlayLayer::update(dt);
            return;
        }

        auto& srv = rl::Server::get();
        srv.ensureStarted(static_cast<uint16_t>(Mod::get()->getSettingValue<int64_t>("port")));
        if (!srv.connected() || !m_player1) return; // frozen until a client connects

        const float tickDt = kTickDt * static_cast<float>(Mod::get()->getSettingValue<double>("dt-scale"));
        const auto t0 = std::chrono::steady_clock::now();

        rl::Request r;
        while (srv.popRequest(r, std::chrono::microseconds(1000))) {
            uint16_t err = rl::ErrNone;
            float dx = 0.f;

            switch (r.cmd) {
                case rl::Hello:
                    if (r.aux != rl::kProtocolVersion) err = rl::ErrVersion;
                    break;
                case rl::Reset:
                    this->resetLevel(); // step 4: honor r.startX via snapshots
                    m_fields->completed = false;
                    m_fields->tick = 0;
                    g_clickHeld = true;          // force an actual release below
                    setClick(this, false);
                    break;
                case rl::Step: {
                    auto* p = m_player1;
                    if (!p) { err = rl::ErrNoPlayer; break; }
                    int skip = std::clamp<int>(r.skip, 1, 16);
                    setClick(this, r.action != 0);
                    float x0 = p->getPositionX();
                    for (int i = 0; i < skip; ++i) {
                        if (p->m_isDead || m_fields->completed) break;
                        // THE one-tick step (same call as step 2's debug run)
                        PlayLayer::update(tickDt);
                        ++m_fields->tick;
                    }
                    dx = p->getPositionX() - x0;
                    break;
                }
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

            uint8_t out[rl::kRespLen];
            rl::encodeResponse(out, flags, r.cmd, err, m_fields->tick, dx, obs);
            srv.respond(r.seq, out);

            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(12)) break;
        }
    }

    // Without these, GD auto-respawns ~1 s after death and leaves the level on completion,
    // both on real time, which would corrupt episodes.
    void delayedResetLevel() {
        if (rlEnabled()) return;
        PlayLayer::delayedResetLevel();
    }

    void levelComplete() {
        if (rlEnabled()) { m_fields->completed = true; return; }
        PlayLayer::levelComplete();
    }
};
