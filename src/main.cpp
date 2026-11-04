#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <cmath>
#include <map>
#include "rl_core.hpp"

using namespace geode::prelude;

namespace {

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
    if (held == rl::g_clickHeld) return;
    rl::g_clickHeld = held;
    // VERIFY this signature in the bindings. If it fails to compile, try:
    //   held ? gl->pushButton(PlayerButton::Jump, true)
    //        : gl->releaseButton(PlayerButton::Jump, true);
    gl->handleButton(held, 1, true);
}

} // namespace

class $modify(RLPlayLayer, PlayLayer) {
    struct Fields {
        bool started = false;
        bool reported = false;
        int tick = 0;
        int total = 0;
        float prevX = 0.f;
        std::map<long, int> hist; // round(dx*1000) -> count
    };

    void update(float dt) {
        if (!Mod::get()->getSettingValue<bool>("rl-enabled")) {
            PlayLayer::update(dt);
            return;
        }

        auto* p = m_player1;
        if (!p) {
            PlayLayer::update(dt);
            return;
        }

        if (!m_fields->started) {
            m_fields->started = true;
            m_fields->total = static_cast<int>(Mod::get()->getSettingValue<int64_t>("debug-ticks"));
            m_fields->prevX = p->getPositionX();
            rl::g_clickHeld = false;
            log::info("[gdrl] debug run start: {} ticks, dt={:.6f}", m_fields->total, rl::kTickDt);
        }

        if (m_fields->reported) return; // frozen after the run; turn the setting off to resume

        if (m_fields->tick >= m_fields->total || p->m_isDead) {
            m_fields->reported = true;
            log::info("[gdrl] run ended after {} ticks (dead={})", m_fields->tick, p->m_isDead);
            int shown = 0;
            for (auto& [k, n] : m_fields->hist) {
                if (shown++ >= 8) break;
                log::info("[gdrl]   dx*1000={} -> {} ticks", k, n);
            }
            return;
        }

        bool click = ((m_fields->tick / 60) % 2) == 1;
        setClick(this, click);

        // The one-tick step: call the ORIGINAL update with a fixed dt.
        PlayLayer::update(rl::kTickDt);

        float x = p->getPositionX();
        float y = p->getPositionY();
        float dx = x - m_fields->prevX;
        m_fields->prevX = x;
        m_fields->hist[std::lround(dx * 1000.0f)]++;
        ++m_fields->tick;

        if (m_fields->tick <= 5 || m_fields->tick % 10 == 0) {
            log::info(
                "[gdrl] t={} x={:.3f} y={:.3f} vy={:.3f} dx={:.4f} mode={} spd={:.2f} gnd={} up={} click={} dead={}",
                m_fields->tick, x, y, p->m_yVelocity, dx, gamemodeId(p), p->m_playerSpeed,
                p->m_isOnGround, p->m_isUpsideDown, click, p->m_isDead
            );
        }
    }
};
