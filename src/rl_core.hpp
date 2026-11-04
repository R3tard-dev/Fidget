#pragma once

namespace rl {
    // GD's default physics rate is 240 ticks/s (VERIFY for 2.2091).
    constexpr float kTickDt = 1.0f / 240.0f;

    // Last click state we applied, so we only send press/release on changes.
    inline bool g_clickHeld = false;
}
