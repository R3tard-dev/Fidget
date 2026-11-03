#include <Geode/Geode.hpp>
#include <imgui-cocos.hpp>

using namespace geode::prelude;

// Mod menu state
static bool g_menuOpen = false;
static bool g_godMode = false;
static bool g_noclip = false;
static bool g_autoClicker = false;
static bool g_speedHack = false;
static float g_speedMultiplier = 1.0f;

$on_mod(Loaded) {
    ImGuiCocos::get().setup([] {
        ImGui::StyleColorsDark();
    }).draw([] {
        ImGui::SetNextWindowSize(ImVec2(420, 360), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Fidget Mod Menu", &g_menuOpen)) {
            ImGui::Text("Geometry Dash Mod Menu");
            ImGui::Separator();

            ImGui::Checkbox("God Mode", &g_godMode);
            ImGui::Checkbox("Noclip", &g_noclip);
            ImGui::Checkbox("Auto Clicker", &g_autoClicker);

            ImGui::Separator();
            ImGui::Checkbox("Speed Hack", &g_speedHack);
            if (g_speedHack) {
                ImGui::SliderFloat("Speed Multiplier", &g_speedMultiplier, 0.1f, 10.0f);
            }

            ImGui::Separator();
            ImGui::Text("Press TAB to toggle this menu.");
        }
        ImGui::End();

        if (!g_menuOpen) {
            ImGuiCocos::get().setVisible(false);
        }
    });
}

$execute {
    KeyboardInputEvent().listen([](KeyboardInputData& evt) {
        if (evt.key == cocos2d::enumKeyCodes::KEY_Tab &&
            evt.action == KeyboardInputData::Action::Press) {
            g_menuOpen = !g_menuOpen;
            ImGuiCocos::get().setVisible(g_menuOpen);
        }
        return ListenerResult::Propagate;
    }).leak();
}