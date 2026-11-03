#include <Geode/Geode.hpp>
#include <imgui-cocos.hpp>
#include <imgui.h>

using namespace geode::prelude;

// Mod menu state
static bool g_menuOpen = false;

// Categories
static int g_selectedCategory = 0;
static const char* g_categories[] = { "Player", "Movement", "Visual", "Misc" };

// Player
static bool g_godMode = false;
static bool g_noclip = false;
static bool g_autoClicker = false;

// Movement
static bool g_speedHack = false;
static float g_speedMultiplier = 1.0f;
static bool g_jumpHack = false;
static float g_jumpMultiplier = 1.0f;

// Visual
static bool g_showHitboxes = false;
static bool g_showTrajectory = false;
static bool g_noGlow = false;

// Misc
static bool g_safeMode = false;
static bool g_autoSave = false;

static void applyMegaHackStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(8, 6);
    style.WindowPadding = ImVec2(10, 10);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, 0.98f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_Border] = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.28f, 0.28f, 0.34f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.50f, 0.80f, 1.00f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.18f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.26f, 0.32f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.34f, 0.34f, 0.40f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.18f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.26f, 0.32f, 1.00f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.34f, 0.34f, 0.40f, 1.00f);
    colors[ImGuiCol_Separator] = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
    colors[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 0.97f, 1.00f);
}

$on_mod(Loaded) {
    ImGuiCocos::get().setup([] {
        applyMegaHackStyle();
    }).draw([] {
        if (!g_menuOpen) return;

        ImGui::SetNextWindowSize(ImVec2(520, 360), ImGuiCond_FirstUseEver);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;
        if (ImGui::Begin("Fidget", &g_menuOpen, flags)) {
            // Sidebar
            ImGui::BeginChild("##sidebar", ImVec2(130, 0), true);
            for (int i = 0; i < IM_ARRAYSIZE(g_categories); i++) {
                if (ImGui::Selectable(g_categories[i], g_selectedCategory == i, 0, ImVec2(0, 22))) {
                    g_selectedCategory = i;
                }
            }
            ImGui::EndChild();

            ImGui::SameLine();

            // Content
            ImGui::BeginChild("##content", ImVec2(0, 0), true);
            if (g_selectedCategory == 0) { // Player
                ImGui::TextDisabled("Player");
                ImGui::Separator();
                ImGui::Checkbox("God Mode", &g_godMode);
                ImGui::Checkbox("Noclip", &g_noclip);
                ImGui::Checkbox("Auto Clicker", &g_autoClicker);
            } else if (g_selectedCategory == 1) { // Movement
                ImGui::TextDisabled("Movement");
                ImGui::Separator();
                ImGui::Checkbox("Speed Hack", &g_speedHack);
                if (g_speedHack) ImGui::SliderFloat("Speed", &g_speedMultiplier, 0.1f, 10.0f);
                ImGui::Checkbox("Jump Hack", &g_jumpHack);
                if (g_jumpHack) ImGui::SliderFloat("Jump", &g_jumpMultiplier, 0.1f, 10.0f);
            } else if (g_selectedCategory == 2) { // Visual
                ImGui::TextDisabled("Visual");
                ImGui::Separator();
                ImGui::Checkbox("Show Hitboxes", &g_showHitboxes);
                ImGui::Checkbox("Show Trajectory", &g_showTrajectory);
                ImGui::Checkbox("No Glow", &g_noGlow);
            } else { // Misc
                ImGui::TextDisabled("Misc");
                ImGui::Separator();
                ImGui::Checkbox("Safe Mode", &g_safeMode);
                ImGui::Checkbox("Auto Save", &g_autoSave);
            }
            ImGui::EndChild();
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