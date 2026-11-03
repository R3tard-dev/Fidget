#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <imgui-cocos.hpp>
#include <imgui.h>

using namespace geode::prelude;

// Mod menu state
static bool g_menuOpen = false;

// Categories
static int g_selectedCategory = 0;
static const char* g_categories[] = { "Music", "Settings" };

// Music
static bool g_musicUnlocker = false;

static void applyStyle() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 12.0f;
    s.ChildRounding = 10.0f;
    s.FrameRounding = 8.0f;
    s.GrabRounding = 8.0f;
    s.ScrollbarRounding = 8.0f;
    s.TabRounding = 8.0f;
    s.WindowBorderSize = 0.0f;
    s.ChildBorderSize = 1.0f;
    s.FrameBorderSize = 0.0f;
    s.PopupBorderSize = 1.0f;
    s.WindowPadding = ImVec2(0, 0);
    s.FramePadding = ImVec2(12, 8);
    s.ItemSpacing = ImVec2(10, 8);
    s.ItemInnerSpacing = ImVec2(8, 6);
    s.IndentSpacing = 20.0f;
    s.ScrollbarSize = 14.0f;
    s.GrabMinSize = 12.0f;

    ImVec4* c = s.Colors;
    // Base
    c[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.08f, 0.99f);
    c[ImGuiCol_ChildBg] = ImVec4(0.08f, 0.08f, 0.11f, 1.00f);
    c[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.11f, 0.98f);
    c[ImGuiCol_Border] = ImVec4(0.18f, 0.18f, 0.24f, 1.00f);
    c[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    // Text
    c[ImGuiCol_Text] = ImVec4(0.96f, 0.96f, 0.98f, 1.00f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.45f, 0.45f, 0.52f, 1.00f);
    // Frames
    c[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.17f, 0.17f, 0.22f, 1.00f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);
    // Title
    c[ImGuiCol_TitleBg] = ImVec4(0.05f, 0.05f, 0.07f, 1.00f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.10f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed] = ImVec4(0.05f, 0.05f, 0.07f, 1.00f);
    // Menu bar
    c[ImGuiCol_MenuBarBg] = ImVec4(0.08f, 0.08f, 0.11f, 1.00f);
    // Scrollbar
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.06f, 0.06f, 0.08f, 1.00f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.30f, 0.30f, 0.36f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.38f, 0.38f, 0.44f, 1.00f);
    // Checkmark / slider (accent: electric blue)
    c[ImGuiCol_CheckMark] = ImVec4(0.30f, 0.65f, 1.00f, 1.00f);
    c[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.65f, 1.00f, 1.00f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.45f, 0.75f, 1.00f, 1.00f);
    // Buttons
    c[ImGuiCol_Button] = ImVec4(0.14f, 0.14f, 0.19f, 1.00f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.20f, 0.20f, 0.26f, 1.00f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.26f, 0.26f, 0.32f, 1.00f);
    // Headers (selectable)
    c[ImGuiCol_Header] = ImVec4(0.14f, 0.14f, 0.19f, 1.00f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.20f, 0.20f, 0.26f, 1.00f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.65f, 1.00f, 0.35f);
    // Separator
    c[ImGuiCol_Separator] = ImVec4(0.18f, 0.18f, 0.24f, 1.00f);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.30f, 0.65f, 1.00f, 0.60f);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.30f, 0.65f, 1.00f, 1.00f);
    // Resize grip
    c[ImGuiCol_ResizeGrip] = ImVec4(0.30f, 0.65f, 1.00f, 0.20f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(0.30f, 0.65f, 1.00f, 0.50f);
    c[ImGuiCol_ResizeGripActive] = ImVec4(0.30f, 0.65f, 1.00f, 0.90f);
    // Tabs
    c[ImGuiCol_Tab] = ImVec4(0.10f, 0.10f, 0.14f, 1.00f);
    c[ImGuiCol_TabHovered] = ImVec4(0.20f, 0.20f, 0.26f, 1.00f);
    c[ImGuiCol_TabActive] = ImVec4(0.16f, 0.16f, 0.22f, 1.00f);
    c[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.08f, 0.11f, 1.00f);
    c[ImGuiCol_TabUnfocusedActive] = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
}

$on_mod(Loaded) {
    ImGuiCocos::get().setup([] {
        applyStyle();
    }).draw([] {        if (!g_menuOpen) return;

        ImGui::SetNextWindowSize(ImVec2(640, 440), ImGuiCond_FirstUseEver);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar;
        if (ImGui::Begin("##Fidget", &g_menuOpen, flags)) {
            // Custom title bar
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f, 0.05f, 0.07f, 1.00f));
            ImGui::BeginChild("##titlebar", ImVec2(0, 44), true, ImGuiWindowFlags_NoScrollbar);
            ImGui::SetCursorPos(ImVec2(16, 10));
            ImGui::TextColored(ImVec4(0.30f, 0.65f, 1.00f, 1.00f), "FIDGET");
            ImGui::SameLine();
            ImGui::SetCursorPosY(12);
            ImGui::TextDisabled("Mod Menu");
            ImGui::SameLine(ImGui::GetWindowWidth() - 40);
            ImGui::SetCursorPosY(8);
            if (ImGui::Button("X", ImVec2(28, 28))) g_menuOpen = false;
            ImGui::EndChild();
            ImGui::PopStyleColor();

            // Body
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 12));
            // Sidebar
            ImGui::BeginChild("##sidebar", ImVec2(160, 0), true);
            ImGui::SetCursorPosY(8);
            ImGui::TextDisabled("CATEGORIES");
            ImGui::Spacing();
            for (int i = 0; i < IM_ARRAYSIZE(g_categories); i++) {
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
                if (ImGui::Selectable(g_categories[i], g_selectedCategory == i, 0, ImVec2(0, 32))) {
                    g_selectedCategory = i;
                }
                ImGui::PopStyleVar();
            }
            ImGui::EndChild();

            ImGui::SameLine();

            // Content
            ImGui::BeginChild("##content", ImVec2(0, 0), true);
            ImGui::SetCursorPosY(8);

            if (g_selectedCategory == 0) { // Music
                ImGui::TextColored(ImVec4(0.30f, 0.65f, 1.00f, 1.00f), "Music");
                ImGui::Separator();
                ImGui::Spacing();

                // Card
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.10f, 0.14f, 1.00f));
                ImGui::BeginChild("##musiccard", ImVec2(0, 90), true);
                ImGui::SetCursorPos(ImVec2(14, 12));
                ImGui::Text("Music Unlocker");
                ImGui::SetCursorPos(ImVec2(14, 34));
                ImGui::TextDisabled("Allows playback of locked music tracks.");
                ImGui::SetCursorPos(ImVec2(14, 58));
                ImGui::Checkbox("##mu", &g_musicUnlocker);
                ImGui::SameLine();
                ImGui::TextDisabled(g_musicUnlocker ? "Enabled" : "Disabled");
                ImGui::EndChild();
                ImGui::PopStyleColor();
            } else { // Settings
                ImGui::TextColored(ImVec4(0.30f, 0.65f, 1.00f, 1.00f), "Settings");
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::TextDisabled("Press TAB to toggle this menu.");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextDisabled("Fidget v1.0.0");
            ImGui::EndChild();
            ImGui::PopStyleVar();
        }
        ImGui::End();

        if (!g_menuOpen) {
            ImGuiCocos::get().setVisible(false);
        }
    });
}

class $modify(MusicUnlocker, PlayLayer) {
    void toggleMusicInPractice() {
        m_practiceMusicSync = true;
        PlayLayer::toggleMusicInPractice();
    }
};

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