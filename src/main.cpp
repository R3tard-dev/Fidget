#include <windows.h>
#include <gl/GL.h>

#include "MinHook.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"

// ---------------------------------------------------------------------------
// Mod menu state
// ---------------------------------------------------------------------------
static bool g_menuOpen = false;
static bool g_initialized = false;
static HWND g_hwnd = nullptr;
static WNDPROC g_originalWndProc = nullptr;

// Example toggles
static bool g_godMode = false;
static bool g_speedHack = false;
static float g_speedMultiplier = 1.0f;
static bool g_noclip = false;
static bool g_autoClicker = false;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ---------------------------------------------------------------------------
// Window procedure hook (handles Tab toggle + ImGui input)
// ---------------------------------------------------------------------------
LRESULT CALLBACK WndProcHook(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (g_initialized && ImGui::GetCurrentContext()) {
        if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam))
            return 1;
    }

    if (msg == WM_KEYDOWN && wParam == VK_TAB) {
        g_menuOpen = !g_menuOpen;
        return 0;
    }

    if (g_originalWndProc)
        return CallWindowProc(g_originalWndProc, hwnd, msg, wParam, lParam);
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// ---------------------------------------------------------------------------
// wglSwapBuffers hook (rendering)
// ---------------------------------------------------------------------------
typedef BOOL (WINAPI *wglSwapBuffers_t)(HDC hdc);
static wglSwapBuffers_t oSwapBuffers = nullptr;

BOOL WINAPI hkSwapBuffers(HDC hdc) {
    if (!g_initialized) {
        g_hwnd = WindowFromDC(hdc);
        if (g_hwnd) {
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            ImGui::StyleColorsDark();

            ImGui_ImplWin32_Init(g_hwnd);
            ImGui_ImplOpenGL3_Init(nullptr);

            g_originalWndProc = (WNDPROC)SetWindowLongPtr(g_hwnd, GWLP_WNDPROC, (LONG_PTR)WndProcHook);
            g_initialized = true;
        }
    }

    if (g_initialized) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        if (g_menuOpen) {
            ImGui::SetNextWindowSize(ImVec2(420, 360), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("GD Mod Menu", &g_menuOpen)) {
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
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    return oSwapBuffers ? oSwapBuffers(hdc) : TRUE;
}

// ---------------------------------------------------------------------------
// DLL entry
// ---------------------------------------------------------------------------
static DWORD WINAPI InitThread(LPVOID) {
    if (MH_Initialize() != MH_OK)
        return 1;

    HMODULE hOpenGL = GetModuleHandleA("opengl32.dll");
    if (!hOpenGL)
        hOpenGL = LoadLibraryA("opengl32.dll");

    if (hOpenGL) {
        void *target = (void *)GetProcAddress(hOpenGL, "wglSwapBuffers");
        if (target) {
            MH_STATUS status = MH_CreateHook(target, &hkSwapBuffers, (LPVOID *)&oSwapBuffers);
            if (status == MH_OK) {
                MH_EnableHook(target);
            }
        }
    }
    return 0;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hModule);
        CreateThread(nullptr, 0, InitThread, nullptr, 0, nullptr);
        break;
    case DLL_PROCESS_DETACH:
        if (g_originalWndProc && g_hwnd)
            SetWindowLongPtr(g_hwnd, GWLP_WNDPROC, (LONG_PTR)g_originalWndProc);
        MH_DisableHook(MH_ALL_HOOKS);
        MH_Uninitialize();
        break;
    }
    return TRUE;
}