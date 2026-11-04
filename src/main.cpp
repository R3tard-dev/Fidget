#include <Geode/Geode.hpp>
#include <Geode/modify/MenuLayer.hpp>

using namespace geode::prelude;

class $modify(RLMenuLayer, MenuLayer) {
    bool init() {
        if (!MenuLayer::init()) return false;
        log::info("[gdrl] mod loaded, MenuLayer initialised");
        return true;
    }
};
