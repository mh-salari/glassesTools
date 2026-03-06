from imgui_bundle import imgui

error = imgui.ImColor.hsv(0.9667, 0.88, 0.64).value
error_bright = imgui.ImColor.hsv(0.9667, 0.88, 0.93).value
error_dark = imgui.ImColor.hsv(0.9667, 0.88, 0.43).value

warning = imgui.ImColor.hsv(45 / 360, 0.97, 1.0).value
warning_bright = imgui.ImColor.hsv(54 / 360, 0.97, 1.0).value
warning_dark = imgui.ImColor.hsv(45 / 360, 0.97, 0.8).value

ok = imgui.ImVec4(0.0000, 0.8500, 0.0000, 1.0)

gray = imgui.ImVec4(0.5000, 0.5000, 0.5000, 1.0)
