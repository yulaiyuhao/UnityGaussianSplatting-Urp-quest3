// SPDX-License-Identifier: MIT
// URP: include UnityInput, then break unity_StereoEyeIndex -> gl_ViewID BEFORE UnityInstancing.hlsl is parsed.
// Core.hlsl pulls Input.hlsl which includes UnityInput then UnityInstancing; UnitySetupInstanceID assigns unity_StereoEyeIndex,
// which expands to gl_ViewID = ... when both stereo instancing and multiview are active (illegal on Vulkan/D3D/GLES).
#ifndef GAUSSIAN_SPLAT_STEREO_BOOTSTRAP_INCLUDED
#define GAUSSIAN_SPLAT_STEREO_BOOTSTRAP_INCLUDED

// UnityInput expects Core's Common types/macros (same order as Core.hlsl before Input.hlsl).
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/UnityInput.hlsl"

#if defined(UNITY_STEREO_INSTANCING_ENABLED) && defined(UNITY_STEREO_MULTIVIEW_ENABLED)
#if defined(SHADER_STAGE_VERTEX) || !defined(SHADER_STAGE_FRAGMENT)
#undef unity_StereoEyeIndex
static uint unity_StereoEyeIndex;
#endif
#endif

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"

#endif
