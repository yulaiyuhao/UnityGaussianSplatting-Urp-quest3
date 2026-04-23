// SPDX-License-Identifier: MIT
// Built-in CGPROGRAM: same ordering fix as GaussianSplatStereoBootstrap.hlsl — variables + instancing before UnityCG.cginc.
#ifndef GAUSSIAN_SPLAT_CG_STEREO_BOOTSTRAP_INCLUDED
#define GAUSSIAN_SPLAT_CG_STEREO_BOOTSTRAP_INCLUDED

#include "UnityShaderUtilities.cginc"
#include "UnityShaderVariables.cginc"

#if defined(UNITY_STEREO_INSTANCING_ENABLED) && defined(UNITY_STEREO_MULTIVIEW_ENABLED)
#if !defined(SHADER_STAGE_FRAGMENT)
#undef unity_StereoEyeIndex
static uint unity_StereoEyeIndex;
#endif
#endif

#include "UnityInstancing.cginc"

#endif
