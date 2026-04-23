// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            Blend OneMinusDstAlpha One
            Cull Off
            
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc
#pragma exclude_renderers webgpu

// Stereo defines come from UnityShaderVariables / engine (STEREO_*_ON); multi_compile UNITY_STEREO_* conflicts with UnityInput.hlsl.

// UnityShaderVariables + UnityInstancing before UnityCG so unity_StereoEyeIndex is not gl_ViewID when UnityInstancing is parsed.
#include "GaussianSplatCGStereoBootstrap.hlsl"
#include "UnityCG.cginc"
#include "GaussianSplatting.hlsl"

StructuredBuffer<uint> _OrderBuffer;

struct v2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float4 vertex : SV_POSITION;
    UNITY_VERTEX_OUTPUT_STEREO
};

StructuredBuffer<SplatViewData> _SplatViewData;
ByteAddressBuffer _SplatSelectedBits;
uint _SplatBitsValid;
uint _SplatViewEyeStride;
// Same pixel size as CSCalcViewData (_VecScreenParams). Do not use built-in _ScreenParams here — it can disagree under XR/stereo and breaks covariance vs quad scale.
float4 _VecScreenParams;
int _ForcedEyeIndex;

// In single-pass XR, each eye needs its own precomputed center/axes row from CSCalcViewData.
// Mixing a per-eye UNITY_MATRIX_VP center with a shared view row causes left/right desync.
v2f vertCore(uint vtxID, uint splatOrderIndex, uint eyeIdxForBuffer)
{
    v2f o = (v2f)0;
#if defined(UNITY_STEREO_MULTIVIEW_ENABLED) && defined(UNITY_STEREO_INSTANCING_ENABLED)
    // Our bootstrap turns unity_StereoEyeIndex into a writable local in the combined instancing+multiview case.
    // Keep Unity's stereo output tags aligned with the eye row we sampled from _SplatViewData.
    unity_StereoEyeIndex = eyeIdxForBuffer;
#endif
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    uint splatIdx = _OrderBuffer[splatOrderIndex];
    uint viewIdx = splatIdx + eyeIdxForBuffer * _SplatViewEyeStride;

    SplatViewData view = _SplatViewData[viewIdx];
    float4 centerClipPos = view.pos;
    bool behindCam = centerClipPos.w <= 0.0;
    if (behindCam)
    {
        o.vertex = asfloat(0x7fc00000);
    }
    else
    {
        o.col.r = f16tof32(view.color.x >> 16);
        o.col.g = f16tof32(view.color.x);
        o.col.b = f16tof32(view.color.y >> 16);
        o.col.a = f16tof32(view.color.y);

        uint idx = vtxID;
        float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
        quadPos *= 2;

        o.pos = quadPos;

        float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2 / max(_VecScreenParams.xy, 1);
        o.vertex = centerClipPos;
        o.vertex.xy += deltaScreenPos * centerClipPos.w;

        if (_SplatBitsValid)
        {
            uint wordIdx = splatIdx / 32;
            uint bitIdx = splatIdx & 31;
            uint selVal = _SplatSelectedBits.Load(wordIdx * 4);
            if (selVal & (1 << bitIdx))
            {
                o.col.a = -1;
            }
        }
    }
    FlipProjectionIfBackbuffer(o.vertex);
    return o;
}

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    uint splatLookupIndex = instID;
    uint eyeIdx = 0u;
#if defined(UNITY_STEREO_MULTIVIEW_ENABLED)
    #if defined(SHADER_API_VULKAN) || defined(SHADER_API_GLES3) || defined(SHADER_API_GLCORE)
    eyeIdx = uint(gl_ViewID);
    #else
    eyeIdx = (uint)unity_StereoEyeIndex;
    #endif
#elif defined(UNITY_STEREO_INSTANCING_ENABLED)
    UnitySetupInstanceID(instID);
    splatLookupIndex = unity_InstanceID;
    eyeIdx = (uint)unity_StereoEyeIndex;
#endif
    if (_ForcedEyeIndex >= 0)
        eyeIdx = (uint)_ForcedEyeIndex;
    return vertCore(vtxID, splatLookupIndex, eyeIdx);
}

half4 frag (v2f i) : SV_Target
{
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
	float power = -dot(i.pos, i.pos);
	half alpha = exp(power);
	if (i.col.a >= 0)
	{
		alpha = saturate(alpha * i.col.a);
	}
	else
	{
		// "selected" splat: magenta outline, increase opacity, magenta tint
		half3 selectedColor = half3(1,0,1);
		if (alpha > 7.0/255.0)
		{
			if (alpha < 10.0/255.0)
			{
				alpha = 1;
				i.col.rgb = selectedColor;
			}
			alpha = saturate(alpha + 0.3);
		}
		i.col.rgb = lerp(i.col.rgb, selectedColor, 0.5);
	}
	
    if (alpha < 1.0/255.0)
        discard;

    half4 res = half4(i.col.rgb * alpha, alpha);
    return res;
}
ENDCG
        }
    }
}
