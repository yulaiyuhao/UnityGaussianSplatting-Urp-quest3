// SPDX-License-Identifier: MIT
// Fullscreen composite: match URP Hidden/Universal Render Pipeline/Blit (no FlipProjectionIfBackbuffer).
// BiRP-style flip in the composite pass stacks badly with Blitter._BlitScaleBias and caused upside-down Game view for Splats.
// XR: SAMPLE_TEXTURE2D_X + stereo varyings (URP Core — do not include TextureXR.hlsl).

Shader "Hidden/Gaussian Splatting/Composite"
{
    Properties
    {
        [HideInInspector] _BlitScaleBias("Blit Scale Bias", Vector) = (1, 1, 0, 0)
    }

    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" }
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma require compute
            #pragma use_dxc
            // Do not use #pragma multi_compile for UNITY_STEREO_* — UnityInput.hlsl already defines them from STEREO_*_ON (avoids macro redefinition vs line 8/12).
            // WebGPU backend: XR stereo builtins can fail SPIR-V compile ("unknown builtin"); Quest/Android builds do not need this target.
            #pragma exclude_renderers webgpu

            // UnityInput must be included before UnityInstancing; when instancing + multiview both apply, fix unity_StereoEyeIndex
            // before UnityInstancing.hlsl is parsed (see GaussianSplatStereoBootstrap.hlsl). Core.hlsl would include them in the wrong order.
            #include "GaussianSplatStereoBootstrap.hlsl"

            // URP Core.hlsl already defines TEXTURE2D_X / SAMPLE_TEXTURE2D_X / SLICE_ARRAY_INDEX for stereo.
            // Do NOT include TextureXR.hlsl — it redeclares unity_StereoEyeIndex (already in UnityInput.hlsl via Core).
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/DynamicScaling.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"

            // HLSLSupport can redefine UNITY_STEREO_MULTIVIEW_ENABLED after Core; strip again for instancing+multiview combo.
#if defined(UNITY_STEREO_INSTANCING_ENABLED) && defined(UNITY_STEREO_MULTIVIEW_ENABLED)
            #undef UNITY_STEREO_MULTIVIEW_ENABLED
#endif

            TEXTURE2D_X(_BlitTexture);
            // Match Hidden/Universal Render Pipeline/Blit — Blitter binds _BlitTexture with this sampler name.
            SAMPLER(sampler_BlitTexture);

            float4 _BlitScaleBias;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 texcoord : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                // Same as URP Runtime/Utilities/Blit.hlsl Vert — Blitter expects this geometry + DYNAMIC_SCALING_APPLY_SCALEBIAS.
                float4 pos = GetFullScreenTriangleVertexPosition(input.vertexID);
                output.positionCS = pos;
                float2 uv = GetFullScreenTriangleTexCoord(input.vertexID);
                output.texcoord = DYNAMIC_SCALING_APPLY_SCALEBIAS(uv);
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                // Multiview uses BLENDWEIGHT0; stereo instancing uses SV_RenderTargetArrayIndex — macro picks the right field.
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                half4 col = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_BlitTexture, input.texcoord);
                float a = max(col.a, 1e-5);
                return half4(FastSRGBToLinear(col.rgb / a), col.a);
            }
            ENDHLSL
        }
    }
}
