// SPDX-License-Identifier: MIT
#if GS_ENABLE_URP

#if !UNITY_6000_0_OR_NEWER
#error Unity Gaussian Splatting URP support only works in Unity 6 or later
#endif

using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;

namespace GaussianSplatting.Runtime
{
    // Note: I have no idea what is the purpose of ScriptableRendererFeature vs ScriptableRenderPass, which one of those
    // is supposed to do resource management vs logic, etc. etc. Code below "seems to work" but I'm just fumbling along,
    // without understanding any of it.
    //
    // ReSharper disable once InconsistentNaming
    class GaussianSplatURPFeature : ScriptableRendererFeature
    {
        class GSRenderPass : ScriptableRenderPass
        {
            const string GaussianSplatRTName = "_GaussianSplatRT";

            const string ProfilerTag = "GaussianSplatRenderGraph";
            static readonly ProfilingSampler s_profilingSampler = new(ProfilerTag);
            static readonly int s_gaussianSplatRT = Shader.PropertyToID(GaussianSplatRTName);

            class PassData
            {
                internal UniversalCameraData CameraData;
                internal TextureHandle SourceTexture;
                internal TextureHandle SourceDepth;
                internal TextureHandle GaussianSplatRT;
                internal bool UrpGpuProjectionRenderIntoTexture;
                internal bool DirectToCameraTarget;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();

                bool directToCameraTarget = cameraData.xr != null && cameraData.xr.enabled;
                TextureHandle textureHandle = TextureHandle.nullHandle;
                if (!directToCameraTarget)
                {
                    RenderTextureDescriptor rtDesc = cameraData.cameraTargetDescriptor;
                    rtDesc.depthBufferBits = 0;
                    rtDesc.msaaSamples = 1;
                    rtDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
                    textureHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, rtDesc, GaussianSplatRTName, true);
                }

                passData.CameraData = cameraData;
                passData.SourceTexture = resourceData.activeColorTexture;
                passData.SourceDepth = resourceData.activeDepthTexture;
                passData.GaussianSplatRT = textureHandle;
                passData.UrpGpuProjectionRenderIntoTexture =
                    GaussianSplatRenderer.ComputeUrpGpuProjectionRenderIntoTexture(cameraData, resourceData.isActiveTargetBackBuffer);
                passData.DirectToCameraTarget = directToCameraTarget;

                builder.UseTexture(resourceData.activeColorTexture, AccessFlags.ReadWrite);
                builder.UseTexture(resourceData.activeDepthTexture);
                if (!directToCameraTarget)
                    builder.UseTexture(textureHandle, AccessFlags.Write);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc(static (PassData data, UnsafeGraphContext context) =>
                {
                    var commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                    using var _ = new ProfilingScope(commandBuffer, s_profilingSampler);
                    // Required for FlipProjectionIfBackbuffer in RenderGaussianSplats.shader (same as BiRP path).
                    // If unset, _CameraTargetTexture_TexelSize defaults to (1,1,1,1) and splats get an incorrect Y flip vs depth,
                    // which looks like content "sticking" to / rotating with the camera when you orbit (especially on Y).
                    commandBuffer.SetGlobalTexture(GaussianSplatRenderer.Props.CameraTargetTexture, data.SourceTexture);
                    // Game view / RenderGraph: binding SourceTexture alone can leave _CameraTargetTexture_TexelSize.z == 1,
                    // which mis-triggers the "backbuffer" branch and flips splats upside-down; Scene view often differs.
                    var camData = data.CameraData;
                    int tw = camData.cameraTargetDescriptor.width > 0 ? camData.cameraTargetDescriptor.width : camData.scaledWidth;
                    int th = camData.cameraTargetDescriptor.height > 0 ? camData.cameraTargetDescriptor.height : camData.scaledHeight;
                    tw = Mathf.Max(1, tw);
                    th = Mathf.Max(1, th);
                    GaussianSplatRenderer.LogXrPassDiagnostics(camData, data.DirectToCameraTarget, data.UrpGpuProjectionRenderIntoTexture);
                    commandBuffer.SetGlobalVector(GaussianSplatRenderer.Props.CameraTargetTextureTexelSize,
                        new Vector4(1f / tw, 1f / th, tw, th));
                    if (data.DirectToCameraTarget)
                    {
                        // XR path: render splats directly to camera target to avoid an extra fullscreen blit and reduce
                        // eye-texture array handling issues that can look like head-locked content.
                        var desc = data.CameraData.cameraTargetDescriptor;
                        bool explicitPerEyeDraw =
                            desc.dimension == TextureDimension.Tex2DArray &&
                            desc.volumeDepth > 1 &&
                            GaussianSplatRenderSystem.instance.NeedsExplicitPerEyeDraw();
                        if (explicitPerEyeDraw)
                        {
                            for (int eye = 0; eye < Mathf.Min(2, desc.volumeDepth); ++eye)
                            {
                                CoreUtils.SetRenderTarget(commandBuffer, data.SourceTexture, data.SourceDepth, ClearFlag.None, Color.clear, 0, CubemapFace.Unknown, eye);
                                GaussianSplatRenderSystem.instance.SortAndRenderSplats(
                                    data.CameraData.camera,
                                    commandBuffer,
                                    data.CameraData,
                                    data.UrpGpuProjectionRenderIntoTexture,
                                    eye);
                            }
                        }
                        else
                        {
                            CoreUtils.SetRenderTarget(commandBuffer, data.SourceTexture, data.SourceDepth, ClearFlag.None, Color.clear);
                            GaussianSplatRenderSystem.instance.SortAndRenderSplats(data.CameraData.camera, commandBuffer, data.CameraData, data.UrpGpuProjectionRenderIntoTexture);
                        }
                    }
                    else
                    {
                        commandBuffer.SetGlobalTexture(s_gaussianSplatRT, data.GaussianSplatRT);
                        CoreUtils.SetRenderTarget(commandBuffer, data.GaussianSplatRT, data.SourceDepth, ClearFlag.Color, Color.clear);
                        Material matComposite = GaussianSplatRenderSystem.instance.SortAndRenderSplats(data.CameraData.camera, commandBuffer, data.CameraData, data.UrpGpuProjectionRenderIntoTexture);
                        commandBuffer.BeginSample(GaussianSplatRenderSystem.s_ProfCompose);
                        Blitter.BlitCameraTexture(commandBuffer, data.GaussianSplatRT, data.SourceTexture, matComposite, 0);
                        commandBuffer.EndSample(GaussianSplatRenderSystem.s_ProfCompose);
                    }
                });
            }
        }

        GSRenderPass m_Pass;

        public override void Create()
        {
            m_Pass = new GSRenderPass
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents
            };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            // Do not use OnCameraPreCull + a shared bool: with Scene + Game cameras, PreCull order can leave
            // m_HasCamera false when AddRenderPasses runs for the Game camera, so splats never render in Game view.
            if (!GaussianSplatRenderSystem.instance.GatherSplatsForCamera(renderingData.cameraData.camera))
                return;
            renderer.EnqueuePass(m_Pass);
        }

        protected override void Dispose(bool disposing)
        {
            m_Pass = null;
        }
    }
}

#endif // #if GS_ENABLE_URP
