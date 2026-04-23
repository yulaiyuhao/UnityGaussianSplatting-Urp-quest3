// SPDX-License-Identifier: MIT

using System;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Writes XR diagnostics to a persistent file so device builds can be inspected after the run.
    /// </summary>
    public static class GaussianSplatFileLogger
    {
        static readonly object s_Lock = new();
        static bool s_Initialized;
        static string s_LogPath;

        public static string currentSceneName
        {
            get
            {
                var scene = SceneManager.GetActiveScene();
                return string.IsNullOrEmpty(scene.name) ? "<no-scene>" : scene.name;
            }
        }

        public static string logPath
        {
            get
            {
                EnsureInitialized();
                return s_LogPath;
            }
        }

        static void EnsureInitialized()
        {
            if (s_Initialized)
                return;

            lock (s_Lock)
            {
                if (s_Initialized)
                    return;

                string root = Application.persistentDataPath;
                if (string.IsNullOrEmpty(root))
                    root = Application.temporaryCachePath;
                if (string.IsNullOrEmpty(root))
                    root = ".";

                s_LogPath = Path.Combine(root, "gaussian_splat_xr.log");
                Directory.CreateDirectory(Path.GetDirectoryName(s_LogPath) ?? ".");
                File.WriteAllText(
                    s_LogPath,
                    $"=== GaussianSplat XR log start {DateTime.Now:O} ==={Environment.NewLine}" +
                    $"scene={currentSceneName}{Environment.NewLine}" +
                    $"persistentDataPath={Application.persistentDataPath}{Environment.NewLine}" +
                    $"temporaryCachePath={Application.temporaryCachePath}{Environment.NewLine}" +
                    $"unityVersion={Application.unityVersion} platform={Application.platform}{Environment.NewLine}",
                    Encoding.UTF8);
                s_Initialized = true;
                Debug.Log($"[GaussianSplatXR] file logging -> {s_LogPath}");
            }
        }

        public static void AppendLine(string message)
        {
            try
            {
                EnsureInitialized();
                lock (s_Lock)
                {
                    File.AppendAllText(s_LogPath, message + Environment.NewLine, Encoding.UTF8);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GaussianSplatXR] failed to append log file: {ex.Message}");
            }
        }
    }
}
