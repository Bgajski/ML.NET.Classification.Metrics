using System.Runtime.InteropServices;

namespace ML.NET.Metrics.Evaluation.Internal
{
    /// <summary>
    //// custom DLL import resolver for the native "coremetrics" library
    /// </summary>
    internal static class NativeResolver
    {
        private static bool _initialized;

        internal static void EnsureRegistered()
        {
            if (_initialized) return;
            _initialized = true;

            NativeLibrary.SetDllImportResolver(typeof(NativeResolver).Assembly, Resolve);
        }

        private static IntPtr Resolve(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, "coremetrics", StringComparison.OrdinalIgnoreCase))
                return IntPtr.Zero;

            string fileName =
                RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "coremetrics.dll" :
                RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "libcoremetrics.so" :
                RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "libcoremetrics.dylib" :
                "coremetrics";

            // let the runtime resolve it
            if (NativeLibrary.TryLoad(fileName, assembly, searchPath, out var handle))
                return handle;

            // fallback AppContext.BaseDirectory
            string candidate = Path.Combine(AppContext.BaseDirectory, fileName);
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out handle))
                return handle;

            return IntPtr.Zero;
        }
    }
}
