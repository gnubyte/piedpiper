# Shrinker rules shipped inside the AAR (applied to every app that uses the library).
# libpiedpiper_jni binds and calls these by name through JNI.

# Native methods: JNI symbol names are derived from the class and method names.
-keepclasseswithmembernames,includedescriptorclasses class io.piedpiper.PiedPiper {
    native <methods>;
}

# Thrown from native code.
-keep class io.piedpiper.PiedPiperException {
    <init>(java.lang.String, int);
}
-keep class io.piedpiper.PiedPiperCancelledException {
    <init>(java.lang.String);
}

# Called from native code (looked up by name and signature on the listener's class).
-keep interface io.piedpiper.ProgressListener {
    boolean onProgress(long, long);
}
