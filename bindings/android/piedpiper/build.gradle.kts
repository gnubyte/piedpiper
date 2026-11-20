import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "io.piedpiper"
    compileSdk = 37

    defaultConfig {
        // Must match ANDROID_API (24) in build-native.sh, which links the .so against this level.
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("androidTest") {
            // The cross-platform golden vectors (testdata/golden: manifest.json, *.bin, *.bin.pp),
            // packaged as test assets straight from the repository; nothing is copied or moved.
            assets.directories.add(rootProject.layout.projectDirectory.dir("../../testdata/golden").asFile.path)
        }
    }

    testOptions {
        targetSdk = 37
    }

    lint {
        targetSdk = 37
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // Only for the suspending helpers (compressSuspend / decompressSuspend).
    implementation(libs.kotlinx.coroutines.core)

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.junit)
}

// The JNI libraries are build outputs of ../../crates/jni (./build-native.sh); fail early with a
// useful message instead of producing an AAR/APK that crashes with UnsatisfiedLinkError.
val jniLibsDir = layout.projectDirectory.dir("src/main/jniLibs")
val checkNativeLibs = tasks.register("checkNativeLibs") {
    val dir = jniLibsDir.asFile
    val abis = listOf("arm64-v8a", "armeabi-v7a", "x86_64")
    inputs.files(abis.map { File(dir, "$it/libpiedpiper_jni.so") }).optional()
    doLast {
        val missing = abis.filterNot { File(dir, "$it/libpiedpiper_jni.so").isFile }
        if (missing.size == abis.size) {
            throw GradleException(
                "No libpiedpiper_jni.so in $dir. Build the native library first: " +
                    "bindings/android/build-native.sh (or build-native.ps1).",
            )
        }
        if (missing.isNotEmpty()) {
            logger.warn("PiedPiper: libpiedpiper_jni.so missing for ${missing.joinToString()} (run build-native.sh)")
        }
    }
}
tasks.named("preBuild") { dependsOn(checkNativeLibs) }

// PiedPiper's MIT license and the third-party notices (Apache-2.0 attribution for the vendored
// lepton_jpeg and for preflate-rs, plus the license texts of every compiled-in crate) must travel
// with the binary. They are copied from the repository root into generated Java resources, so the
// AAR carries them in classes.jar under META-INF/piedpiper/ (same layout as the repository, which
// keeps the links in THIRD_PARTY_NOTICES.md valid) and apps built with it carry them in the APK.
// Keep noticePaths in step with bindings/sync_notices.py.
abstract class CopyLicenseNotices : DefaultTask() {
    @get:Internal
    abstract val repoRoot: DirectoryProperty

    @get:Input
    abstract val paths: ListProperty<String>

    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sources: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val root = repoRoot.get().asFile
        val dest = outputDir.get().asFile.resolve("META-INF/piedpiper")
        dest.deleteRecursively()
        for (path in paths.get()) {
            val src = root.resolve(path)
            if (!src.isFile) {
                throw GradleException("Missing $src (THIRD_PARTY_LICENSES.md is made by python scripts/gen_licenses.py)")
            }
            src.copyTo(dest.resolve(path))
        }
    }
}

val noticeRoot = rootProject.layout.projectDirectory.dir("../..")
val noticePaths = listOf(
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_LICENSES.md",
    "third_party/lepton_jpeg/LICENSE.txt",
    "third_party/lepton_jpeg/NOTICE.txt",
)
androidComponents {
    onVariants { variant ->
        val copyNotices = tasks.register<CopyLicenseNotices>(
            "copy${variant.name.replaceFirstChar { it.uppercase() }}LicenseNotices",
        ) {
            repoRoot.set(noticeRoot)
            paths.set(noticePaths)
            sources.from(noticePaths.map { noticeRoot.file(it) })
        }
        variant.sources.resources?.addGeneratedSourceDirectory(copyNotices, CopyLicenseNotices::outputDir)
    }
}
