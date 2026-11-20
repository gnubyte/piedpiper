pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "piedpiper-android"

// :piedpiper — the Android library (AAR): Kotlin API + libpiedpiper_jni.so (crates/jni).
// :app       — a small demo app that compresses / decompresses files picked with the SAF.
include(":piedpiper", ":app")
