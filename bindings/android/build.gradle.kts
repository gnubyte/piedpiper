// Plugin versions live in gradle/libs.versions.toml. AGP 9 compiles Kotlin itself ("built-in
// Kotlin"), so the org.jetbrains.kotlin.android plugin is deliberately not applied anywhere.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
}
