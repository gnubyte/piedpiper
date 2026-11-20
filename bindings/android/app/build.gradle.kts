import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "io.piedpiper.demo"
    compileSdk = 37

    defaultConfig {
        applicationId = "io.piedpiper.demo"
        minSdk = 24
        targetSdk = 37
        versionCode = 2
        versionName = "0.1.2"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
            // Demo only: sign release builds with the debug key so they can be installed directly.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // Plain framework widgets only: the demo has no other dependencies.
    implementation(project(":piedpiper"))
}
