package io.piedpiper

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized

/**
 * Cross-platform determinism proof: the golden vectors in testdata/golden were produced by the
 * desktop CLI on Windows x86_64. On this device (any ABI) every `.bin.pp` must decompress to its
 * `.bin` bit-exactly, and compressing the `.bin` with the manifest's flags must reproduce the
 * `.bin.pp` byte for byte. Any floating-point difference anywhere in the model would break both.
 */
@RunWith(Parameterized::class)
class GoldenVectorsTest(private val vector: GoldenVector) {

    companion object {
        private const val TAG = "PiedPiperGolden"

        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun vectors(): List<Array<Any>> = Golden.vectors().map { arrayOf<Any>(it) }
    }

    private val input by lazy { Golden.read(vector.input) }
    private val compressed by lazy { Golden.read(vector.compressed) }
    // ABI of the loaded libpiedpiper_jni.so (e.g. "arm64" when an x86_64 emulator runs the
    // arm64-v8a library through ARM translation), and the device's primary ABI.
    private val abi: String =
        InstrumentationRegistry.getInstrumentation().context.applicationInfo.nativeLibraryDir.substringAfterLast('/') +
            " on " + (Build.SUPPORTED_ABIS.firstOrNull() ?: "?")

    @Test
    fun filesMatchManifest() {
        assertEquals("${vector.input} sha256", vector.inputSha256, Golden.sha256(input))
        assertEquals("${vector.compressed} sha256", vector.compressedSha256, Golden.sha256(compressed))
        assertEquals(vector.inputSize, input.size)
        assertEquals(vector.compressedSize, compressed.size)
    }

    @Test
    fun decompressReproducesInput() {
        val t0 = SystemClock.elapsedRealtime()
        val out = PiedPiper.decompress(compressed)
        val ms = SystemClock.elapsedRealtime() - t0
        Log.i(TAG, "[$abi] ${vector.name}: decompressed ${compressed.size} -> ${out.size} bytes in $ms ms, sha256 ${Golden.sha256(out)}")
        assertEquals("decompressed size", vector.inputSize, out.size)
        assertEquals("decompressed sha256", vector.inputSha256, Golden.sha256(out))
        assertArrayEquals(input, out)
    }

    @Test
    fun compressReproducesGoldenBytes() {
        val opts = Golden.optionsFromCliFlags(vector.flags)
        val t0 = SystemClock.elapsedRealtime()
        val out = PiedPiper.compress(input, opts)
        val ms = SystemClock.elapsedRealtime() - t0
        val sha = Golden.sha256(out)
        Log.i(TAG, "[$abi] ${vector.name}: compressed ${input.size} -> ${out.size} bytes in $ms ms (flags: ${vector.flags}), sha256 $sha, identical=${out.contentEquals(compressed)}")
        if (!out.contentEquals(compressed)) {
            val first = (0 until maxOf(out.size, compressed.size)).first { it >= out.size || it >= compressed.size || out[it] != compressed[it] }
            fail("compressed bytes differ from ${vector.compressed}: ${out.size} vs ${compressed.size} bytes, first difference at offset $first")
        }
        assertEquals("compressed sha256", vector.compressedSha256, sha)
    }

    @Test
    fun headerMatchesFlags() {
        val opts = Golden.optionsFromCliFlags(vector.flags)
        val info = PiedPiper.info(compressed)
        assertFalse(info.stored)
        assertEquals(vector.inputSize.toLong(), info.originalSize)
        assertEquals(vector.compressedSize.toLong(), info.compressedSize)
        // Container modes (video, JPEG, audio) carry no network header of their own.
        if (info.model == null) return
        if (opts.model == null) {
            val p = PiedPiper.preset(opts.level)
            assertEquals(p.model.dModel, info.model!!.dModel)
            assertEquals(p.model.layers, info.model!!.layers)
            if (vector.name.startsWith("text-prior")) assertNotEquals(0L, info.priorId)
            if (info.priorId != 0L) {
                // Default settings on text: the level's built-in prior, fine-tuned at prior_lr, or
                // at half of it when the encoder's prefix trial found that compresses better.
                // The level's prior or one of its alternates ("text-s-code", ...).
                val base = listOf("text-s", "text-m", "text-l")[opts.level - 1]
                assertTrue("prior ${info.priorName}", info.priorName == base || info.priorName!!.startsWith("$base-"))
                val lr = info.train!!.lr
                assertTrue("lr $lr", lr == p.priorLr || lr == p.priorLr * 0.5f)
                assertEquals(lr * 0.25f, info.train!!.lrMin)
            } else {
                // Binary input (executables, filtered pixels) trains from scratch.
                assertEquals(p.train.lr, info.train!!.lr)
            }
            return
        }
        assertEquals(0L, info.priorId)
        // Every explicit golden pins the whole network, so the header holds exactly the flag values.
        assertEquals(opts.model, info.model!!.toOverrides())
        // The training recipe: preset(level).train with the flag overrides (bit-exact floats).
        assertEquals(Golden.expectedTrain(opts), info.train)
        val tok = opts.tokenizer
        assertEquals(if (tok is Tokenizer.Bpe) "bpe" else "bytes", info.tokenizer.substringBefore('-'))
    }
}
