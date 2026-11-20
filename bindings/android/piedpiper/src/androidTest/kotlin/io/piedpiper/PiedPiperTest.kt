package io.piedpiper

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Random
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/** API behaviour of the Kotlin bindings: roundtrips, progress, cancellation, errors. */
@RunWith(AndroidJUnit4::class)
class PiedPiperTest {

    private val text = "PiedPiper compresses text with a transformer that learns as it goes. ".repeat(40).encodeToByteArray()

    private fun randomBytes(n: Int, seed: Long) = ByteArray(n).also { Random(seed).nextBytes(it) }

    @Test
    fun versionIsReported() {
        assertTrue(PiedPiper.version.matches(Regex("\\d+\\.\\d+\\.\\d+.*")))
        assertEquals(1, PiedPiper.formatVersion)
    }

    @Test
    fun randomDataRoundtripsAtLevel1() {
        val data = randomBytes(6000, 42)
        // allowStored = false forces the neural path (random data would otherwise be stored).
        val packed = PiedPiper.compress(data, CompressOptions(level = 1, allowStored = false))
        val info = PiedPiper.info(packed)
        assertFalse(info.stored)
        assertEquals(64, info.model!!.dModel)
        assertEquals(data.size.toLong(), info.originalSize)
        assertArrayEquals(data, PiedPiper.decompress(packed))

        // With the default options incompressible data falls back to a stored copy.
        val stored = PiedPiper.compress(data, CompressOptions(level = 1))
        assertTrue(PiedPiper.info(stored).stored)
        assertArrayEquals(data, PiedPiper.decompress(stored))
    }

    @Test
    fun textRoundtripReportsProgressFromAWorkerThread() {
        val caller = Thread.currentThread()
        val calls = mutableListOf<Pair<Long, Long>>()
        var listenerThread: Thread? = null
        val packed = PiedPiper.compress(text, CompressOptions(level = 1)) { done, total ->
            synchronized(calls) {
                calls += done to total
                listenerThread = Thread.currentThread()
            }
            true
        }
        assertTrue("ratio ${text.size}/${packed.size}", packed.size < text.size / 2)
        synchronized(calls) {
            assertTrue(calls.isNotEmpty())
            assertTrue(calls.all { it.second == text.size.toLong() })
            assertTrue(calls.zipWithNext().all { (a, b) -> a.first <= b.first })
            assertEquals(text.size.toLong(), calls.last().first)
            assertNotSame(caller, listenerThread)
        }

        val last = AtomicLong(-1)
        val out = PiedPiper.decompress(packed) { done, total ->
            assertEquals(text.size.toLong(), total)
            last.set(done)
            true
        }
        assertArrayEquals(text, out)
        assertEquals(text.size.toLong(), last.get())
    }

    @Test
    fun listenerReturningFalseCancels() {
        val calls = AtomicInteger()
        assertThrows(PiedPiperCancelledException::class.java) {
            PiedPiper.compress(text, CompressOptions(level = 1)) { _, _ -> calls.incrementAndGet() < 3 }
        }
        assertEquals(3, calls.get())

        val packed = PiedPiper.compress(text, CompressOptions(level = 1))
        assertThrows(PiedPiperCancelledException::class.java) {
            PiedPiper.decompress(packed) { _, _ -> false }
        }
    }

    @Test
    fun listenerExceptionIsRethrown() {
        val boom = IllegalStateException("boom")
        val thrown = assertThrows(IllegalStateException::class.java) {
            PiedPiper.compress(text, CompressOptions(level = 1)) { _, _ -> throw boom }
        }
        assertSame(boom, thrown)
        // The library is still usable afterwards (no exception left pending on a worker thread).
        assertArrayEquals(text, PiedPiper.decompress(PiedPiper.compress(text, CompressOptions(level = 1))))
    }

    @Test
    fun suspendingVariantsRoundtrip() = runBlocking {
        val packed = PiedPiper.compressSuspend(text, CompressOptions(level = 1))
        assertArrayEquals(text, PiedPiper.decompressSuspend(packed))
    }

    @Test
    fun cancellingTheCoroutineStopsTheCodec() = runBlocking {
        val big = "Cancellation must reach the native codec quickly. 0123456789 ".repeat(4000).encodeToByteArray()
        val started = CompletableDeferred<Unit>()
        val lastDone = AtomicLong()
        val job = launch(Dispatchers.Default) {
            PiedPiper.compressSuspend(big, CompressOptions(level = 1)) { done, _ ->
                lastDone.set(done)
                started.complete(Unit)
                true
            }
        }
        started.await()
        job.cancelAndJoin()
        assertTrue(job.isCancelled)
        assertTrue("stopped at ${lastDone.get()} of ${big.size}", lastDone.get() < big.size)
    }

    @Test
    fun corruptInputIsRejected() {
        val notAStream = assertThrows(PiedPiperException::class.java) { PiedPiper.decompress("nope".encodeToByteArray()) }
        assertEquals(PiedPiperException.Kind.FORMAT, notAStream.kind)
        assertEquals(PiedPiperException.Kind.FORMAT, assertThrows(PiedPiperException::class.java) { PiedPiper.info(byteArrayOf(1, 2, 3)) }.kind)

        val packed = PiedPiper.compress(text, CompressOptions(level = 1, allowStored = false))
        val k = packed.size / 2 + 20
        val corrupt = packed.copyOf().also { it[k] = (it[k].toInt() xor 0x40).toByte() }
        val e = assertThrows(PiedPiperException::class.java) { PiedPiper.decompress(corrupt) }
        assertTrue("${e.kind}: ${e.message}", e.kind == PiedPiperException.Kind.CHECKSUM || e.kind == PiedPiperException.Kind.FORMAT)

        val truncated = assertThrows(PiedPiperException::class.java) { PiedPiper.decompress(packed.copyOf(10)) }
        assertEquals(PiedPiperException.Kind.FORMAT, truncated.kind)
    }

    @Test
    fun invalidOptionsAreRejected() {
        fun kindOf(opts: CompressOptions) = assertThrows(PiedPiperException::class.java) { PiedPiper.compress(text, opts) }.kind
        assertEquals(PiedPiperException.Kind.OPTIONS, kindOf(CompressOptions(level = 0)))
        assertEquals(PiedPiperException.Kind.OPTIONS, kindOf(CompressOptions(level = 10)))
        assertEquals(PiedPiperException.Kind.OPTIONS, kindOf(CompressOptions(threads = -1)))
        // d_model 64 (level 1) is not a multiple of 3 heads.
        assertEquals(PiedPiperException.Kind.OPTIONS, kindOf(CompressOptions(level = 1, model = ModelOverrides(heads = 3))))
        assertEquals(PiedPiperException.Kind.OPTIONS, kindOf(CompressOptions(level = 1, train = TrainOverrides(warmup = 0))))
        assertEquals(PiedPiperException.Kind.OPTIONS, kindOf(CompressOptions(level = 1, train = TrainOverrides(beta1 = 1.5f))))
        assertEquals(PiedPiperException.Kind.OPTIONS, kindOf(CompressOptions(level = 1, model = ModelOverrides(dModel = -8))))
        assertEquals(PiedPiperException.Kind.OPTIONS, assertThrows(PiedPiperException::class.java) { PiedPiper.preset(0) }.kind)
    }

    @Test
    fun threadCountDoesNotChangeOutput() {
        val one = PiedPiper.compress(text, CompressOptions(level = 1, threads = 1))
        val four = PiedPiper.compress(text, CompressOptions(level = 1, threads = 4))
        val all = PiedPiper.compress(text, CompressOptions(level = 1))
        assertArrayEquals(one, four)
        assertArrayEquals(one, all)
    }

    @Test
    fun presetsComeFromTheCore() {
        val p = PiedPiper.preset(3)
        assertEquals(3, p.level)
        assertEquals(ModelConfig(dModel = 128, layers = 4, heads = 4, dFf = 352, window = 128, segLen = 16, streams = 32), p.model)
        assertEquals(1.0e-3f.toRawBits(), p.train.lr.toRawBits())
        assertEquals((1.0e-3f * 0.25f).toRawBits(), p.train.lrMin.toRawBits())
        assertEquals(0x5049_4544_5049_5045L, p.train.seed)
        assertEquals(4096, p.maxVocab)
        assertEquals(64, PiedPiper.preset(1).model.dModel)
    }

    @Test
    fun headerSettingsReproduceTheStream() {
        // Network + training recipe read back from a header (exact floats through JSON) and
        // passed in as overrides re-create the same stream, whatever the level.
        // No built-in prior: a prior is not part of the header settings being reproduced.
        val opts = CompressOptions(level = 1, tokenizer = Tokenizer.Bytes, allowStored = false, builtinPrior = false)
        val a = PiedPiper.compress(text, opts)
        val info = PiedPiper.info(a)
        val b = PiedPiper.compress(text, opts.copy(level = 7, model = info.model!!.toOverrides(), train = info.train!!.toOverrides()))
        assertArrayEquals(a, b)
    }

    @Test
    fun bpeTokenizerRoundtrips() {
        val words = (0 until 3000).joinToString(" ") { listOf("alpha", "beta", "gamma", "delta", "pied", "piper")[(it * 7 + it / 3) % 6] }.encodeToByteArray()
        val packed = PiedPiper.compress(words, CompressOptions(level = 1, tokenizer = Tokenizer.Bpe(300)))
        assertTrue(PiedPiper.info(packed).tokenizer.startsWith("bpe-"))
        assertArrayEquals(words, PiedPiper.decompress(packed))
    }

    @Test
    fun emptyInputRoundtrips() {
        val packed = PiedPiper.compress(ByteArray(0))
        assertTrue(PiedPiper.isCompressed(packed))
        assertTrue(PiedPiper.info(packed).stored)
        assertEquals(0, PiedPiper.decompress(packed).size)
        assertFalse(PiedPiper.isCompressed(text))
    }
}
