package io.piedpiper

import org.json.JSONObject

/**
 * PiedPiper — neural lossless compression (the TufaZip method): an online-learning Transformer
 * predicts every next token and a range coder codes it with that prediction. Encoder and
 * decoder learn identical weights from the data itself, so nothing but the coded bits is stored.
 *
 * Output is **bit-identical on every platform**: a stream compressed on Android (ARM or x86),
 * a desktop or in a browser decompresses everywhere else, and the same input with the same
 * [CompressOptions] produces the same bytes everywhere.
 *
 * ### Threading
 * Every call blocks until the work is done, and the work is CPU-heavy (very roughly 5–50 KB/s,
 * depending on the level and the device): **never call these functions on the main thread.**
 * Use a background thread / executor, or the suspending variants [compressSuspend] and
 * [decompressSuspend]. The functions are thread-safe and may run concurrently. The computation
 * itself runs on PiedPiper's own worker threads (`threads` in the options, 0 = all cores), so
 * the calling thread mostly waits.
 *
 * ### Errors
 * Failures throw [PiedPiperException] (see [PiedPiperException.kind]); a cancelled operation
 * throws [PiedPiperCancelledException]; an exception thrown by a [ProgressListener] cancels the
 * operation and is re-thrown unchanged.
 *
 * ```
 * val packed = PiedPiper.compress(bytes, CompressOptions(level = 3)) { done, total ->
 *     post { bar.progress = (done * 100 / total).toInt() }
 *     !cancelled            // return false to cancel
 * }
 * val original = PiedPiper.decompress(packed)
 * ```
 */
object PiedPiper {
    /** Fastest level. */
    const val MIN_LEVEL: Int = 1

    /** Strongest (biggest network, slowest) level. */
    const val MAX_LEVEL: Int = 9

    /** Level used by [CompressOptions] unless told otherwise (the desktop CLI's default too). */
    const val DEFAULT_LEVEL: Int = 2

    private val MAGIC = byteArrayOf('P'.code.toByte(), 'I'.code.toByte(), 'E'.code.toByte(), 'D'.code.toByte())

    init {
        System.loadLibrary("piedpiper_jni")
    }

    /** Version of the native PiedPiper library, e.g. `0.1.0`. */
    @JvmStatic
    val version: String
        get() = nativeVersion()

    /** Stream format version this library writes (and the newest it reads). */
    @JvmStatic
    val formatVersion: Int
        get() = nativeFormatVersion()

    /**
     * Compresses [data].
     *
     * @param progress optional listener, called from a worker thread after every coded segment
     *   with `(input bytes processed, input size)`; return `false` to cancel.
     * @return the compressed stream (starts with the magic bytes `PIED`).
     * @throws PiedPiperException on invalid options (kind `OPTIONS`) or resource limits.
     * @throws PiedPiperCancelledException if [progress] returned `false`.
     */
    @JvmStatic
    @JvmOverloads
    fun compress(data: ByteArray, options: CompressOptions = CompressOptions(), progress: ProgressListener? = null): ByteArray {
        val tok = options.tokenizer
        val (tokKind, bpeVocab) = when (tok) {
            Tokenizer.Auto -> 0 to 0
            Tokenizer.Bytes -> 1 to 0
            is Tokenizer.Bpe -> 2 to tok.vocab
        }

        var modelMask = 0
        val model = IntArray(MODEL_FIELDS)
        options.model?.let { m ->
            listOf(m.dModel, m.layers, m.heads, m.dFf, m.window, m.segLen, m.streams).forEachIndexed { i, v ->
                if (v != null) {
                    modelMask = modelMask or (1 shl i)
                    model[i] = v
                }
            }
        }

        var trainMask = 0
        val trainF = FloatArray(TRAIN_F_FIELDS)
        val trainI = LongArray(TRAIN_I_FIELDS)
        options.train?.let { t ->
            listOf(t.lr, t.lrMin, t.beta1, t.beta2, t.eps, t.clip, t.initScale).forEachIndexed { i, v ->
                if (v != null) {
                    trainMask = trainMask or (1 shl i)
                    trainF[i] = v
                }
            }
            listOf(t.warmup, t.decay, t.seed).forEachIndexed { i, v ->
                if (v != null) {
                    trainMask = trainMask or (1 shl (TRAIN_I_SHIFT + i))
                    trainI[i] = v
                }
            }
        }

        return nativeCompress(
            data, options.level, tokKind, bpeVocab, options.threads, options.allowStored, options.builtinPrior, options.prior,
            modelMask, model, trainMask, trainF, trainI, progress,
        )
    }

    /**
     * Decompresses a PiedPiper stream. The result is verified against the CRC-32 stored in the
     * stream (kind `CHECKSUM` on mismatch).
     *
     * @param progress optional listener, called from a worker thread with
     *   `(output bytes produced, output size)`; return `false` to cancel.
     * @throws PiedPiperException if [data] is not a valid stream (kinds `FORMAT`, `VERSION`,
     *   `CHECKSUM`) or needs a prior that was not supplied (`PRIOR`).
     * @throws PiedPiperCancelledException if [progress] returned `false`.
     */
    @JvmStatic
    @JvmOverloads
    fun decompress(data: ByteArray, options: DecompressOptions = DecompressOptions(), progress: ProgressListener? = null): ByteArray =
        nativeDecompress(data, options.threads, options.prior, progress)

    /**
     * Reads the header of a compressed stream (sizes, tokenizer, network, training recipe)
     * without decompressing it. Fast.
     *
     * @throws PiedPiperException if [data] is not a PiedPiper stream.
     */
    @JvmStatic
    fun info(data: ByteArray): Info = Info.fromJson(JSONObject(nativeInfo(data)))

    /**
     * The network shape and training recipe of compression [level] (1..9) — the values that
     * [ModelOverrides] / [TrainOverrides] fields replace.
     */
    @JvmStatic
    fun preset(level: Int): Preset = Preset.fromJson(JSONObject(nativePreset(level)))

    /** True if [data] starts with the PiedPiper magic bytes (cheap; does not validate the stream). */
    @JvmStatic
    fun isCompressed(data: ByteArray): Boolean =
        data.size >= MAGIC.size && MAGIC.indices.all { data[it] == MAGIC[it] }

    // --- JNI (crates/jni) -----------------------------------------------------------------------
    // Layout of the override arrays: model = [dModel, layers, heads, dFf, window, segLen, streams];
    // trainF = [lr, lrMin, beta1, beta2, eps, clip, initScale] (mask bits 0..6);
    // trainI = [warmup, decay, seed] (mask bits 8..10). A set mask bit means "override this field".

    private const val MODEL_FIELDS = 7
    private const val TRAIN_F_FIELDS = 7
    private const val TRAIN_I_FIELDS = 3
    private const val TRAIN_I_SHIFT = 8

    @JvmStatic
    private external fun nativeVersion(): String

    @JvmStatic
    private external fun nativeFormatVersion(): Int

    @JvmStatic
    private external fun nativePreset(level: Int): String

    @JvmStatic
    private external fun nativeInfo(input: ByteArray): String

    @JvmStatic
    private external fun nativeCompress(
        input: ByteArray,
        level: Int,
        tokenizer: Int,
        bpeVocab: Int,
        threads: Int,
        allowStored: Boolean,
        builtinPrior: Boolean,
        prior: ByteArray?,
        modelMask: Int,
        model: IntArray,
        trainMask: Int,
        trainF: FloatArray,
        trainI: LongArray,
        progress: ProgressListener?,
    ): ByteArray

    @JvmStatic
    private external fun nativeDecompress(input: ByteArray, threads: Int, prior: ByteArray?, progress: ProgressListener?): ByteArray
}
