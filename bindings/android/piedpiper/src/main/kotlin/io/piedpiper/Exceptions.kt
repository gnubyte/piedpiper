package io.piedpiper

import java.util.concurrent.CancellationException

/**
 * A PiedPiper operation failed. [message] is the native library's description.
 *
 * @property code numeric error code (see [Kind]); thrown with this constructor by the native code.
 */
open class PiedPiperException(message: String, val code: Int) : RuntimeException(message) {

    /** What went wrong. */
    val kind: Kind
        get() = Kind.of(code)

    /** Error categories (mirroring the core's `Error` enum). */
    enum class Kind(val code: Int) {
        /** Not a PiedPiper stream, or the header is malformed / truncated. */
        FORMAT(1),

        /** The stream was written by an incompatible (newer) format version. */
        VERSION(2),

        /** The decoded data failed its CRC-32 check: corrupt or truncated input, or the wrong prior. */
        CHECKSUM(3),

        /** Invalid compression options (level, network shape, training recipe, ...). */
        OPTIONS(4),

        /** The stream or configuration needs more memory than this build allows. */
        TOO_LARGE(6),

        /** A prior is required, missing, mismatched or malformed. */
        PRIOR(7),

        /** Unexpected internal failure (a bug: please report the message). */
        INTERNAL(8),
        ;

        internal companion object {
            fun of(code: Int): Kind = entries.firstOrNull { it.code == code } ?: INTERNAL
        }
    }
}

/**
 * The operation was cancelled: a [ProgressListener] returned `false`, or the coroutine running
 * [compressSuspend] / [decompressSuspend] was cancelled. A [CancellationException], so coroutines
 * treat it as cancellation rather than failure.
 */
class PiedPiperCancelledException(message: String) : CancellationException(message)
