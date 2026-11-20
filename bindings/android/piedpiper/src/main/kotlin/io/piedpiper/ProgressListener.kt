package io.piedpiper

/**
 * Receives the progress of a [PiedPiper.compress] / [PiedPiper.decompress] call and can cancel it.
 *
 * Called from one of PiedPiper's worker threads (not the calling thread, never the main thread)
 * after every coded segment — possibly thousands of times per second on small segments — so keep
 * it cheap: update an atomic, or post to the UI at most every few hundred milliseconds. Blocking
 * here stalls the codec. If the listener throws, the operation is cancelled and the exception is
 * re-thrown from the `compress` / `decompress` call.
 */
fun interface ProgressListener {
    /**
     * @param done bytes processed so far: input bytes when compressing, output bytes when
     *   decompressing.
     * @param total total bytes (the input size when compressing, the original size when
     *   decompressing). The last call has `done == total`. Nothing is reported for empty input
     *   or when decompressing a stored (incompressible) stream.
     * @return `true` to continue, `false` to cancel; the call then throws
     *   [PiedPiperCancelledException].
     */
    fun onProgress(done: Long, total: Long): Boolean
}
