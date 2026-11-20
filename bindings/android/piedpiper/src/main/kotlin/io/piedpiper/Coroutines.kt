package io.piedpiper

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.job
import kotlinx.coroutines.withContext

/**
 * [PiedPiper.compress] as a cancellable suspending call. It waits on [Dispatchers.IO] (the work
 * runs on PiedPiper's worker threads); cancelling the calling coroutine stops the codec at its
 * next progress step and throws [PiedPiperCancelledException] (a `CancellationException`).
 *
 * ```
 * lifecycleScope.launch {
 *     val packed = PiedPiper.compressSuspend(bytes, CompressOptions(level = 3))
 * }
 * ```
 */
suspend fun PiedPiper.compressSuspend(
    data: ByteArray,
    options: CompressOptions = CompressOptions(),
    progress: ProgressListener? = null,
): ByteArray = withContext(Dispatchers.IO) {
    val job = coroutineContext.job
    compress(data, options) { done, total -> job.isActive && (progress?.onProgress(done, total) ?: true) }
}

/** [PiedPiper.decompress] as a cancellable suspending call; see [compressSuspend]. */
suspend fun PiedPiper.decompressSuspend(
    data: ByteArray,
    options: DecompressOptions = DecompressOptions(),
    progress: ProgressListener? = null,
): ByteArray = withContext(Dispatchers.IO) {
    val job = coroutineContext.job
    decompress(data, options) { done, total -> job.isActive && (progress?.onProgress(done, total) ?: true) }
}
