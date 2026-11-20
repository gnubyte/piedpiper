package io.piedpiper.demo

import android.app.Activity
import android.content.Intent
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.OpenableColumns
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.TextView
import io.piedpiper.CompressOptions
import io.piedpiper.Info
import io.piedpiper.PiedPiper
import io.piedpiper.PiedPiperCancelledException
import io.piedpiper.PiedPiperException
import io.piedpiper.ProgressListener
import java.io.FileNotFoundException
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * PiedPiper demo: pick a file with the Storage Access Framework (or use the built-in sample
 * text), compress or decompress it on a background thread with a progress bar and cancellation,
 * and save the result with the Storage Access Framework.
 */
class MainActivity : Activity() {

    /** The loaded input; [info] is set when it is a PiedPiper stream. */
    private class Input(val name: String, val bytes: ByteArray, val info: Info?)

    private class Output(val name: String, val bytes: ByteArray)

    private val worker: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "piedpiper-demo") }
    private val main = Handler(Looper.getMainLooper())

    private lateinit var inputInfo: TextView
    private lateinit var levelValue: TextView
    private lateinit var level: SeekBar
    private lateinit var pick: Button
    private lateinit var sample: Button
    private lateinit var compress: Button
    private lateinit var decompress: Button
    private lateinit var cancel: Button
    private lateinit var reuse: Button
    private lateinit var save: Button
    private lateinit var progress: ProgressBar
    private lateinit var status: TextView
    private lateinit var statusColors: ColorStateList

    private var input: Input? = null
    private var output: Output? = null
    private var resultText: CharSequence = ""
    private var loading = false
    private var running = false
    private var destroyed = false

    @Volatile
    private var cancelRequested = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        applySystemBarInsets(findViewById(R.id.root))

        inputInfo = findViewById(R.id.inputInfo)
        levelValue = findViewById(R.id.levelValue)
        level = findViewById(R.id.level)
        pick = findViewById(R.id.pick)
        sample = findViewById(R.id.sample)
        compress = findViewById(R.id.compress)
        decompress = findViewById(R.id.decompress)
        cancel = findViewById(R.id.cancel)
        reuse = findViewById(R.id.reuse)
        save = findViewById(R.id.save)
        progress = findViewById(R.id.progress)
        status = findViewById(R.id.status)
        statusColors = status.textColors

        findViewById<TextView>(R.id.subtitle).text = getString(R.string.subtitle, PiedPiper.version)
        level.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, value: Int, fromUser: Boolean) = showLevel()
            override fun onStartTrackingTouch(seekBar: SeekBar) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar) = Unit
        })
        showLevel()

        pick.setOnClickListener { chooseFile() }
        sample.setOnClickListener { setInput(Input("sample.txt", SampleText.generate(SAMPLE_SIZE), null)) }
        compress.setOnClickListener { run(compress = true) }
        decompress.setOnClickListener { run(compress = false) }
        cancel.setOnClickListener {
            cancelRequested = true
            cancel.isEnabled = false
        }
        reuse.setOnClickListener { useOutputAsInput() }
        save.setOnClickListener { chooseSaveLocation() }
        updateButtons()
    }

    override fun onDestroy() {
        destroyed = true
        cancelRequested = true
        worker.shutdown()
        super.onDestroy()
    }

    // --- Input ---------------------------------------------------------------------------------

    private fun chooseFile() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            .addCategory(Intent.CATEGORY_OPENABLE)
            .setType("*/*")
        @Suppress("DEPRECATION")
        startActivityForResult(intent, REQUEST_OPEN)
    }

    // The framework Activity result API keeps this demo free of AndroidX dependencies.
    @Suppress("OVERRIDE_DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (resultCode != RESULT_OK || uri == null) return
        when (requestCode) {
            REQUEST_OPEN -> load(uri)
            REQUEST_SAVE -> store(uri)
        }
    }

    private fun load(uri: Uri) {
        loading = true
        inputInfo.text = getString(R.string.loading)
        updateButtons()
        worker.execute {
            val result = runCatching {
                val (name, size) = queryNameAndSize(uri)
                if (size != null && size > MAX_INPUT) {
                    throw IllegalArgumentException("${human(size)} is more than this demo loads (${human(MAX_INPUT)})")
                }
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: throw FileNotFoundException("cannot open $uri")
                if (bytes.size > MAX_INPUT) {
                    throw IllegalArgumentException("${human(bytes.size.toLong())} is more than this demo loads (${human(MAX_INPUT)})")
                }
                val info = if (PiedPiper.isCompressed(bytes)) runCatching { PiedPiper.info(bytes) }.getOrNull() else null
                Input(name ?: uri.lastPathSegment ?: "input", bytes, info)
            }
            main.post {
                loading = false
                if (destroyed) return@post
                result.fold(
                    onSuccess = { setInput(it) },
                    onFailure = {
                        inputInfo.text = "Could not read the file: ${it.message ?: it}"
                        updateButtons()
                    },
                )
            }
        }
    }

    private fun queryNameAndSize(uri: Uri): Pair<String?, Long?> =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (!c.moveToFirst()) return@use null to null
            val nameCol = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            val sizeCol = c.getColumnIndex(OpenableColumns.SIZE)
            val name = if (nameCol >= 0 && !c.isNull(nameCol)) c.getString(nameCol) else null
            val size = if (sizeCol >= 0 && !c.isNull(sizeCol)) c.getLong(sizeCol) else null
            name to size
        } ?: (null to null)

    private fun setInput(newInput: Input) {
        input = newInput
        output = null
        resultText = ""
        progress.progress = 0
        status.setTextColor(statusColors)
        status.text = getString(R.string.idle)
        val info = newInput.info
        inputInfo.text = buildString {
            append(newInput.name).append(" · ").append(human(newInput.bytes.size.toLong()))
            if (info != null) {
                append("\nPiedPiper stream → ").append(human(info.originalSize)).append(" original")
                val m = info.model
                if (m != null) {
                    append("\n").append(info.tokenizer).append(" tokens, ").append(m.layers).append(" layers × d").append(m.dModel)
                    append(", ").append(String.format(Locale.getDefault(), "%.2fM", info.params / 1e6)).append(" params")
                } else {
                    append(" (stored)")
                }
            }
        }
        updateButtons()
    }

    // --- Compress / decompress -----------------------------------------------------------------

    private fun showLevel() {
        levelValue.text = getString(R.string.level_value, level.progress + 1)
    }

    private fun run(compress: Boolean) {
        val src = input ?: return
        val options = CompressOptions(level = level.progress + 1)
        running = true
        cancelRequested = false
        output = null
        progress.progress = 0
        status.setTextColor(statusColors)
        status.text = if (compress) "Compressing at level ${options.level}…" else "Decompressing…"
        updateButtons()
        val t0 = SystemClock.elapsedRealtime()
        worker.execute {
            var lastPost = 0L
            // Called on a PiedPiper worker thread after every segment: throttle UI updates.
            val listener = ProgressListener { done, total ->
                val now = SystemClock.elapsedRealtime()
                if (now - lastPost >= 100 || done == total) {
                    lastPost = now
                    main.post { if (running && !destroyed) showProgress(done, total, now - t0) }
                }
                !cancelRequested
            }
            val result = runCatching {
                if (compress) {
                    PiedPiper.compress(src.bytes, options, listener)
                } else {
                    PiedPiper.decompress(src.bytes, progress = listener)
                }
            }
            val ms = SystemClock.elapsedRealtime() - t0
            main.post { finished(compress, src, options.level, result, ms) }
        }
    }

    private fun showProgress(done: Long, total: Long, ms: Long) {
        val fraction = if (total > 0) done.toDouble() / total else 1.0
        progress.progress = (fraction * progress.max).toInt()
        val rate = done * 1000.0 / ms.coerceAtLeast(1)
        val eta = if (rate > 0) ((total - done) / rate).toLong() else 0L
        status.text = String.format(Locale.getDefault(), "%.1f%%  ·  %s/s  ·  %s elapsed  ·  ETA %s", fraction * 100, human(rate.toLong()), clock(ms / 1000), clock(eta))
    }

    private fun finished(compress: Boolean, src: Input, level: Int, result: Result<ByteArray>, ms: Long) {
        running = false
        if (destroyed) return
        result.fold(
            onSuccess = { out ->
                progress.progress = progress.max
                output = Output(outputName(src.name, compress), out)
                val secs = ms / 1000.0
                val speed = human((src.bytes.size.coerceAtLeast(out.size) / secs.coerceAtLeast(1e-3)).toLong())
                resultText = if (compress) {
                    val ratio = if (out.isEmpty()) 0.0 else src.bytes.size.toDouble() / out.size
                    val bpb = if (src.bytes.isEmpty()) 0.0 else out.size * 8.0 / src.bytes.size
                    val stored = if (PiedPiper.info(out).stored) " (stored: not compressible)" else ""
                    String.format(
                        Locale.getDefault(),
                        "%s → %s%s\nratio %.3f×  ·  %.3f bits/byte\n%.2f s  ·  %s/s  ·  level %d",
                        human(src.bytes.size.toLong()), human(out.size.toLong()), stored, ratio, bpb, secs, speed, level,
                    )
                } else {
                    String.format(
                        Locale.getDefault(),
                        "%s → %s  (CRC-32 verified)\n%.2f s  ·  %s/s",
                        human(src.bytes.size.toLong()), human(out.size.toLong()), secs, speed,
                    )
                }
                status.text = resultText
            },
            onFailure = { e ->
                progress.progress = 0
                resultText = when (e) {
                    is PiedPiperCancelledException -> getString(R.string.cancelled)
                    is PiedPiperException -> "Failed (${e.kind}): ${e.message}"
                    is OutOfMemoryError -> "Out of memory: the input is too large for this device."
                    else -> "Failed: $e"
                }
                if (e !is PiedPiperCancelledException) status.setTextColor(getColor(R.color.error))
                status.text = resultText
            },
        )
        updateButtons()
    }

    /** Feeds the result back in, e.g. to decompress what was just compressed (a roundtrip check). */
    private fun useOutputAsInput() {
        val out = output ?: return
        val info = if (PiedPiper.isCompressed(out.bytes)) runCatching { PiedPiper.info(out.bytes) }.getOrNull() else null
        setInput(Input(out.name, out.bytes, info))
    }

    // --- Save ----------------------------------------------------------------------------------

    private fun chooseSaveLocation() {
        val out = output ?: return
        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT)
            .addCategory(Intent.CATEGORY_OPENABLE)
            .setType("application/octet-stream")
            .putExtra(Intent.EXTRA_TITLE, out.name)
        @Suppress("DEPRECATION")
        startActivityForResult(intent, REQUEST_SAVE)
    }

    private fun store(uri: Uri) {
        val out = output ?: return
        worker.execute {
            val result = runCatching {
                // "wt" truncates an existing document; some providers only support "w".
                val stream = try {
                    contentResolver.openOutputStream(uri, "wt")
                } catch (e: IllegalArgumentException) {
                    contentResolver.openOutputStream(uri, "w")
                } catch (e: FileNotFoundException) {
                    contentResolver.openOutputStream(uri, "w")
                }
                (stream ?: throw FileNotFoundException("cannot open $uri")).use { it.write(out.bytes) }
                queryNameAndSize(uri).first ?: out.name
            }
            main.post {
                if (destroyed) return@post
                val line = result.fold(
                    onSuccess = { getString(R.string.saved, it, human(out.bytes.size.toLong())) },
                    onFailure = { "Could not save: ${it.message ?: it}" },
                )
                status.text = if (resultText.isEmpty()) line else "$resultText\n\n$line"
            }
        }
    }

    // --- Helpers -------------------------------------------------------------------------------

    private fun updateButtons() {
        val idle = !loading && !running
        pick.isEnabled = idle
        sample.isEnabled = idle
        level.isEnabled = idle
        compress.isEnabled = idle && input != null
        decompress.isEnabled = idle && input?.info != null
        save.isEnabled = idle && output != null
        reuse.isEnabled = idle && output != null
        cancel.isEnabled = running && !cancelRequested
        cancel.visibility = if (running) View.VISIBLE else View.GONE
    }

    /** Edge-to-edge is enforced from Android 15 for this targetSdk: keep content off the system bars. */
    private fun applySystemBarInsets(root: View) {
        if (Build.VERSION.SDK_INT < 35) return
        val (l, t, r, b) = listOf(root.paddingLeft, root.paddingTop, root.paddingRight, root.paddingBottom)
        root.setOnApplyWindowInsetsListener { v, insets ->
            val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
            v.setPadding(l + bars.left, t + bars.top, r + bars.right, b + bars.bottom)
            insets
        }
    }

    private fun outputName(name: String, compressed: Boolean): String = when {
        compressed -> "$name.pp"
        name.endsWith(".pp") && name.length > 3 -> name.removeSuffix(".pp")
        else -> "$name.out"
    }

    private fun human(bytes: Long): String = when {
        bytes >= 1_000_000_000 -> String.format(Locale.getDefault(), "%.2f GB", bytes / 1e9)
        bytes >= 1_000_000 -> String.format(Locale.getDefault(), "%.2f MB", bytes / 1e6)
        bytes >= 1_000 -> String.format(Locale.getDefault(), "%.1f KB", bytes / 1e3)
        else -> "$bytes B"
    }

    private fun clock(seconds: Long): String =
        if (seconds >= 3600) {
            String.format(Locale.getDefault(), "%d:%02d:%02d", seconds / 3600, seconds / 60 % 60, seconds % 60)
        } else {
            String.format(Locale.getDefault(), "%d:%02d", seconds / 60, seconds % 60)
        }

    private companion object {
        const val REQUEST_OPEN = 1
        const val REQUEST_SAVE = 2
        const val SAMPLE_SIZE = 32_000
        const val MAX_INPUT = 64L shl 20
    }
}
