package io.piedpiper

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import java.security.MessageDigest

/** One entry of testdata/golden/manifest.json (packaged as androidTest assets). */
data class GoldenVector(
    val name: String,
    val input: String,
    val compressed: String,
    val flags: String,
    val inputSha256: String,
    val compressedSha256: String,
    val inputSize: Int,
    val compressedSize: Int,
) {
    override fun toString(): String = name
}

object Golden {
    private val assets
        get() = InstrumentationRegistry.getInstrumentation().context.assets

    fun vectors(): List<GoldenVector> {
        val json = JSONArray(read("manifest.json").decodeToString())
        return (0 until json.length()).map { i ->
            val o = json.getJSONObject(i)
            GoldenVector(
                name = o.getString("name"),
                input = o.getString("input"),
                compressed = o.getString("compressed"),
                flags = o.getString("flags"),
                inputSha256 = o.getString("input_sha256"),
                compressedSha256 = o.getString("compressed_sha256"),
                inputSize = o.getInt("input_size"),
                compressedSize = o.getInt("compressed_size"),
            )
        }
    }

    fun read(name: String): ByteArray = assets.open(name).use { it.readBytes() }

    fun sha256(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }

    /** Level given by the flags (`-1`..`-9` / `--level N`), else the CLI default. */
    fun levelFromCliFlags(flags: String): Int = parse(flags).level

    /**
     * The desktop CLI's compression flags (crates/cli: `parse_args` + `build_options`) as
     * [CompressOptions]: `-1`..`-9`, `--level N`, `-T/--tokenizer auto|bytes|bpe|bpe:N`,
     * `-t/--threads N` and the advanced network / training flags.
     */
    fun optionsFromCliFlags(flags: String): CompressOptions = parse(flags)

    private fun parse(flags: String): CompressOptions {
        val args = flags.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        var opts = CompressOptions()
        var model = ModelOverrides()
        var train = TrainOverrides()
        var i = 0
        while (i < args.size) {
            val a = args[i++]
            fun value(): String = args.getOrNull(i++) ?: error("$a needs a value")
            when {
                a.length == 2 && a[0] == '-' && a[1] in '1'..'9' -> opts = opts.copy(level = a[1] - '0')
                a == "--level" -> opts = opts.copy(level = value().toInt())
                a == "-t" || a == "--threads" -> opts = opts.copy(threads = value().toInt())
                a == "-T" || a == "--tokenizer" -> opts = opts.copy(tokenizer = tokenizer(value()))
                a == "--d-model" -> model = model.copy(dModel = value().toInt())
                a == "--layers" -> model = model.copy(layers = value().toInt())
                a == "--heads" -> model = model.copy(heads = value().toInt())
                a == "--d-ff" -> model = model.copy(dFf = value().toInt())
                a == "--window" -> model = model.copy(window = value().toInt())
                a == "--seg" -> model = model.copy(segLen = value().toInt())
                a == "--streams" -> model = model.copy(streams = value().toInt())
                a == "--lr" -> train = train.copy(lr = value().toFloat())
                a == "--lr-min" -> train = train.copy(lrMin = value().toFloat())
                a == "--warmup" -> train = train.copy(warmup = value().toLong())
                a == "--decay" -> train = train.copy(decay = value().toLong())
                a == "--beta1" -> train = train.copy(beta1 = value().toFloat())
                a == "--beta2" -> train = train.copy(beta2 = value().toFloat())
                a == "--clip" -> train = train.copy(clip = value().toFloat())
                a == "--init-scale" -> train = train.copy(initScale = value().toFloat())
                a == "--seed" -> train = train.copy(seed = value().toULong().toLong())
                else -> error("unsupported CLI flag in golden manifest: $a")
            }
        }
        return opts.copy(
            model = model.takeIf { it != ModelOverrides() },
            train = train.takeIf { it != TrainOverrides() },
        )
    }

    private fun tokenizer(s: String): Tokenizer = when {
        s == "auto" -> Tokenizer.Auto
        s == "bytes" || s == "byte" -> Tokenizer.Bytes
        s == "bpe" || s == "words" || s == "word" -> Tokenizer.Bpe(4096)
        s.startsWith("bpe:") -> Tokenizer.Bpe(s.removePrefix("bpe:").toInt())
        else -> error("unknown tokenizer $s")
    }

    /** What the CLI would store in the header for these options' training recipe. */
    fun expectedTrain(opts: CompressOptions): TrainConfig {
        val p = PiedPiper.preset(opts.level).train
        val t = opts.train ?: TrainOverrides()
        return TrainConfig(
            lr = t.lr ?: p.lr,
            lrMin = t.lrMin ?: p.lrMin,
            warmup = t.warmup ?: p.warmup,
            decay = t.decay ?: p.decay,
            beta1 = t.beta1 ?: p.beta1,
            beta2 = t.beta2 ?: p.beta2,
            eps = t.eps ?: p.eps,
            clip = t.clip ?: p.clip,
            initScale = t.initScale ?: p.initScale,
            seed = t.seed ?: p.seed,
        )
    }
}
