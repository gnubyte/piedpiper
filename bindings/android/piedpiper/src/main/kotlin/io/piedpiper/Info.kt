package io.piedpiper

import org.json.JSONObject

/** A network shape, as stored in a stream header or defined by a level preset. */
data class ModelConfig(
    val dModel: Int,
    val layers: Int,
    val heads: Int,
    val dFf: Int,
    val window: Int,
    val segLen: Int,
    val streams: Int,
) {
    /** These values as overrides, e.g. to compress again with exactly this network. */
    fun toOverrides(): ModelOverrides = ModelOverrides(dModel, layers, heads, dFf, window, segLen, streams)

    internal companion object {
        fun fromJson(o: JSONObject) = ModelConfig(
            dModel = o.getInt("d_model"),
            layers = o.getInt("n_layer"),
            heads = o.getInt("n_head"),
            dFf = o.getInt("d_ff"),
            window = o.getInt("window"),
            segLen = o.getInt("seg_len"),
            streams = o.getInt("n_streams"),
        )
    }
}

/** An online-training recipe, as stored in a stream header or defined by a level preset. */
data class TrainConfig(
    val lr: Float,
    val lrMin: Float,
    val warmup: Long,
    val decay: Long,
    val beta1: Float,
    val beta2: Float,
    val eps: Float,
    val clip: Float,
    val initScale: Float,
    val seed: Long,
) {
    /** These values as overrides, e.g. to compress again with exactly this recipe. */
    fun toOverrides(): TrainOverrides = TrainOverrides(lr, lrMin, warmup, decay, beta1, beta2, eps, clip, initScale, seed)

    internal companion object {
        // The native side writes every float as the exact decimal of its f32 value, so the
        // double -> float conversion below reproduces the stored bits.
        private fun JSONObject.float(key: String) = getDouble(key).toFloat()

        fun fromJson(o: JSONObject) = TrainConfig(
            lr = o.float("lr"),
            lrMin = o.float("lr_min"),
            warmup = o.getLong("warmup"),
            decay = o.getLong("decay"),
            beta1 = o.float("beta1"),
            beta2 = o.float("beta2"),
            eps = o.float("eps"),
            clip = o.float("clip"),
            initScale = o.float("init_scale"),
            seed = o.getLong("seed"),
        )
    }
}

/**
 * Header of a compressed stream ([PiedPiper.info]).
 *
 * @property originalSize size of the decompressed data in bytes.
 * @property compressedSize size of the stream in bytes.
 * @property crc32 CRC-32 (IEEE) of the original data, 0..2^32-1.
 * @property stored true if the data is stored verbatim (incompressible input).
 * @property tokenizer `bytes`, `bpe-<vocab>` or `stored`.
 * @property vocabSize number of symbols.
 * @property tokens number of coded symbols.
 * @property params trainable parameters of the network (0 if stored).
 * @property priorId id of the prior needed to decompress (0 = none), an unsigned 64-bit value.
 * @property priorName name of the built-in prior with that id (`text-s` / `text-m` / `text-l`), if any.
 * @property model network shape (null if stored).
 * @property train training recipe (null if stored).
 */
data class Info(
    val originalSize: Long,
    val compressedSize: Long,
    val crc32: Long,
    val stored: Boolean,
    val tokenizer: String,
    val vocabSize: Int,
    val tokens: Long,
    val params: Long,
    val priorId: Long,
    val priorName: String?,
    val model: ModelConfig?,
    val train: TrainConfig?,
) {
    /** Compression ratio (original / compressed). */
    val ratio: Double
        get() = if (compressedSize == 0L) 0.0 else originalSize.toDouble() / compressedSize

    /** Compressed bits per original byte. */
    val bitsPerByte: Double
        get() = if (originalSize == 0L) 0.0 else compressedSize * 8.0 / originalSize

    internal companion object {
        fun fromJson(o: JSONObject) = Info(
            originalSize = o.getLong("original_size"),
            compressedSize = o.getLong("compressed_size"),
            crc32 = o.getLong("crc32"),
            stored = o.getBoolean("stored"),
            tokenizer = o.getString("tokenizer"),
            vocabSize = o.getInt("vocab_size"),
            tokens = o.getLong("n_tokens"),
            params = o.getLong("n_params"),
            priorId = o.getString("prior_id").toULong(16).toLong(),
            priorName = if (o.isNull("prior_name")) null else o.getString("prior_name"),
            model = o.optJSONObject("model")?.let(ModelConfig::fromJson),
            train = o.optJSONObject("train")?.let(TrainConfig::fromJson),
        )
    }
}

/**
 * A compression level's defaults ([PiedPiper.preset]).
 *
 * @property model the network; `streams` is the maximum — for inputs without model overrides
 *   the encoder picks streams and segment length from the input size.
 * @property train the training recipe.
 * @property minStreamTokens minimum tokens per stream before another stream is added.
 * @property maxVocab largest BPE vocabulary the automatic tokenizer may choose.
 * @property priorLr peak learning rate used when fine-tuning a prior.
 */
data class Preset(
    val level: Int,
    val model: ModelConfig,
    val train: TrainConfig,
    val minStreamTokens: Int,
    val maxVocab: Int,
    val priorLr: Float,
) {
    internal companion object {
        fun fromJson(o: JSONObject) = Preset(
            level = o.getInt("level"),
            model = ModelConfig.fromJson(o.getJSONObject("model")),
            train = TrainConfig.fromJson(o.getJSONObject("train")),
            minStreamTokens = o.getInt("min_stream_tokens"),
            maxVocab = o.getInt("max_vocab"),
            priorLr = o.getDouble("prior_lr").toFloat(),
        )
    }
}
