package io.piedpiper

/** How the input is split into the symbols the model predicts. */
sealed class Tokenizer {
    /** Word-level BPE for large text (1 MiB or more), raw bytes otherwise. The default. */
    data object Auto : Tokenizer()

    /** One symbol per byte (256 symbols). */
    data object Bytes : Tokenizer()

    /**
     * Word-aware BPE learned from the input, with up to [vocab] symbols (clamped to 257..65535).
     * The desktop CLI's `-T bpe` uses 4096, `-T bpe:N` uses N.
     */
    data class Bpe(val vocab: Int = 4096) : Tokenizer()
}

/**
 * Options of [PiedPiper.compress]. The defaults match the desktop CLI's defaults.
 *
 * @property level 1 (fastest) .. 9 (strongest); see [PiedPiper.preset].
 * @property tokenizer symbolisation of the input.
 * @property threads worker threads, 0 = all cores. Never changes the output.
 * @property model overrides of the level's network shape (CLI: `--d-model`, `--layers`, ...).
 * @property train overrides of the level's training recipe (CLI: `--lr`, `--beta1`, `--clip`, ...).
 * @property prior a pretrained prior (`.ppp` file contents) to start from; its network shape and
 *   tokenizer replace the level's. The same prior is needed to decompress.
 * @property allowStored store the input verbatim when the model cannot beat that (random or
 *   already-compressed data).
 *
 * Overrides follow the CLI exactly, so the same flags give the same bytes on every platform:
 * with any override, the network and the training recipe start from [PiedPiper.preset] of
 * [level]; a [model] override pins the whole network (used verbatim, so `segLen` / `streams`
 * come from the preset unless set, instead of being picked from the input size), and any override
 * at all pins the training recipe (which also turns off the lower fine-tuning learning rate that
 * is otherwise used with a [prior]).
 */
data class CompressOptions @JvmOverloads constructor(
    val level: Int = PiedPiper.DEFAULT_LEVEL,
    val tokenizer: Tokenizer = Tokenizer.Auto,
    val threads: Int = 0,
    val model: ModelOverrides? = null,
    val train: TrainOverrides? = null,
    val prior: ByteArray? = null,
    val allowStored: Boolean = true,
    /** Start text-like input from the built-in prior of levels 1-3 when no [prior] is given. */
    val builtinPrior: Boolean = true,
) {
    override fun equals(other: Any?): Boolean =
        other is CompressOptions && level == other.level && tokenizer == other.tokenizer && threads == other.threads &&
            model == other.model && train == other.train && allowStored == other.allowStored &&
            builtinPrior == other.builtinPrior && prior.contentEquals(other.prior)

    override fun hashCode(): Int =
        listOf(level, tokenizer, threads, model, train, allowStored, builtinPrior, prior.contentHashCode()).hashCode()
}

/**
 * Options of [PiedPiper.decompress].
 *
 * @property threads worker threads, 0 = all cores. Never changes the output.
 * @property prior the prior the stream was compressed with, if any.
 */
data class DecompressOptions @JvmOverloads constructor(
    val threads: Int = 0,
    val prior: ByteArray? = null,
) {
    override fun equals(other: Any?): Boolean =
        other is DecompressOptions && threads == other.threads && prior.contentEquals(other.prior)

    override fun hashCode(): Int = 31 * threads + prior.contentHashCode()
}

/**
 * Overrides of the level's network shape; `null` keeps the preset's value. The desktop CLI's
 * `--d-model --layers --heads --d-ff --window --seg --streams`.
 *
 * @property dModel model width (multiple of [heads], head dimension even).
 * @property layers Transformer blocks.
 * @property heads attention heads.
 * @property dFf hidden width of the feed-forward block.
 * @property window attention span in tokens.
 * @property segLen tokens per stream between two weight updates.
 * @property streams parallel streams the token sequence is split into.
 */
data class ModelOverrides(
    val dModel: Int? = null,
    val layers: Int? = null,
    val heads: Int? = null,
    val dFf: Int? = null,
    val window: Int? = null,
    val segLen: Int? = null,
    val streams: Int? = null,
)

/**
 * Overrides of the level's online-training recipe; `null` keeps the preset's value. The desktop
 * CLI's `--lr --lr-min --warmup --decay --beta1 --beta2 --clip --init-scale --seed` (plus [eps]).
 *
 * @property lr peak learning rate.
 * @property lrMin learning-rate floor.
 * @property warmup updates until the peak learning rate (>= 1).
 * @property decay updates after warm-up at which the decay is half-way to [lrMin].
 * @property beta1 Adam β1 (0 = no momentum, the default).
 * @property beta2 Adam β2.
 * @property eps Adam ε.
 * @property clip global gradient-norm clip (0 = off).
 * @property initScale multiplier on the default weight-init scale.
 * @property seed weight-init seed (any 64-bit pattern; read as unsigned).
 */
data class TrainOverrides(
    val lr: Float? = null,
    val lrMin: Float? = null,
    val warmup: Long? = null,
    val decay: Long? = null,
    val beta1: Float? = null,
    val beta2: Float? = null,
    val eps: Float? = null,
    val clip: Float? = null,
    val initScale: Float? = null,
    val seed: Long? = null,
)
