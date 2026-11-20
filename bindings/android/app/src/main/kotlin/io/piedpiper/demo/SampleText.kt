package io.piedpiper.demo

/** Deterministic, repetitive-but-not-trivial English-like text to try the codec without a file. */
internal object SampleText {
    private val subjects = listOf(
        "The model", "A small transformer", "The range coder", "Every stream", "The decoder",
        "An online learner", "The piper", "Each segment", "The quiet village", "A patient listener",
    )
    private val verbs = listOf(
        "predicts", "encodes", "learns", "remembers", "compresses", "follows", "repeats", "shapes", "guides",
    )
    private val objects = listOf(
        "the next token", "a simple melody", "the hidden pattern", "every byte", "its own weights",
        "a careful guess", "the old stone bridge", "the signal", "the river of numbers", "a familiar phrase",
    )
    private val tails = listOf(
        "again and again", "one step at a time", "without storing a single weight", "on every platform",
        "bit for bit", "while it reads", "a little better each time",
    )

    fun generate(size: Int): ByteArray {
        val sb = StringBuilder(size + 128)
        var state = 0x2545F4914F6CDD1DL
        fun next(n: Int): Int {
            // xorshift64
            state = state xor (state shl 13)
            state = state xor (state ushr 7)
            state = state xor (state shl 17)
            return ((state ushr 1) % n).toInt()
        }
        var sentences = 0
        while (sb.length < size) {
            sb.append(subjects[next(subjects.size)]).append(' ')
                .append(verbs[next(verbs.size)]).append(' ')
                .append(objects[next(objects.size)])
            if (next(3) == 0) sb.append(' ').append(tails[next(tails.size)])
            if (next(5) == 0) sb.append(", and ").append(next(1000))
            sb.append(if (++sentences % 6 == 0) ".\n" else ". ")
        }
        return sb.substring(0, size).encodeToByteArray()
    }
}
