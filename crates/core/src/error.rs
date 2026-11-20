use std::fmt;

/// Everything that can go wrong while compressing or decompressing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Not a PiedPiper stream, or the header is malformed / truncated.
    Format(String),
    /// The stream was written by an incompatible (newer) format version.
    Version(u8),
    /// The decoded data failed its CRC-32 check (corrupt or truncated input,
    /// or a different prior model than the one used for compression).
    Checksum { expected: u32, actual: u32 },
    /// Invalid compression options.
    Options(String),
    /// The progress callback asked to stop.
    Cancelled,
    /// The stream asks for more memory than this build allows.
    TooLarge(String),
    /// A prior model is required, missing, mismatched or malformed.
    Prior(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Format(m) => write!(f, "invalid PiedPiper data: {m}"),
            Error::Version(v) => write!(f, "unsupported PiedPiper format version {v}"),
            Error::Checksum { expected, actual } => write!(
                f,
                "checksum mismatch (expected {expected:08x}, got {actual:08x}): data is corrupt"
            ),
            Error::Options(m) => write!(f, "invalid options: {m}"),
            Error::Cancelled => write!(f, "operation cancelled"),
            Error::TooLarge(m) => write!(f, "stream too large: {m}"),
            Error::Prior(m) => write!(f, "prior: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
