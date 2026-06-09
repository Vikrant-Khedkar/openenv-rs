use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("server error [{code}]: {message}")]
    Server { code: String, message: String },
    #[error("connection closed by server")]
    Closed,
}
