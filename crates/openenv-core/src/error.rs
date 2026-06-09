use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnvError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("execution error: {0}")]
    Execution(String),
    #[error("timeout after {0}s")]
    Timeout(f64),
}

impl EnvError {
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    pub fn execution(msg: impl Into<String>) -> Self {
        Self::Execution(msg.into())
    }
}
