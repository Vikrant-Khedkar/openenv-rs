use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use openenv_core::{
    EnvError, EnvironmentMetadata, HealthResponse, ResetRequest, SchemaResponse, StepRequest,
    StepResponse,
};
use serde_json::{json, Value};

use crate::ServerState;

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/reset", post(reset))
        .route("/step", post(step))
        .route("/state", get(get_state))
        .route("/metadata", get(metadata))
        .route("/health", get(health))
        .route("/schema", get(schema))
        .with_state(state)
}

struct ApiError(StatusCode, String);

impl From<EnvError> for ApiError {
    fn from(e: EnvError) -> Self {
        let status = match e {
            EnvError::Validation(_) => StatusCode::UNPROCESSABLE_ENTITY,
            EnvError::Execution(_) => StatusCode::INTERNAL_SERVER_ERROR,
            EnvError::Timeout(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self(status, e.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"detail": self.1}))).into_response()
    }
}

async fn reset(
    State(state): State<ServerState>,
    Json(req): Json<ResetRequest>,
) -> Result<Json<StepResponse>, ApiError> {
    let mut env = state.http_env.lock().await;
    Ok(Json(env.reset(req)?))
}

async fn step(
    State(state): State<ServerState>,
    Json(req): Json<StepRequest>,
) -> Result<Json<StepResponse>, ApiError> {
    let mut env = state.http_env.lock().await;
    Ok(Json(env.step(req.action)?))
}

async fn get_state(State(state): State<ServerState>) -> Result<Json<Value>, ApiError> {
    let env = state.http_env.lock().await;
    Ok(Json(env.state()?))
}

async fn metadata(State(state): State<ServerState>) -> Json<EnvironmentMetadata> {
    let env = state.http_env.lock().await;
    Json(env.metadata())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse::default())
}

async fn schema(State(state): State<ServerState>) -> Json<SchemaResponse> {
    let env = state.http_env.lock().await;
    Json(env.schemas())
}
