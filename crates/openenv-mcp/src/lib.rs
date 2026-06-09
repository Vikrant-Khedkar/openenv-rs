use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Tool names that collide with the core env protocol and cannot be MCP tools.
pub const RESERVED_TOOL_NAMES: [&str; 4] = ["reset", "step", "state", "close"];

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub id: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(result: Value, id: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    pub fn error(code: i64, message: impl Into<String>, id: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
            id,
        }
    }
}

type ToolHandler = Box<dyn Fn(&Map<String, Value>) -> Result<Value, String> + Send + Sync>;

pub struct ToolEntry {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    handler: ToolHandler,
}

/// Registry of MCP tools exposed by an env server, the Rust counterpart of
/// FastMCP tool registration in Python OpenEnv.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, ToolEntry>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F>(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        handler: F,
    ) -> Result<(), String>
    where
        F: Fn(&Map<String, Value>) -> Result<Value, String> + Send + Sync + 'static,
    {
        let name = name.into();
        if RESERVED_TOOL_NAMES.contains(&name.as_str()) {
            return Err(format!("'{name}' is a reserved tool name"));
        }
        self.tools.insert(
            name.clone(),
            ToolEntry {
                name,
                description: description.into(),
                input_schema,
                handler: Box::new(handler),
            },
        );
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn list(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .values()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })
            })
            .collect();
        json!({ "tools": tools })
    }

    pub fn call(&self, name: &str, arguments: &Map<String, Value>) -> Result<Value, String> {
        let tool = self
            .tools
            .get(name)
            .ok_or(format!("Tool not found: {name}"))?;
        (tool.handler)(arguments)
    }

    /// Handle a JSON-RPC request (`tools/list` / `tools/call`), mirroring the
    /// Python `mcp_handler`.
    pub fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone();
        if req.jsonrpc != "2.0" {
            return JsonRpcResponse::error(INVALID_REQUEST, "jsonrpc must be '2.0'", id);
        }
        match req.method.as_str() {
            "tools/list" => JsonRpcResponse::success(self.list(), id),
            "tools/call" => {
                let Some(name) = req.params.get("name").and_then(|v| v.as_str()) else {
                    return JsonRpcResponse::error(INVALID_PARAMS, "Missing 'name' in params", id);
                };
                let empty = Map::new();
                let arguments = req
                    .params
                    .get("arguments")
                    .and_then(|v| v.as_object())
                    .unwrap_or(&empty);
                match self.call(name, arguments) {
                    Ok(result) => JsonRpcResponse::success(result, id),
                    Err(e) if e.starts_with("Tool not found") => {
                        JsonRpcResponse::error(INVALID_PARAMS, e, id)
                    }
                    Err(e) => JsonRpcResponse::error(INTERNAL_ERROR, e, id),
                }
            }
            other => {
                JsonRpcResponse::error(METHOD_NOT_FOUND, format!("Method not found: {other}"), id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(
            "echo_message",
            "Echo back the provided message",
            json!({"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}),
            |args| {
                let msg = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .ok_or("missing 'message'")?;
                Ok(json!(msg))
            },
        )
        .unwrap();
        reg
    }

    fn rpc(method: &str, params: Value, id: i64) -> JsonRpcRequest {
        serde_json::from_value(json!({
            "jsonrpc": "2.0", "method": method, "params": params, "id": id
        }))
        .unwrap()
    }

    #[test]
    fn reserved_names_rejected() {
        let mut reg = ToolRegistry::new();
        let err = reg
            .register("reset", "x", json!({}), |_| Ok(Value::Null))
            .unwrap_err();
        assert!(err.contains("reserved"));
    }

    #[test]
    fn tools_list_shape() {
        let resp = registry().handle(rpc("tools/list", json!({}), 1));
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["tools"][0]["name"], "echo_message");
        assert!(v["result"]["tools"][0]["inputSchema"].is_object());
        assert!(v.get("error").is_none());
    }

    #[test]
    fn tools_call_success_and_errors() {
        let reg = registry();

        let resp = reg.handle(rpc(
            "tools/call",
            json!({"name": "echo_message", "arguments": {"message": "hi"}}),
            2,
        ));
        assert_eq!(resp.result, Some(json!("hi")));

        let resp = reg.handle(rpc("tools/call", json!({"name": "nope"}), 3));
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);

        let resp = reg.handle(rpc("tools/call", json!({}), 4));
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);

        let resp = reg.handle(rpc("bogus/method", json!({}), 5));
        assert_eq!(resp.error.as_ref().unwrap().code, METHOD_NOT_FOUND);
    }
}
