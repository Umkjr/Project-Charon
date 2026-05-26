use axum::{
    routing::post,
    Router,
    Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::net::SocketAddr;

// ==============================================================================
// 1. Structs for OpenAI-Compatible Payloads
// ==============================================================================

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

// ==============================================================================
// 2. Conductor Routing Engine & Cache Index
// ==============================================================================

struct ConductorState {
    // Distributed Token Cache Index: Maps prompt prefix hashes to GPU Worker Node Addresses
    cache_index: HashMap<String, String>,
    // Cluster Node Fleets
    prefill_nodes: Vec<String>,
    decode_nodes: Vec<String>,
    next_node_index: usize,
}

impl ConductorState {
    fn new() -> Self {
        Self {
            cache_index: HashMap::new(),
            prefill_nodes: vec!["http://127.0.0.1:8001".to_string()],
            decode_nodes: vec!["http://127.0.0.1:8002".to_string()],
            next_node_index: 0,
        }
    }

    // Dynamic routing resolution based on prompt prefix hashing
    fn resolve_target_node(&mut self, prompt: &str) -> (String, bool) {
        // Hash calculation of prompt prefix (first 100 characters for simulation prefix mapping)
        let prefix_len = std::cmp::min(100, prompt.len());
        let prefix = &prompt[0..prefix_len];
        let hash = format!("{}", md5::compute(prefix.as_bytes()));
        
        println!("[Conductor Engine] Computed Prompt Prefix Hash: {}", hash);

        if let Some(target_gpu_node) = self.cache_index.get(&hash) {
            // Cache Hit: Route directly to the specific GPU that already holds this context
            println!(
                "[Conductor CACHE HIT] Routing request directly to owner GPU node: {}",
                target_gpu_node
            );
            return (target_gpu_node.clone(), true);
        }

        // Cache Miss: Route to a compute-heavy Prefill Node to build initial cache,
        // and register this node as the future owner of this token context
        let target_prefill_node = &self.prefill_nodes[self.next_node_index % self.prefill_nodes.len()];
        self.cache_index.insert(hash, target_prefill_node.clone());
        self.next_node_index += 1;

        println!(
            "[Conductor CACHE MISS] Routing request to Prefill Node to generate cache: {}",
            target_prefill_node
        );
        (target_prefill_node.clone(), false)
    }
}

// ==============================================================================
// 3. API Handlers
// ==============================================================================

async fn handle_chat_completion(
    axum::extract::State(state): axum::extract::State<Arc<Mutex<ConductorState>>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    
    // Conjoin messages content as prompt string
    let full_prompt: String = payload
        .messages
        .iter()
        .map(|m| m.content.clone())
        .collect::<Vec<String>>()
        .join(" ");

    println!("\n[Conductor Proxy] Intercepted new completions request!");
    println!("[Conductor Proxy] Prompt length: {} characters", full_prompt.len());

    // Resolve routing node
    let (target_node, is_cache_hit) = {
        let mut engine = state.lock().map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("State lock poisoned: {}", e))
        })?;
        engine.resolve_target_node(&full_prompt)
    };

    // Construct response
    let hit_type = if is_cache_hit { "CACHE_HIT" } else { "CACHE_MISS" };
    let response_text = format!(
        "[Conductor Route: {}] Hello! Your request was routed to node: {}.",
        hit_type, target_node
    );

    let response = ChatCompletionResponse {
        id: "chatcmpl-conductor-991".to_string(),
        object: "chat.completion".to_string(),
        created: 1716716000,
        model: payload.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: "stop".to_string(),
        }],
    };

    Ok(Json(response))
}

// ==============================================================================
// 4. Main Entry Point
// ==============================================================================

#[tokio::main]
async fn main() {
    println!("================================================================================");
    println!("                  PROJECT CHARON: THE CONDUCTOR GLOBAL PROXY");
    println!("================================================================================");

    let state = Arc::new(Mutex::new(ConductorState::new()));

    // Define Axum API Router
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("[Conductor Startup] Listening and routing on: http://{}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

// MD5 custom mock implementation for single-file self-contained compilation
mod md5 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    pub struct Digest(u64);

    impl std::fmt::Display for Digest {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:016x}", self.0)
        }
    }

    pub fn compute<T: Hash>(t: T) -> Digest {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        Digest(s.finish())
    }
}
