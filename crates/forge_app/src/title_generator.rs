use std::sync::Arc;

use derive_setters::Setters;
use forge_domain::{
    ChatCompletionMessageFull, Context, ContextMessage, ConversationId, ModelId, ProviderId,
    ReasoningConfig, ResponseFormat, ResultStreamExt, StripUnsupportedReasoning, Transformer,
    UserPrompt,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::TemplateEngine;
use crate::agent::AgentService as AS;

/// Structured response for title generation using JSON format
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "title")]
pub struct TitleResponse {
    /// The generated title for the conversation
    pub title: String,
}

/// Service for generating contextually appropriate titles
#[derive(Setters)]
pub struct TitleGenerator<S> {
    /// Shared reference to the agent services used for AI interactions
    services: Arc<S>,
    /// The user prompt to generate a title for
    user_prompt: UserPrompt,
    /// The model ID to use for title generation
    model_id: ModelId,
    /// Reasoning configuration for the generator.
    reasoning: Option<ReasoningConfig>,
    /// The provider ID to use for title generation
    provider_id: Option<ProviderId>,
}

impl<S: AS> TitleGenerator<S> {
    pub fn new(
        services: Arc<S>,
        user_prompt: UserPrompt,
        model_id: ModelId,
        provider_id: Option<ProviderId>,
    ) -> Self {
        Self {
            services,
            user_prompt,
            model_id,
            reasoning: None,
            provider_id,
        }
    }

    pub async fn generate(&self) -> anyhow::Result<Option<String>> {
        let template = TemplateEngine::default().render(
            "forge-system-prompt-title-generation.md",
            &Default::default(),
        )?;

        let prompt = format!("<user_prompt>{}</user_prompt>", self.user_prompt.as_str());

        // Generate JSON schema from TitleResponse using schemars
        let schema = schemars::schema_for!(TitleResponse);

        let mut ctx = Context::default()
            .temperature(1.0f32)
            .conversation_id(ConversationId::generate())
            .add_message(ContextMessage::system(template))
            .add_message(ContextMessage::user(prompt, Some(self.model_id.clone())))
            .response_format(ResponseFormat::JsonSchema(Box::new(schema)));

        // Set the reasoning if configured.
        if let Some(reasoning) = self.reasoning.as_ref() {
            ctx = ctx.reasoning(reasoning.clone());
        }

        let provider_id = self.provider_id.clone();
        let ctx = StripUnsupportedReasoning::new(self.model_id.clone()).transform(ctx);

        let stream = self
            .services
            .chat_agent(&self.model_id, ctx, provider_id)
            .await?;
        let ChatCompletionMessageFull { content, .. } = stream.into_full(false).await?;

        // Parse the response - try JSON first (structured output), fallback to plain
        // text
        match serde_json::from_str::<TitleResponse>(&content) {
            Ok(response) => Ok(Some(response.title)),
            Err(_) => {
                // Fallback: Some providers don't support structured output, treat as plain text
                Ok(Some(content.trim().to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use forge_domain::{
        Agent, ChatCompletionMessage, Conversation, Effort, ProviderId, ReasoningConfig,
        ToolCallContext, ToolCallFull, ToolResult,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(Clone)]
    struct MockAgentService {
        captured_context: Arc<Mutex<Option<Context>>>,
    }

    #[async_trait]
    impl AS for MockAgentService {
        async fn chat_agent(
            &self,
            _id: &ModelId,
            context: Context,
            _provider_id: Option<ProviderId>,
        ) -> forge_domain::ResultStream<ChatCompletionMessage, anyhow::Error> {
            *self.captured_context.lock().unwrap() = Some(context);

            let message = ChatCompletionMessage::assistant("{\"title\":\"fixture title\"}");
            Ok(Box::pin(tokio_stream::iter(vec![Ok(message)])))
        }

        async fn call(
            &self,
            _agent: &Agent,
            _context: &ToolCallContext,
            _call: ToolCallFull,
        ) -> ToolResult {
            unreachable!("Not used in tests")
        }

        async fn update(&self, _conversation: Conversation) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_generate_keeps_effort_for_claude_sonnet_4_5_title_requests() {
        let captured_context = Arc::new(Mutex::new(None));
        let service = Arc::new(MockAgentService { captured_context: captured_context.clone() });
        let generator = TitleGenerator::new(
            service,
            UserPrompt::from("Generate a title".to_string()),
            ModelId::from("claude-sonnet-4-5"),
            Some(ProviderId::OPENCODE_ZEN),
        )
        .reasoning(Some(
            ReasoningConfig::default()
                .enabled(true)
                .effort(Effort::High),
        ));

        let actual = generator.generate().await.unwrap();
        let actual_context = captured_context.lock().unwrap().clone().unwrap();
        let expected_reasoning = Some(
            ReasoningConfig::default()
                .enabled(true)
                .effort(Effort::High),
        );
        let expected_title = Some("fixture title".to_string());

        assert_eq!(actual, expected_title);
        assert_eq!(actual_context.reasoning, expected_reasoning);
    }
}
