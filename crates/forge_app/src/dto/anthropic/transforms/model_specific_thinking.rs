use forge_domain::Transformer;
use tracing::warn;

use crate::dto::anthropic::{OutputConfig, OutputEffort, Request, Thinking, ThinkingDisplay};

/// Default budget applied when converting `Adaptive` thinking to the legacy
/// `Enabled` shape for pre-4.6 models that don't accept adaptive.
const DEFAULT_LEGACY_BUDGET_TOKENS: u64 = 10000;

/// Rewrites the `thinking` and `output_config` fields and strips rejected
/// sampling parameters to match the target model's API contract. Classifies
/// models into four tiers based on the official Anthropic docs:
///
/// | Tier                | Matching ids                   | Thinking          | Effort     | xhigh | max |
/// |---------------------|--------------------------------|-------------------|------------|-------|-----|
/// | `AdaptiveOnly`      | `opus-4-7`                     | adaptive only     | ✓          | ✓     | ✓   |
/// | `AdaptiveFriendly`  | `opus-4-6`, `sonnet-4-6`       | both shapes       | ✓          | →max  | ✓   |
/// | `LegacyWithEffort`  | `opus-4-5`                     | enabled only      | ✓          | →high | →high |
/// | `LegacyNoEffort`    | everything else                | enabled only      | drop       | drop  | drop |
///
/// Additional behaviour:
/// - `AdaptiveOnly` strips `temperature`/`top_p`/`top_k` (4.7 rejects non-
///   defaults) and warns when a caller-supplied `budget_tokens` is dropped.
/// - The display preference carried from the caller's `ReasoningConfig`
///   (`exclude: true` → `Omitted`) is applied when rewriting `Enabled` to
///   `Adaptive` on 4.7.
pub struct ModelSpecificThinking {
    model_id: String,
    /// Defaults to `Summarized` because the legacy `Enabled` shape always
    /// produced visible reasoning; a 4.7 migration should preserve that unless
    /// the caller opts out via `exclude`.
    display: ThinkingDisplay,
}

impl ModelSpecificThinking {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            display: ThinkingDisplay::Summarized,
        }
    }

    pub fn display(mut self, display: ThinkingDisplay) -> Self {
        self.display = display;
        self
    }

    fn family(&self) -> ModelFamily {
        let id = self.model_id.to_lowercase();
        if id.contains("opus-4-7") {
            ModelFamily::AdaptiveOnly
        } else if id.contains("opus-4-6") || id.contains("sonnet-4-6") {
            ModelFamily::AdaptiveFriendly
        } else if id.contains("opus-4-5") {
            ModelFamily::LegacyWithEffort
        } else {
            ModelFamily::LegacyNoEffort
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ModelFamily {
    AdaptiveOnly,
    AdaptiveFriendly,
    LegacyWithEffort,
    LegacyNoEffort,
}

/// Falls back to `max` — the closest supported level, even though it spends
/// more tokens than `xhigh` would have.
fn replace_xhigh_with_max(output_config: &mut Option<OutputConfig>) {
    if let Some(config) = output_config.as_mut()
        && config.effort == OutputEffort::XHigh
    {
        config.effort = OutputEffort::Max;
    }
}

fn clamp_effort_to_high(output_config: &mut Option<OutputConfig>) {
    if let Some(config) = output_config.as_mut()
        && matches!(config.effort, OutputEffort::XHigh | OutputEffort::Max)
    {
        config.effort = OutputEffort::High;
    }
}

impl Transformer for ModelSpecificThinking {
    type Value = Request;

    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        match self.family() {
            ModelFamily::AdaptiveOnly => {
                if let Some(Thinking::Enabled { budget_tokens }) = request.thinking {
                    warn!(
                        model = %self.model_id,
                        dropped_budget_tokens = budget_tokens,
                        "Dropping `reasoning.max_tokens` for Opus 4.7: extended thinking \
                         budgets are unsupported. Use `reasoning.effort` to control \
                         thinking depth instead."
                    );
                    request.thinking = Some(Thinking::Adaptive {
                        display: Some(self.display),
                    });
                }
                request.temperature = None;
                request.top_p = None;
                request.top_k = None;
            }
            ModelFamily::LegacyWithEffort => {
                if let Some(Thinking::Adaptive { .. }) = request.thinking {
                    request.thinking = Some(Thinking::Enabled {
                        budget_tokens: DEFAULT_LEGACY_BUDGET_TOKENS,
                    });
                }
                clamp_effort_to_high(&mut request.output_config);
            }
            ModelFamily::LegacyNoEffort => {
                if let Some(Thinking::Adaptive { .. }) = request.thinking {
                    request.thinking = Some(Thinking::Enabled {
                        budget_tokens: DEFAULT_LEGACY_BUDGET_TOKENS,
                    });
                }
                if request.output_config.is_some() {
                    warn!(
                        model = %self.model_id,
                        "Dropping `output_config.effort`: the effort parameter is only \
                         supported on Opus 4.5, Opus 4.6, Sonnet 4.6, and Opus 4.7."
                    );
                    request.output_config = None;
                }
            }
            ModelFamily::AdaptiveFriendly => {
                replace_xhigh_with_max(&mut request.output_config);
            }
        }
        request
    }
}

#[cfg(test)]
mod tests {
    use forge_domain::Transformer;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::dto::anthropic::{Request, Thinking};

    fn fixture_request_with_thinking(thinking: Thinking) -> Request {
        Request::default()
            .thinking(thinking)
            .temperature(0.5f32)
            .top_p(0.9f32)
            .top_k(40u64)
    }

    #[test]
    fn test_opus_4_7_rewrites_enabled_to_adaptive_with_summarized_display() {
        let fixture = fixture_request_with_thinking(Thinking::Enabled { budget_tokens: 8000 });
        let actual = ModelSpecificThinking::new("claude-opus-4-7").transform(fixture);

        assert_eq!(
            actual.thinking,
            Some(Thinking::Adaptive { display: Some(ThinkingDisplay::Summarized) })
        );
    }

    #[test]
    fn test_opus_4_7_strips_sampling_params() {
        let fixture = fixture_request_with_thinking(Thinking::Enabled { budget_tokens: 8000 });
        let actual = ModelSpecificThinking::new("claude-opus-4-7").transform(fixture);

        assert_eq!(actual.temperature, None);
        assert_eq!(actual.top_p, None);
        assert_eq!(actual.top_k, None);
    }

    #[test]
    fn test_opus_4_7_strips_sampling_params_even_without_thinking() {
        let fixture = Request::default().temperature(0.5f32).top_p(0.9f32).top_k(40u64);
        let actual = ModelSpecificThinking::new("claude-opus-4-7").transform(fixture);

        assert_eq!(actual.temperature, None);
        assert_eq!(actual.top_p, None);
        assert_eq!(actual.top_k, None);
    }

    #[test]
    fn test_opus_4_7_bedrock_prefix_still_matches() {
        let fixture = fixture_request_with_thinking(Thinking::Enabled { budget_tokens: 8000 });
        let actual =
            ModelSpecificThinking::new("us.anthropic.claude-opus-4-7").transform(fixture);

        assert_eq!(
            actual.thinking,
            Some(Thinking::Adaptive { display: Some(ThinkingDisplay::Summarized) })
        );
        assert_eq!(actual.temperature, None);
    }

    #[test]
    fn test_opus_4_7_passes_through_adaptive_with_existing_display() {
        let fixture = fixture_request_with_thinking(Thinking::Adaptive {
            display: Some(ThinkingDisplay::Omitted),
        });
        let actual = ModelSpecificThinking::new("claude-opus-4-7").transform(fixture);

        assert_eq!(
            actual.thinking,
            Some(Thinking::Adaptive { display: Some(ThinkingDisplay::Omitted) })
        );
    }

    #[test]
    fn test_opus_4_6_passes_both_shapes_through() {
        let fixture1 = fixture_request_with_thinking(Thinking::Enabled { budget_tokens: 8000 });
        let actual1 = ModelSpecificThinking::new("claude-opus-4-6").transform(fixture1);
        assert_eq!(actual1.thinking, Some(Thinking::Enabled { budget_tokens: 8000 }));
        assert_eq!(actual1.temperature, Some(0.5));

        let fixture2 = fixture_request_with_thinking(Thinking::Adaptive {
            display: Some(ThinkingDisplay::Summarized),
        });
        let actual2 = ModelSpecificThinking::new("claude-sonnet-4-6").transform(fixture2);
        assert_eq!(
            actual2.thinking,
            Some(Thinking::Adaptive { display: Some(ThinkingDisplay::Summarized) })
        );
    }

    #[test]
    fn test_opus_4_5_rewrites_adaptive_to_enabled() {
        let fixture = fixture_request_with_thinking(Thinking::Adaptive {
            display: Some(ThinkingDisplay::Summarized),
        });
        let actual = ModelSpecificThinking::new("claude-opus-4-5-20251101").transform(fixture);

        assert_eq!(
            actual.thinking,
            Some(Thinking::Enabled { budget_tokens: DEFAULT_LEGACY_BUDGET_TOKENS })
        );
    }

    #[test]
    fn test_legacy_no_effort_model_passes_enabled_through() {
        let fixture = fixture_request_with_thinking(Thinking::Enabled { budget_tokens: 8000 });
        let actual = ModelSpecificThinking::new("claude-3-7-sonnet-20250219").transform(fixture);

        assert_eq!(actual.thinking, Some(Thinking::Enabled { budget_tokens: 8000 }));
        assert_eq!(actual.temperature, Some(0.5));
    }

    #[test]
    fn test_no_thinking_is_preserved_everywhere() {
        for model in [
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-opus-4-5-20251101",
            "claude-3-7-sonnet-20250219",
        ] {
            let fixture = Request::default();
            let actual = ModelSpecificThinking::new(model).transform(fixture);
            assert_eq!(actual.thinking, None, "model {}", model);
        }
    }

    #[test]
    fn test_opus_4_7_uses_configured_display_when_rewriting_enabled() {
        // The caller's `exclude: true` preference (→ `Omitted` display) must
        // survive the 4.7 `Enabled → Adaptive` rewrite; otherwise hidden
        // reasoning would silently become visible.
        let fixture = fixture_request_with_thinking(Thinking::Enabled { budget_tokens: 8000 });
        let actual = ModelSpecificThinking::new("claude-opus-4-7")
            .display(ThinkingDisplay::Omitted)
            .transform(fixture);

        assert_eq!(
            actual.thinking,
            Some(Thinking::Adaptive { display: Some(ThinkingDisplay::Omitted) })
        );
    }

    #[test]
    fn test_opus_4_7_preserves_effort_when_dropping_budget() {
        // When both `max_tokens` and `effort` are set on 4.7, dropping the
        // budget must not take the effort signal with it — effort is the only
        // remaining depth knob on 4.7.
        let fixture = Request::default()
            .thinking(Thinking::Enabled { budget_tokens: 8000 })
            .output_config(OutputConfig { effort: OutputEffort::XHigh });
        let actual = ModelSpecificThinking::new("claude-opus-4-7").transform(fixture);

        assert_eq!(
            actual.thinking,
            Some(Thinking::Adaptive { display: Some(ThinkingDisplay::Summarized) })
        );
        assert_eq!(
            actual.output_config,
            Some(OutputConfig { effort: OutputEffort::XHigh })
        );
    }

    #[test]
    fn test_opus_4_7_preserves_xhigh_effort() {
        let fixture = Request::default().output_config(OutputConfig { effort: OutputEffort::XHigh });
        let actual = ModelSpecificThinking::new("claude-opus-4-7").transform(fixture);

        assert_eq!(
            actual.output_config,
            Some(OutputConfig { effort: OutputEffort::XHigh })
        );
    }

    #[test]
    fn test_opus_4_6_replaces_xhigh_with_max() {
        let fixture = Request::default().output_config(OutputConfig { effort: OutputEffort::XHigh });
        let actual = ModelSpecificThinking::new("claude-opus-4-6").transform(fixture);

        assert_eq!(
            actual.output_config,
            Some(OutputConfig { effort: OutputEffort::Max })
        );
    }

    #[test]
    fn test_opus_4_5_clamps_xhigh_to_high() {
        // Opus 4.5 supports effort but not xhigh or max; clamp to high.
        let fixture = Request::default().output_config(OutputConfig { effort: OutputEffort::XHigh });
        let actual = ModelSpecificThinking::new("claude-opus-4-5-20251101").transform(fixture);

        assert_eq!(
            actual.output_config,
            Some(OutputConfig { effort: OutputEffort::High })
        );
    }

    #[test]
    fn test_opus_4_5_clamps_max_to_high() {
        let fixture = Request::default().output_config(OutputConfig { effort: OutputEffort::Max });
        let actual = ModelSpecificThinking::new("claude-opus-4-5-20251101").transform(fixture);

        assert_eq!(
            actual.output_config,
            Some(OutputConfig { effort: OutputEffort::High })
        );
    }

    #[test]
    fn test_opus_4_5_preserves_supported_effort_levels() {
        for level in [OutputEffort::Low, OutputEffort::Medium, OutputEffort::High] {
            let fixture = Request::default().output_config(OutputConfig { effort: level });
            let actual =
                ModelSpecificThinking::new("claude-opus-4-5-20251101").transform(fixture);
            assert_eq!(
                actual.output_config,
                Some(OutputConfig { effort: level }),
                "level {:?}",
                level
            );
        }
    }

    #[test]
    fn test_legacy_no_effort_models_drop_output_config() {
        for model in [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
        ] {
            let fixture = Request::default().output_config(OutputConfig { effort: OutputEffort::High });
            let actual = ModelSpecificThinking::new(model).transform(fixture);
            assert_eq!(actual.output_config, None, "model {}", model);
        }
    }

    #[test]
    fn test_adaptive_friendly_preserves_high_effort() {
        let fixture = Request::default().output_config(OutputConfig { effort: OutputEffort::High });
        let actual = ModelSpecificThinking::new("claude-opus-4-6").transform(fixture);

        assert_eq!(
            actual.output_config,
            Some(OutputConfig { effort: OutputEffort::High })
        );
    }
}
