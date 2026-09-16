//! Validated command-line model for the CUDA Nova smoke and profile runner.

use std::{num::NonZeroUsize, path::PathBuf};

use thiserror::Error;

/// The audited step counts required by the backend performance plan.
const STANDARD_PROFILE_STEPS: [usize; 4] = [1, 8, 64, 1_024];

/// The bounded step counts retained for quick smoke measurements.
const SMOKE_PROFILE_STEPS: [usize; 4] = [1, 2, 8, 16];

/// Maximum recursive length accepted from an untrusted command line.
const MAX_PROFILE_STEPS: usize = 1_024;

/// Classifies how a validated forward-profile step plan was selected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ForwardProfileMode {
    /// The audited `1/8/64/1024` benchmark sequence.
    Standard,
    /// The quick `1/2/8/16` smoke sequence.
    Smoke,
    /// A caller-supplied strictly increasing sequence.
    Custom,
}

impl ForwardProfileMode {
    /// Returns the stable receipt label for this profile mode.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Smoke => "smoke",
            Self::Custom => "custom",
        }
    }
}

/// A non-empty, strictly increasing recursive-step plan and its receipt metadata.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ForwardProfileRequest {
    /// Origin of the selected step sequence.
    mode: ForwardProfileMode,
    /// Positive recursive lengths measured in ascending order.
    step_counts: Vec<NonZeroUsize>,
    /// Largest recursive length, stored separately to preserve non-emptiness.
    maximum_step_count: NonZeroUsize,
    /// Optional create-new destination for the structured receipt.
    receipt_path: Option<PathBuf>,
    /// Optional source revision supplied by the benchmark operator.
    source_revision: Option<String>,
}

impl ForwardProfileRequest {
    /// Returns the stable receipt label for the selected mode.
    pub(crate) const fn mode_label(&self) -> &'static str {
        self.mode.label()
    }

    /// Returns the positive recursive lengths in ascending order.
    pub(crate) fn step_counts(&self) -> &[NonZeroUsize] {
        &self.step_counts
    }

    /// Returns the largest recursive length needed by the fixture generator.
    pub(crate) const fn maximum_step_count(&self) -> NonZeroUsize {
        self.maximum_step_count
    }

    /// Returns the optional create-new receipt destination.
    pub(crate) fn receipt_path(&self) -> Option<&std::path::Path> {
        self.receipt_path.as_deref()
    }

    /// Returns the optional hexadecimal source revision.
    pub(crate) fn source_revision(&self) -> Option<&str> {
        self.source_revision.as_deref()
    }

    /// Constructs a request after proving positivity, non-emptiness, and order.
    fn new(
        mode: ForwardProfileMode,
        step_counts: Vec<NonZeroUsize>,
        receipt_path: Option<PathBuf>,
        source_revision: Option<String>,
    ) -> Result<Self, CliError> {
        if !is_strictly_increasing(&step_counts) {
            return Err(CliError::NonIncreasingStepPlan);
        }
        if let Some(value) = step_counts
            .iter()
            .map(|step_count| step_count.get())
            .find(|value| *value > MAX_PROFILE_STEPS)
        {
            return Err(CliError::StepCountExceedsLimit {
                value,
                maximum: MAX_PROFILE_STEPS,
            });
        }
        let maximum_step_count = step_counts
            .iter()
            .copied()
            .last()
            .ok_or(CliError::EmptyStepPlan)?;
        Ok(Self {
            mode,
            step_counts,
            maximum_step_count,
            receipt_path,
            source_revision,
        })
    }
}

/// Fully validated options consumed by the runner's effectful boundary.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct RunnerOptions {
    /// Whether to execute the fixed topology and weighted-forward proof smoke.
    should_prove: bool,
    /// Optional reusable-parameter forward profile request.
    profile: Option<ForwardProfileRequest>,
    /// Optional trusted Powers-of-Tau directory for a `HyperKZG` build.
    ptau_dir: Option<PathBuf>,
}

impl RunnerOptions {
    /// Returns whether the proof smoke path is enabled.
    pub(crate) const fn should_prove(&self) -> bool {
        self.should_prove
    }

    /// Returns the validated profile request, when profiling was requested.
    pub(crate) const fn profile(&self) -> Option<&ForwardProfileRequest> {
        self.profile.as_ref()
    }

    /// Returns the optional trusted Powers-of-Tau directory.
    pub(crate) fn ptau_dir(&self) -> Option<&std::path::Path> {
        self.ptau_dir.as_deref()
    }
}

/// Command-line validation failures detected before CUDA initialization.
#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum CliError {
    /// An option that may appear once was repeated.
    #[error("command-line option {argument} was supplied more than once")]
    DuplicateArgument {
        /// Stable spelling of the repeated option.
        argument: &'static str,
    },
    /// An option requiring a following value reached the end of the arguments.
    #[error("command-line option {argument} requires a value")]
    MissingValue {
        /// Stable spelling of the incomplete option.
        argument: &'static str,
    },
    /// A profile mode was neither `standard` nor `smoke`.
    #[error("unknown forward-profile mode {value}; expected standard or smoke")]
    InvalidProfileMode {
        /// Unrecognized mode supplied by the caller.
        value: String,
    },
    /// One comma-separated step value was not a positive host integer.
    #[error("invalid positive forward-profile step count {value}")]
    InvalidStepCount {
        /// Invalid step token supplied by the caller.
        value: String,
    },
    /// A custom profile contained no step counts.
    #[error("the forward-profile step plan must not be empty")]
    EmptyStepPlan,
    /// A custom profile was not strictly increasing.
    #[error("forward-profile step counts must be unique and strictly increasing")]
    NonIncreasingStepPlan,
    /// A custom profile exceeded the bounded fixture capacity.
    #[error("forward-profile step count {value} exceeds the supported maximum {maximum}")]
    StepCountExceedsLimit {
        /// Requested recursive length.
        value: usize,
        /// Largest recursive length supported by this runner.
        maximum: usize,
    },
    /// Receipt metadata was supplied without enabling the profile.
    #[error("command-line option {argument} requires --profile-forward")]
    ProfileOptionWithoutProfile {
        /// Profile-only option supplied by the caller.
        argument: &'static str,
    },
    /// A path-valued option received an empty path.
    #[error("command-line option {argument} requires a non-empty path")]
    EmptyPath {
        /// Stable spelling of the path-valued option.
        argument: &'static str,
    },
    /// A source revision was not a canonical abbreviated or full hexadecimal ID.
    #[error("profile source revision must contain 7 to 64 hexadecimal characters")]
    InvalidSourceRevision,
    /// The runner does not recognize the supplied argument.
    #[error("unknown command-line argument {argument}")]
    UnknownArgument {
        /// Unrecognized argument supplied by the caller.
        argument: String,
    },
}

/// Partially parsed options before cross-option validation.
#[derive(Debug, Default)]
struct ArgumentAccumulator {
    /// Whether the fixed proof smoke was requested.
    should_prove: bool,
    /// Optional named profile preset.
    profile_mode: Option<ForwardProfileMode>,
    /// Optional custom comma-separated step specification.
    profile_steps: Option<String>,
    /// Optional structured receipt destination.
    receipt_path: Option<PathBuf>,
    /// Optional benchmark source revision.
    source_revision: Option<String>,
    /// Optional trusted Powers-of-Tau directory.
    ptau_dir: Option<PathBuf>,
}

impl ArgumentAccumulator {
    /// Consumes one standalone argument and any required following value.
    fn consume<'a, I>(&mut self, argument: &str, remaining: &mut I) -> Result<(), CliError>
    where
        I: Iterator<Item = &'a String>,
    {
        match argument {
            "--prove" => set_once(&mut self.should_prove, "--prove"),
            "--profile-forward" => set_option_once(
                &mut self.profile_mode,
                ForwardProfileMode::Standard,
                "--profile-forward",
            ),
            "--profile-steps" => {
                let value = required_value(remaining, "--profile-steps")?;
                set_option_once(&mut self.profile_steps, value.to_owned(), "--profile-steps")
            }
            "--profile-json" => {
                let value = required_path(remaining, "--profile-json")?;
                set_option_once(&mut self.receipt_path, value, "--profile-json")
            }
            "--profile-revision" => {
                let value = required_value(remaining, "--profile-revision")?;
                set_option_once(
                    &mut self.source_revision,
                    validate_revision(value)?,
                    "--profile-revision",
                )
            }
            "--ptau-dir" => {
                let value = required_path(remaining, "--ptau-dir")?;
                set_option_once(&mut self.ptau_dir, value, "--ptau-dir")
            }
            value => self.consume_attached_or_unknown(value),
        }
    }

    /// Consumes one `--option=value` argument or returns an unknown-option error.
    fn consume_attached_or_unknown(&mut self, argument: &str) -> Result<(), CliError> {
        let Some((name, value)) = argument.split_once('=') else {
            return Err(CliError::UnknownArgument {
                argument: argument.to_owned(),
            });
        };
        match name {
            "--profile-forward" => set_option_once(
                &mut self.profile_mode,
                parse_profile_mode(value)?,
                "--profile-forward",
            ),
            "--profile-steps" => {
                set_option_once(&mut self.profile_steps, value.to_owned(), "--profile-steps")
            }
            "--profile-json" => set_option_once(
                &mut self.receipt_path,
                validated_path(value, "--profile-json")?,
                "--profile-json",
            ),
            "--profile-revision" => set_option_once(
                &mut self.source_revision,
                validate_revision(value)?,
                "--profile-revision",
            ),
            "--ptau-dir" => set_option_once(
                &mut self.ptau_dir,
                validated_path(value, "--ptau-dir")?,
                "--ptau-dir",
            ),
            _ => Err(CliError::UnknownArgument {
                argument: argument.to_owned(),
            }),
        }
    }

    /// Validates cross-option relationships and produces the execution model.
    fn finish(self) -> Result<RunnerOptions, CliError> {
        let profile = build_profile_request(
            self.profile_mode,
            self.profile_steps,
            self.receipt_path,
            self.source_revision,
        )?;
        Ok(RunnerOptions {
            should_prove: self.should_prove,
            profile,
            ptau_dir: self.ptau_dir,
        })
    }
}

/// Parses all runner arguments into a validated execution model.
///
/// `--profile-forward` selects the standard `1/8/64/1024` plan.
/// `--profile-forward=smoke` retains the previous `1/2/8/16` path, and
/// `--profile-steps` replaces either preset with a custom increasing plan.
pub(crate) fn parse_arguments(arguments: &[String]) -> Result<RunnerOptions, CliError> {
    let mut accumulator = ArgumentAccumulator::default();
    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        accumulator.consume(argument, &mut remaining)?;
    }
    accumulator.finish()
}

/// Marks a boolean flag once and rejects a duplicate spelling.
fn set_once(target: &mut bool, argument: &'static str) -> Result<(), CliError> {
    if *target {
        return Err(CliError::DuplicateArgument { argument });
    }
    *target = true;
    Ok(())
}

/// Assigns one optional value and rejects repeated options.
fn set_option_once<T>(
    target: &mut Option<T>,
    value: T,
    argument: &'static str,
) -> Result<(), CliError> {
    if target.is_some() {
        return Err(CliError::DuplicateArgument { argument });
    }
    *target = Some(value);
    Ok(())
}

/// Takes the required value following one path-independent option.
fn required_value<'a, I>(remaining: &mut I, argument: &'static str) -> Result<&'a str, CliError>
where
    I: Iterator<Item = &'a String>,
{
    remaining
        .next()
        .map(String::as_str)
        .ok_or(CliError::MissingValue { argument })
}

/// Takes and validates the required value following one path option.
fn required_path<'a, I>(remaining: &mut I, argument: &'static str) -> Result<PathBuf, CliError>
where
    I: Iterator<Item = &'a String>,
{
    validated_path(required_value(remaining, argument)?, argument)
}

/// Rejects an empty path before it reaches filesystem effects.
fn validated_path(value: &str, argument: &'static str) -> Result<PathBuf, CliError> {
    if value.is_empty() {
        return Err(CliError::EmptyPath { argument });
    }
    Ok(PathBuf::from(value))
}

/// Parses one named profile preset.
fn parse_profile_mode(value: &str) -> Result<ForwardProfileMode, CliError> {
    match value {
        "standard" => Ok(ForwardProfileMode::Standard),
        "smoke" => Ok(ForwardProfileMode::Smoke),
        _ => Err(CliError::InvalidProfileMode {
            value: value.to_owned(),
        }),
    }
}

/// Builds the profile request only after all cross-option invariants hold.
fn build_profile_request(
    profile_mode: Option<ForwardProfileMode>,
    profile_steps: Option<String>,
    receipt_path: Option<PathBuf>,
    source_revision: Option<String>,
) -> Result<Option<ForwardProfileRequest>, CliError> {
    let Some(selected_mode) = profile_mode else {
        if profile_steps.is_some() {
            return Err(CliError::ProfileOptionWithoutProfile {
                argument: "--profile-steps",
            });
        }
        if receipt_path.is_some() {
            return Err(CliError::ProfileOptionWithoutProfile {
                argument: "--profile-json",
            });
        }
        if source_revision.is_some() {
            return Err(CliError::ProfileOptionWithoutProfile {
                argument: "--profile-revision",
            });
        }
        return Ok(None);
    };

    let (mode, step_counts) = if let Some(step_specification) = profile_steps {
        (
            ForwardProfileMode::Custom,
            parse_step_specification(&step_specification)?,
        )
    } else {
        (selected_mode, preset_steps(selected_mode)?)
    };
    ForwardProfileRequest::new(mode, step_counts, receipt_path, source_revision).map(Some)
}

/// Converts a preset into positive step counts without unchecked construction.
fn preset_steps(mode: ForwardProfileMode) -> Result<Vec<NonZeroUsize>, CliError> {
    let values = match mode {
        ForwardProfileMode::Standard => STANDARD_PROFILE_STEPS.as_slice(),
        ForwardProfileMode::Smoke => SMOKE_PROFILE_STEPS.as_slice(),
        ForwardProfileMode::Custom => return Err(CliError::EmptyStepPlan),
    };
    values
        .iter()
        .copied()
        .map(|value| {
            NonZeroUsize::new(value).ok_or_else(|| CliError::InvalidStepCount {
                value: value.to_string(),
            })
        })
        .collect()
}

/// Parses a comma-separated positive step sequence.
fn parse_step_specification(value: &str) -> Result<Vec<NonZeroUsize>, CliError> {
    if value.is_empty() {
        return Err(CliError::EmptyStepPlan);
    }
    value
        .split(',')
        .map(|token| {
            let parsed = token
                .parse::<usize>()
                .map_err(|_| CliError::InvalidStepCount {
                    value: token.to_owned(),
                })?;
            NonZeroUsize::new(parsed).ok_or_else(|| CliError::InvalidStepCount {
                value: token.to_owned(),
            })
        })
        .collect()
}

/// Returns whether each adjacent pair is strictly increasing.
fn is_strictly_increasing(values: &[NonZeroUsize]) -> bool {
    values.windows(2).all(|pair| match pair {
        [left, right] => left < right,
        _ => true,
    })
}

/// Validates a reproducible abbreviated or full hexadecimal source revision.
fn validate_revision(value: &str) -> Result<String, CliError> {
    let has_valid_length = (7..=64).contains(&value.len());
    let is_hexadecimal = value.bytes().all(|byte| byte.is_ascii_hexdigit());
    if !has_valid_length || !is_hexadecimal {
        return Err(CliError::InvalidSourceRevision);
    }
    Ok(value.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::{CliError, ForwardProfileMode, parse_arguments};

    /// Converts string literals into the owned argument representation.
    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn standard_profile_covers_the_audited_recursive_lengths() {
        let parsed = parse_arguments(&arguments(&["--profile-forward"]));
        assert!(parsed.is_ok());
        if let Ok(options) = parsed {
            let profile = options.profile();
            assert!(profile.is_some());
            if let Some(profile) = profile {
                let steps = profile
                    .step_counts()
                    .iter()
                    .map(|value| value.get())
                    .collect::<Vec<_>>();
                assert_eq!(steps, vec![1, 8, 64, 1_024]);
                assert_eq!(profile.mode, ForwardProfileMode::Standard);
            }
        }
    }

    #[test]
    fn smoke_profile_preserves_the_bounded_legacy_plan() {
        let parsed = parse_arguments(&arguments(&["--profile-forward=smoke"]));
        assert!(parsed.is_ok());
        if let Ok(options) = parsed
            && let Some(profile) = options.profile()
        {
            let steps = profile
                .step_counts()
                .iter()
                .map(|value| value.get())
                .collect::<Vec<_>>();
            assert_eq!(steps, vec![1, 2, 8, 16]);
            assert_eq!(profile.mode, ForwardProfileMode::Smoke);
        }
    }

    #[test]
    fn custom_profile_requires_positive_strictly_increasing_steps() {
        let parsed = parse_arguments(&arguments(&["--profile-forward", "--profile-steps=1,4,32"]));
        assert!(parsed.is_ok());
        if let Ok(options) = parsed
            && let Some(profile) = options.profile()
        {
            assert_eq!(profile.mode, ForwardProfileMode::Custom);
            assert_eq!(profile.maximum_step_count().get(), 32);
        }

        assert!(matches!(
            parse_arguments(&arguments(&["--profile-forward", "--profile-steps=1,0"])),
            Err(CliError::InvalidStepCount { .. })
        ));
        assert_eq!(
            parse_arguments(&arguments(&["--profile-forward", "--profile-steps=1,8,8"])),
            Err(CliError::NonIncreasingStepPlan)
        );
        assert_eq!(
            parse_arguments(&arguments(&["--profile-forward", "--profile-steps=8,1"])),
            Err(CliError::NonIncreasingStepPlan)
        );
        assert_eq!(
            parse_arguments(&arguments(&["--profile-forward", "--profile-steps=1,1025"])),
            Err(CliError::StepCountExceedsLimit {
                value: 1_025,
                maximum: 1_024,
            })
        );
    }

    #[test]
    fn receipt_options_require_a_profile_and_validate_the_revision() {
        assert_eq!(
            parse_arguments(&arguments(&["--profile-json", "receipt.json"])),
            Err(CliError::ProfileOptionWithoutProfile {
                argument: "--profile-json"
            })
        );
        assert_eq!(
            parse_arguments(&arguments(&[
                "--profile-forward",
                "--profile-revision=not-a-commit"
            ])),
            Err(CliError::InvalidSourceRevision)
        );

        let parsed = parse_arguments(&arguments(&[
            "--profile-forward=smoke",
            "--profile-json=receipt.json",
            "--profile-revision=65738A2",
        ]));
        assert!(parsed.is_ok());
        if let Ok(options) = parsed
            && let Some(profile) = options.profile()
        {
            assert_eq!(profile.source_revision(), Some("65738a2"));
            assert_eq!(
                profile.receipt_path(),
                Some(std::path::Path::new("receipt.json"))
            );
        }
    }
}
