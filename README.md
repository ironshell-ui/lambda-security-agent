# Lambda Security Agent

Defense-in-depth purple agent for the Lambda Agent Security Custom Track (AgentX-AgentBeats).

## Architecture

```
User Input → [1.Decode] → [2.Classify] → [3.Entropy] → [4.Policy] ─┐
                                                                     ↓
Tool Response → [6.Validate] ──────────────────────────────→ [7.Decision Gate]
                                                                     ↓
Agent Output → [4.Policy] → [5.PII Scan] ──────────────────→ safe / caution / block
```

### 7 Stages

1. **Input Decoding**: Unicode NFKC, zero-width stripping, base64/hex/URL decode, ANSI removal
2. **Threat Classification**: 71 OWASP-aligned patterns across 7 categories (prompt injection, info disclosure, tool abuse, encoding evasion, social engineering, insecure output, PII extraction)
3. **Entropy Analysis**: Shannon character entropy for adversarial suffix detection (Jain et al. 2023)
4. **Policy Verification**: 7 policy surfaces (access, privacy, disclosure, process, safety, governance, ambiguity) with configurable require/forbid rules
5. **Output Validation**: PII scanning (SSN, credit cards, API keys, JWT, private keys, connection strings, internal paths)
6. **Tool Response Validation**: Pre-execution anomaly detection for error-injected tool responses
7. **Decision Gate**: Aggregation with RESTRAINT mode — blocks attacks while allowing benign requests through

### Key Features

- **RESTRAINT mode**: Calibrated to NOT over-refuse benign requests
- **Pre-execution validation**: Detects tool errors BEFORE acting on them
- **Ambiguity handling**: Acknowledges uncertainty instead of guessing
- **Encoding decode**: Normalizes obfuscated inputs before classification
- **Zero latency overhead**: All checks are regex-based, <1ms per stage

## Quick Start

```bash
docker build -t lambda-security-agent .
docker run -p 9009:9009 -e OPENAI_API_KEY=your_key lambda-security-agent
```

## Evaluation

Designed for three green agents:
- **Pi-Bench**: Policy compliance across 9 dimensions
- **NAAMSE**: Adversarial defense against 128K+ mutated prompts
- **AVER**: Error detection and recovery in tool responses

## References

- OWASP Top 10 for LLM Applications (2023)
- Yi et al. (2023) "Benchmarking and Defending Against Indirect Prompt Injection"
- Jain et al. (2023) "Baseline Defenses for Adversarial Attacks Against Aligned LLMs"
- Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"

## License

MIT
