import asyncio
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, RunUsage, UsageLimits
from pydantic_ai.models.groq import GroqModel

from crypto_tools import get_ta_ema

load_dotenv()

model = GroqModel("openai/gpt-oss-120b")


# ------------------------------------------------------------------
# Shared dependencies
# ------------------------------------------------------------------

@dataclass
class TradingDeps:
    user_id: str
    paper_trading: bool = True

    # Hard application-level limits
    max_risk_per_trade_pct: float = 1.0
    minimum_reward_risk_ratio: float = 2.0

    # Add shared production dependencies here:
    # market_data_client: MarketDataClient
    # portfolio_repository: PortfolioRepository
    # exchange_client: ExchangeClient


# ------------------------------------------------------------------
# Structured outputs
# ------------------------------------------------------------------

class TechnicalReport(BaseModel):
    symbol: str
    timeframe: str
    trend: Literal["bullish", "bearish", "sideways"]
    fast_ema: float | None = None
    slow_ema: float | None = None
    observations: list[str]
    invalidation_conditions: list[str]


class TradeDraft(BaseModel):
    symbol: str
    timeframe: str
    action: Literal["buy", "sell", "hold"]

    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    risk_per_trade_pct: float = Field(ge=0, le=100)
    rationale: list[str]
    warnings: list[str] = []


class RiskReview(BaseModel):
    approved: bool
    concerns: list[str]
    recommended_changes: list[str]


class HardRiskGate(BaseModel):
    approved: bool
    reasons: list[str]


class TradingResponse(BaseModel):
    draft: TradeDraft
    risk_review: RiskReview
    hard_risk_gate: HardRiskGate
    executable: bool


# ------------------------------------------------------------------
# Specialist agent: technical analysis
# ------------------------------------------------------------------

technical_analysis_agent = Agent(
    model,
    name="technical_analysis_agent",
    deps_type=TradingDeps,
    output_type=TechnicalReport,
    tools=[get_ta_ema],
    instructions="""
You are a technical-analysis specialist.

Use the available market-data and indicator tools. Do not invent prices or
indicator values.

Return:
- the current trend,
- relevant EMA values,
- observations supporting the trend,
- conditions that would invalidate the analysis.

Do not make portfolio-level risk decisions and do not place trades.
""",
)


# ------------------------------------------------------------------
# Coordinator agent
# ------------------------------------------------------------------

trading_coordinator = Agent(
    model,
    name="trading_coordinator",
    deps_type=TradingDeps,
    output_type=TradeDraft,
    instructions="""
You coordinate trading analysis.

For instrument-specific requests:
1. Call get_technical_analysis.
2. Construct a proposed trade plan from the returned evidence.
3. Return HOLD when the evidence is incomplete, contradictory, or weak.

Never claim that an order was executed.
Never bypass risk review.
Never invent market prices.

For general questions such as "how to trade", provide an educational HOLD
plan rather than pretending that a specific market setup exists.
""",
)


@trading_coordinator.tool
async def get_technical_analysis(
    ctx: RunContext[TradingDeps],
    symbol: str,
    timeframe: str,
    fast_period: int = 20,
    slow_period: int = 50,
) -> TechnicalReport:
    """Ask the technical-analysis specialist to analyze a market.

    Args:
        symbol: Trading symbol, for example BTCUSDT.
        timeframe: Chart timeframe, for example 1h or 4h.
        fast_period: Period for the faster EMA.
        slow_period: Period for the slower EMA.
    """
    result = await technical_analysis_agent.run(
        (
            f"Analyze {symbol} on the {timeframe} timeframe. "
            f"Use a {fast_period}-period fast EMA and a "
            f"{slow_period}-period slow EMA."
        ),
        deps=ctx.deps,
        usage=ctx.usage,
    )
    return result.output


# ------------------------------------------------------------------
# Mandatory qualitative risk reviewer
# ------------------------------------------------------------------

risk_review_agent = Agent(
    model,
    name="risk_review_agent",
    deps_type=TradingDeps,
    output_type=RiskReview,
    instructions="""
You are an independent trading-risk reviewer.

Review the supplied draft rather than creating a new strategy.

Reject plans when:
- entry, stop loss, or take profit are inconsistent;
- risk is not clearly bounded;
- the rationale is unsupported;
- the plan depends on invented information;
- the appropriate action should be HOLD.

Your review is advisory. Hard limits are enforced separately by application
code.
""",
)


# ------------------------------------------------------------------
# Deterministic risk gate
# ------------------------------------------------------------------

def apply_hard_risk_gate(
    draft: TradeDraft,
    deps: TradingDeps,
) -> HardRiskGate:
    """Enforce non-negotiable rules without an LLM."""

    reasons: list[str] = []

    if draft.action == "hold":
        reasons.append("HOLD decisions are not executable.")
        return HardRiskGate(approved=False, reasons=reasons)

    if draft.risk_per_trade_pct > deps.max_risk_per_trade_pct:
        reasons.append(
            f"Risk {draft.risk_per_trade_pct:.2f}% exceeds "
            f"the {deps.max_risk_per_trade_pct:.2f}% limit."
        )

    if (
        draft.entry is None
        or draft.stop_loss is None
        or draft.take_profit is None
    ):
        reasons.append(
            "Executable plans require entry, stop-loss, and take-profit prices."
        )
        return HardRiskGate(approved=False, reasons=reasons)

    if draft.action == "buy":
        if not draft.stop_loss < draft.entry < draft.take_profit:
            reasons.append(
                "For a buy, prices must satisfy "
                "stop_loss < entry < take_profit."
            )

    if draft.action == "sell":
        if not draft.take_profit < draft.entry < draft.stop_loss:
            reasons.append(
                "For a sell, prices must satisfy "
                "take_profit < entry < stop_loss."
            )

    risk_distance = abs(draft.entry - draft.stop_loss)
    reward_distance = abs(draft.take_profit - draft.entry)

    if risk_distance == 0:
        reasons.append("Entry and stop-loss cannot be equal.")
    else:
        reward_risk_ratio = reward_distance / risk_distance

        if reward_risk_ratio < deps.minimum_reward_risk_ratio:
            reasons.append(
                f"Reward/risk ratio {reward_risk_ratio:.2f} is below "
                f"the required {deps.minimum_reward_risk_ratio:.2f}."
            )

    return HardRiskGate(
        approved=not reasons,
        reasons=reasons or ["All deterministic risk checks passed."],
    )


# ------------------------------------------------------------------
# Application orchestration
# ------------------------------------------------------------------

async def create_trade_plan(
    request: str,
    deps: TradingDeps,
) -> tuple[TradingResponse, RunUsage]:
    usage = RunUsage()

    limits = UsageLimits(
        request_limit=12,
        tool_calls_limit=8,
        total_tokens_limit=12_000,
    )

    # Stage 1: coordinator delegates to specialists as needed.
    draft_result = await trading_coordinator.run(
        request,
        deps=deps,
        usage=usage,
        usage_limits=limits,
    )
    draft = draft_result.output

    # Stage 2: this review cannot be skipped by the coordinator.
    review_result = await risk_review_agent.run(
        (
            "Review this proposed trade plan:\n\n"
            f"{draft.model_dump_json(indent=2)}\n\n"
            f"Maximum risk per trade: {deps.max_risk_per_trade_pct}%.\n"
            f"Minimum reward/risk ratio: "
            f"{deps.minimum_reward_risk_ratio}."
        ),
        deps=deps,
        usage=usage,
        usage_limits=limits,
    )
    risk_review = review_result.output

    # Stage 3: ordinary Python enforces absolute rules.
    hard_gate = apply_hard_risk_gate(draft, deps)

    response = TradingResponse(
        draft=draft,
        risk_review=risk_review,
        hard_risk_gate=hard_gate,
        executable=(
            deps.paper_trading
            and risk_review.approved
            and hard_gate.approved
        ),
    )

    return response, usage


async def main() -> None:
    deps = TradingDeps(
        user_id="demo-user",
        paper_trading=True,
        max_risk_per_trade_pct=1.0,
        minimum_reward_risk_ratio=2.0,
    )

    response, usage = await create_trade_plan(
        "Analyze BTCUSDT on the 1h timeframe and propose a trade only if "
        "the EMA evidence is clear.",
        deps,
    )

    print(response.model_dump_json(indent=2))
    print(usage)


if __name__ == "__main__":
    asyncio.run(main())
