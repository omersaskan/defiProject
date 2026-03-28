from typing import List
from defihunter.core.models import FinalDecision
from defihunter.utils.alerts import TelegramAlerter
import logging


class SignalBroadcaster:
    """
    Orchestrates alerts across multiple channels.
    Reads from FinalDecision top-level fields (not explanation dict).
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.alerters = []

        if config and hasattr(config, 'alerts'):
            if config.alerts.telegram_token and config.alerts.telegram_chat_id:
                self.alerters.append(TelegramAlerter(
                    token=config.alerts.telegram_token,
                    chat_id=config.alerts.telegram_chat_id,
                ))
                self.logger.info("Telegram Alerter initialized.")

    def broadcast(self, decisions: List[FinalDecision]):
        """Format and send decisions to all active alerters."""
        for decision in decisions:
            if decision.decision in ['trade', 'watch']:
                message = self._format_message(decision)
                for alerter in self.alerters:
                    alerter.send_message(message)

    def _format_message(self, d: FinalDecision) -> str:
        """
        Formats a FinalDecision into a readable Telegram markdown message.
        Uses top-level fields only; explanation is NOT read here.
        """
        emoji = "🚀" if d.decision == 'trade' else "👀"
        family = d.explanation.get('family', '—')  # family stays in explanation

        msg = (
            f"{emoji} *DeFiHunter Signal: {d.symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Decision*: {d.decision.upper()}\n"
            f"*Composite Score*: `{d.composite_leader_score:.1f}`\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Discovery (D)*: `{d.discovery_score:.1f}`\n"
            f"*Entry Readiness (E)*: `{d.entry_readiness:.1f}`\n"
            f"*Fakeout Risk (R)*: `{d.fakeout_risk:.1f}`\n"
            f"*Hold Quality (H)*: `{d.hold_quality:.1f}`\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Leader Prob*: `{d.leader_prob * 100:.0f}%` / {family}\n"
            f"*ML Insights*: {d.explanation.get('ml_explanation', 'N/A')}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Timestamp: `{d.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        return msg
