from typing import List
from defihunter.core.models import FinalDecision
from defihunter.utils.alerts import TelegramAlerter
import logging

class SignalBroadcaster:
    """
    Orchestrates alerts across multiple channels.
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.alerters = []
        
        if config and hasattr(config, 'alerts'):
            if config.alerts.telegram_token and config.alerts.telegram_chat_id:
                self.alerters.append(TelegramAlerter(
                    token=config.alerts.telegram_token,
                    chat_id=config.alerts.telegram_chat_id
                ))
                self.logger.info("Telegram Alerter initialized.")

    def broadcast(self, decisions: List[FinalDecision]):
        """
        Formats and sends decisions to all active alerters.
        """
        for decision in decisions:
            if decision.decision in ['trade', 'watch']:
                message = self._format_message(decision)
                for alerter in self.alerters:
                    alerter.send_message(message)

    def _format_message(self, d: FinalDecision) -> str:
        """
        Formats a FinalDecision into a readable Telegram markdown message.
        """
        emoji = "🚀" if d.decision == 'trade' else "👀"
        
        msg = (
            f"{emoji} *DeFiHunter Signal: {d.symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Decision*: {d.decision.upper()}\n"
            f"*Composite Score*: `{d.final_trade_score:.1f}`\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Discovery (D)*: `{d.explanation.get('discovery_score', 0):.1f}`\n"
            f"*Entry Readiness (E)*: `{d.explanation.get('entry_readiness', 0):.1f}`\n"
            f"*Fakeout Risk (R)*: `{d.explanation.get('fakeout_risk', 0):.1f}`\n"
            f"*Hold Quality (H)*: `{d.explanation.get('hold_quality', 0):.1f}`\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Leader Prob*: `{d.explanation.get('leader_prob', 0)*100:.0f}%` / {d.explanation.get('family', '—')}\n"
            f"*ML Insights*: {d.explanation.get('ml_explanation', 'N/A')}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Timestamp: `{d.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        return msg
