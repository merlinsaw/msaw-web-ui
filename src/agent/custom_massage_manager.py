from __future__ import annotations

import json
import logging
from enum import Enum
from typing import List, Optional, Type, Dict, Any

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from termcolor import colored
from ..utils.llm import DeepSeekR1ChatOpenAI
from .custom_prompts import CustomAgentMessagePrompt

logger = logging.getLogger(__name__)

def log_with_color(message: str, level: str = "info", color: str = "white") -> None:
    """Log a message with color and to the logger."""
    print(colored(message, color))
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class CustomMassageManager(MessageManager):
    def __init__(
            self,
            llm: BaseChatModel,
            task: str,
            action_descriptions: str,
            system_prompt_class: Type[SystemPrompt],
            max_input_tokens: int = 128000,
            estimated_tokens_per_character: int = 3,
            image_tokens: int = 800,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            max_actions_per_step: int = 10,
            tool_call_in_content: bool = False,
            use_function_calling: bool = True
    ):
        super().__init__(
            llm=llm,
            task=task,
            action_descriptions=action_descriptions,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            estimated_tokens_per_character=estimated_tokens_per_character,
            image_tokens=image_tokens,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
        )
        self.use_function_calling = use_function_calling
        self.history = MessageHistory()
        
        # Add system message first
        self._add_message_with_tokens(self.system_prompt)
        
        if self.use_function_calling:
            try:
                # Start with a simple system message
                initial_message = SystemMessage(
                    content="Browser automation assistant ready. Waiting for instructions."
                )
                self._add_message_with_tokens(initial_message)
                
                # Add a simple user message to set context
                task_message = HumanMessage(
                    content=f"Task: {self.task}"
                )
                self._add_message_with_tokens(task_message)
                
            except Exception as e:
                log_with_color(f"Error initializing message manager: {e}", level="error", color="red")
                # Fallback to simple message
                fallback_message = SystemMessage(content="Ready to start the task.")
                self._add_message_with_tokens(fallback_message)

    def add_state_message(
            self,
            state: BrowserState,
            result: Optional[List[ActionResult]] = None,
            step_info: Optional[AgentStepInfo] = None,
    ) -> None:
        """Add browser state as human message"""
        try:
            # Get the message from CustomAgentMessagePrompt
            message = CustomAgentMessagePrompt(
                state,
                result,
                include_attributes=self.include_attributes,
                max_error_length=self.max_error_length,
                step_info=step_info,
            ).get_user_message()
            
            # Ensure we have a valid message with string content
            if not isinstance(message.content, str):
                log_with_color("Converting complex content to string", level="warning", color="yellow")
                if isinstance(message.content, (list, dict)):
                    # Handle vision model messages
                    text_content = next((item["text"] for item in message.content if item["type"] == "text"), None)
                    if text_content:
                        message = HumanMessage(content=text_content)
                    else:
                        message = HumanMessage(content=str(message.content))
                else:
                    message = HumanMessage(content=str(message.content))
            
            # Add the message
            self._add_message_with_tokens(message)
            log_with_color("Successfully added state message", level="info", color="green")
            
        except Exception as e:
            log_with_color(f"Error in add_state_message: {e}", level="error", color="red")
            # Fallback to simple message
            fallback_message = HumanMessage(content="Current browser state: Ready for next action")
            self._add_message_with_tokens(fallback_message)

    def cut_messages(self):
        """Get current message list, potentially trimmed to max tokens"""
        diff = self.history.total_tokens - self.max_input_tokens
        while diff > 0 and len(self.history.messages) > 1:
            self.history.remove_message(1) # alway remove the oldest one
            diff = self.history.total_tokens - self.max_input_tokens
        
    def _count_text_tokens(self, text: str) -> int:
        if isinstance(self.llm, (ChatOpenAI, ChatAnthropic, DeepSeekR1ChatOpenAI)):
            try:
                tokens = self.llm.get_num_tokens(text)
            except Exception:
                tokens = len(text) // self.ESTIMATED_TOKENS_PER_CHARACTER
        else:
            tokens = len(text) // self.ESTIMATED_TOKENS_PER_CHARACTER
        return tokens
