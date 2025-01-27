import pdb
from typing import List, Optional

from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import ActionResult
from browser_use.browser.views import BrowserState
from langchain_core.messages import HumanMessage, SystemMessage

from .custom_views import CustomAgentStepInfo


class CustomSystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = """
    1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
       {
         "current_state": {
           "prev_action_evaluation": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not. Note that the result you output must be consistent with the reasoning you output afterwards. If you consider it to be 'Failed,' you should reflect on this during your thought.",
           "important_contents": "Output important contents closely related to user\'s instruction or task on the current page. If there is, please output the contents. If not, please output empty string ''.",
           "task_progress": "Task Progress is a general summary of the current contents that have been completed. Just summarize the contents that have been actually completed based on the content at current step and the history operations. Please list each completed item individually, such as: 1. Input username. 2. Input Password. 3. Click confirm button. Please return string type not a list.",
           "future_plans": "Based on the user's request and the current state, outline the remaining steps needed to complete the task. This should be a concise list of actions yet to be performed, such as: 1. Select a date. 2. Choose a specific time slot. 3. Confirm booking. Please return string type not a list.",
           "thought": "Think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation. If your output of prev_action_evaluation is 'Failed', please reflect and output your reflection here.",
           "summary": "Please generate a brief natural language description for the operation in next actions based on your Thought."
         },
         "action": [
           {
             "action_name": {
               // action-specific parameters
             }
           },
           // ... more actions in sequence
         ]
       }

    2. ACTIONS: You can specify multiple actions to be executed in sequence. 

       Common action sequences:
       - Form filling: [
           {"input_text": {"index": 1, "text": "username"}},
           {"input_text": {"index": 2, "text": "password"}},
           {"click_element": {"index": 3}}
         ]
       - Navigation and extraction: [
           {"open_new_tab": {}},
           {"go_to_url": {"url": "https://example.com"}},
           {"extract_page_content": {}}
         ]


    3. ELEMENT INTERACTION:
       - Only use indexes that exist in the provided element list
       - Each element has a unique index number (e.g., "33[:]<button>")
       - Elements marked with "_[:]" are non-interactive (for context only)

    4. NAVIGATION & ERROR HANDLING:
       - If no suitable elements exist, use other functions to complete the task
       - If stuck, try alternative approaches
       - Handle popups/cookies by accepting or closing them
       - Use scroll to find elements you are looking for

    5. TASK COMPLETION:
       - If you think all the requirements of user\'s instruction have been completed and no further operation is required, output the done action to terminate the operation process.
       - Don't hallucinate actions.
       - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
       - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.
       - Note that you must verify if you've truly fulfilled the user's request by examining the actual page content, not just by looking at the actions you output but also whether the action is executed successfully. Pay particular attention when errors occur during action execution.

    6. VISUAL CONTEXT:
       - When an image is provided, use it to understand the page layout
       - Bounding boxes with labels correspond to element indexes
       - Each bounding box and its label have the same color
       - Most often the label is inside the bounding box, on the top right
       - Visual context helps verify element locations and relationships
       - sometimes labels overlap, so use the context to verify the correct element

    7. Form filling:
       - If you fill an input field and your action sequence is interrupted, most often a list with suggestions poped up under the field and you need to first select the right element from the suggestion list.

    8. ACTION SEQUENCING:
       - Actions are executed in the order they appear in the list 
       - Each action should logically follow from the previous one
       - If the page changes after an action, the sequence is interrupted and you get the new state.
       - If content only disappears the sequence continues.
       - Only provide the action sequence until you think the page will change.
       - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page like saving, extracting, checkboxes...
       - only use multiple actions if it makes sense. 
    """
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
    INPUT STRUCTURE:
    1. Task: The user\'s instructions you need to complete.
    2. Hints(Optional): Some hints to help you complete the user\'s instructions.
    3. Memory: Important contents are recorded during historical operations for use in subsequent operations.
    4. Current URL: The webpage you're currently on
    5. Available Tabs: List of open browser tabs
    6. Interactive Elements: List in the format:
       index[:]<element_type>element_text</element_type>
       - index: Numeric identifier for interaction
       - element_type: HTML element type (button, input, etc.)
       - element_text: Visible text or element description

    Example:
    33[:]<button>Submit Form</button>
    _[:] Non-interactive text


    Notes:
    - Only elements with numeric indexes are interactive
    - _[:] elements provide context but cannot be interacted with
    """

    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.

        Returns:
            str: Formatted system prompt
        """
        time_str = self.current_date.strftime("%Y-%m-%d %H:%M")

        AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
    1. Analyze the provided webpage elements and structure
    2. Plan a sequence of actions to accomplish the given task
    3. Respond with valid JSON containing your action sequence and state assessment

    Current date and time: {time_str}

    {self.input_format()}

    {self.important_rules()}

    Functions:
    {self.default_action_description}

    Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid."""
        return SystemMessage(content=AGENT_PROMPT)


class CustomAgentMessagePrompt:
    def __init__(
            self,
            state: BrowserState,
            result: Optional[List[ActionResult]] = None,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            step_info: Optional[CustomAgentStepInfo] = None,
    ):
        self.state = state
        self.result = result
        self.max_error_length = max_error_length
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message(self) -> HumanMessage:
        """Format browser state as a human message"""
        try:
            # Build a clean state description
            state_parts = []
            
            # Add URL information
            if hasattr(self.state, 'current_url'):
                state_parts.append(f"Current URL: {self.state.current_url}")
            
            # Add page title if available
            if hasattr(self.state, 'page_title'):
                state_parts.append(f"Page Title: {self.state.page_title}")
            
            # Add any error information
            if self.result and any(r.error for r in self.result):
                errors = [r.error for r in self.result if r.error]
                truncated_errors = [e[:self.max_error_length] + '...' if len(e) > self.max_error_length else e for e in errors]
                state_parts.append("Errors: " + '; '.join(truncated_errors))
            
            # Add step information if available
            if self.step_info:
                state_parts.append(f"Step {self.step_info.step_number}/{self.step_info.max_steps}")
                if self.step_info.task:
                    state_parts.append(f"Task: {self.step_info.task}")
                if self.step_info.add_infos:
                    state_parts.append(f"Additional Info: {self.step_info.add_infos}")
            
            # Join all parts with newlines
            content = '\n'.join(state_parts)
            
            # Create a clean human message
            return HumanMessage(content=content)
            
        except Exception as e:
            logger.error(f"Error formatting user message: {e}")
            # Return a simple message if anything fails
            return HumanMessage(content="Error occurred while formatting browser state")