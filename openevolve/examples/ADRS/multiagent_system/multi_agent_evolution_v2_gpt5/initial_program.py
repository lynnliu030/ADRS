"""
Initial multi-agent system for evolution.
This contains the core multi-agent logic that will be evolved to minimize failure modes.
"""

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None

# ============== Fixed Infrastructure (Not Evolved) ==============

class ExecutionTracer:
    """Comprehensive execution tracer for multi-agent interactions"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.trace_id = 0
        
        if self.log_file:
            # Clear the log file at the start
            with open(self.log_file, 'w') as f:
                f.write(f"Execution Trace Started: {datetime.now()}\n")
                f.write("="*80 + "\n")
    
    def log(self, event_type: str, agent: str, details: str):
        """Log an event to the trace file"""
        self.trace_id += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{self.trace_id:04d}] [{timestamp}] [{event_type}] [{agent}] {details}\n"
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        
        return log_entry

class LLMType(Enum):
    """LLM types"""
    OPENAI = "openai"
    QWEN = "qwen"
    CODELLAMA = "codellama"

class LLMConfig:
    """LLM configuration"""
    def __init__(self):
        self.api_type = LLMType.OPENAI
        self.model = "gpt-4o"
        self.api_key = None
        self.base_url = "https://api.openai.com/v1"
        self.proxy = ""
        self.temperature = 0.7
        self.max_token = 2048

class Config:
    """Configuration object"""
    def __init__(self):
        self.llm = LLMConfig()

if BaseModel:
    class Context(BaseModel):
        """Context object that holds configuration and shared state"""
        config: Config = Field(default_factory=Config)
        cost_manager: Optional[Any] = None
        tracer: Optional[Any] = None
        
        class Config:
            arbitrary_types_allowed = True
    
    class Message(BaseModel):
        """Message object for agent communication"""
        id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        content: str
        instruct_content: Optional[str] = None
        role: str
        cause_by: str = ""
        sent_from: Optional[str] = None
        sent_to: Optional[str] = None
        send_to: Set[str] = Field(default_factory=set)
        
        def __str__(self):
            return f"Message(role={self.role}, content={self.content[:50]}...)"
else:
    # Fallback classes if pydantic is not available
    class Context:
        def __init__(self):
            self.config = Config()
            self.cost_manager = None
            self.tracer = None
    
    class Message:
        def __init__(self, content, role, **kwargs):
            self.id = str(uuid.uuid4())
            self.content = content
            self.instruct_content = kwargs.get('instruct_content')
            self.role = role
            self.cause_by = kwargs.get('cause_by', '')
            self.sent_from = kwargs.get('sent_from')
            self.sent_to = kwargs.get('sent_to')
            self.send_to = kwargs.get('send_to', set())

class LLMInterface:
    """Interface for LLM communication"""
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY", "fake-key")
        self.base_url = config.base_url
    
    async def ask(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to LLM and get response"""
        if not aiohttp:
            # Fallback for testing without actual API calls
            return "I'll help you with that task. Let me write the code for you."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # data = {
        #     "model": self.config.model,
        #     "messages": messages,
        #     "temperature": self.config.temperature,
        #     "max_tokens": self.config.max_token
        # }
        data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature        
            }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        return f"Error: {response.status} - {error_text[:200]}"
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

# EVOLVE-BLOCK-START
# This section contains the multi-agent system logic that will be evolved

class Action(ABC):
    """Base action class"""
    name: str = "Action"
    context: Optional[Context] = None
    llm: Optional[LLMInterface] = None
    
    def __init__(self, **kwargs):
        self.context = kwargs.get('context')
        if self.context and self.context.config.llm:
            self.llm = LLMInterface(self.context.config.llm)
    
    @abstractmethod
    async def run(self, *args, **kwargs):
        """Run the action"""
        pass

class SimpleWriteCode(Action):
    """Action to write code based on requirements"""
    name: str = "SimpleWriteCode"
    
    async def run(self, idea: str) -> str:
        """Generate code based on the idea"""
        if self.context and self.context.tracer:
            self.context.tracer.log("ACTION_START", self.name, f"Writing code for: {idea[:100]}")
        
        prompt = f"""You are a professional programmer. Write Python code for the following task:
Task: {idea}

Requirements:
1. Write clean, functional Python code
2. Include proper error handling
3. Add comments explaining the logic
4. Make it production-ready

Please provide only the code without any explanation."""
        
        messages = [
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": prompt}
        ]
        
        if self.llm:
            code = await self.llm.ask(messages)
        else:
            code = f"# Implementation for: {idea}\n# [Code would be generated here]"
        
        if self.context and self.context.tracer:
            self.context.tracer.log("ACTION_END", self.name, f"Generated {len(code)} characters of code")
        
        return code

class SimpleWriteTest(Action):
    """Action to write tests for code"""
    name: str = "SimpleWriteTest"
    
    async def run(self, code: str) -> str:
        """Generate tests for the given code"""
        if self.context and self.context.tracer:
            self.context.tracer.log("ACTION_START", self.name, "Writing tests for code")
        
        prompt = f"""You are a QA engineer. Write comprehensive tests for the following code:

Code:
{code[:2000]}  # Truncate if too long

Requirements:
1. Write pytest-style test cases
2. Cover edge cases and error conditions
3. Include both positive and negative tests
4. Add docstrings to explain what each test does

Please provide only the test code without any explanation."""
        
        messages = [
            {"role": "system", "content": "You are an expert QA engineer."},
            {"role": "user", "content": prompt}
        ]
        
        if self.llm:
            tests = await self.llm.ask(messages)
        else:
            tests = f"# Tests for the implementation\n# [Tests would be generated here]"
        
        if self.context and self.context.tracer:
            self.context.tracer.log("ACTION_END", self.name, f"Generated {len(tests)} characters of tests")
        
        return tests

class SimpleWriteReview(Action):
    """Action to review code and tests"""
    name: str = "SimpleWriteReview"
    
    def __init__(self, is_human: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.is_human = is_human
    
    async def run(self, code: str, tests: str) -> str:
        """Review the code and tests"""
        if self.context and self.context.tracer:
            self.context.tracer.log("ACTION_START", self.name, f"Reviewing code (human={self.is_human})")
        
        if self.is_human:
            # Simulate human review
            review = "Human review: The code looks good overall. Consider adding more error handling."
        else:
            prompt = f"""You are a senior code reviewer. Review the following code and tests:

Code:
{code[:1500]}

Tests:
{tests[:1500]}

Provide a brief review focusing on:
1. Code quality and best practices
2. Test coverage
3. Potential bugs or issues
4. Suggestions for improvement

Keep your review concise and actionable."""
            
            messages = [
                {"role": "system", "content": "You are a senior software engineer doing code review."},
                {"role": "user", "content": prompt}
            ]
            
            if self.llm:
                review = await self.llm.ask(messages)
            else:
                review = "Review: Code structure looks good. Tests cover main functionality."
        
        if self.context and self.context.tracer:
            self.context.tracer.log("ACTION_END", self.name, f"Review completed: {len(review)} characters")
        
        return review

class Role(ABC):
    """Base role class for agents"""
    name: str = "Role"
    profile: str = "Default"
    context: Optional[Context] = None
    actions: List[Action] = []
    watch_list: List[Type[Action]] = []
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.name)
        self.profile = kwargs.get('profile', self.profile)
        self.context = kwargs.get('context')
        self.is_human = kwargs.get('is_human', False)
        self.actions = []
        self.watch_list = []
    
    def set_actions(self, actions: List[Action]):
        """Set the actions this role can perform"""
        self.actions = actions
    
    def _watch(self, actions: List[Type[Action]]):
        """Set the actions this role watches for"""
        self.watch_list = actions
    
    async def act(self, message: Optional[Message] = None) -> Optional[Message]:
        """Perform an action based on the message"""
        if not self.actions:
            return None
        
        # Execute the first action (simplified)
        action = self.actions[0]
        
        if self.context and self.context.tracer:
            self.context.tracer.log("ROLE_ACT", self.name, f"Executing action: {action.name}")
        
        # Execute action based on type
        if isinstance(action, SimpleWriteCode):
            if message and hasattr(message, 'instruct_content'):
                result = await action.run(message.instruct_content or message.content)
            else:
                result = await action.run("")
        elif isinstance(action, SimpleWriteTest):
            if message:
                result = await action.run(message.content)
            else:
                result = await action.run("")
        elif isinstance(action, SimpleWriteReview):
            # For review, we need both code and tests
            if message:
                # Extract code and tests from previous messages (simplified)
                result = await action.run(message.content, "")
            else:
                result = await action.run("", "")
        else:
            result = "Action completed"
        
        # Create response message
        response = Message(
            content=result,
            role=self.profile,
            cause_by=action.name if action else "",
            sent_from=self.name
        )
        
        if self.context and self.context.tracer:
            self.context.tracer.log("ROLE_COMPLETE", self.name, f"Action completed, message created")
        
        return response

class SimpleCoder(Role):
    """Role that writes code"""
    name: str = "Alice"
    profile: str = "SimpleCoder"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteCode(context=self.context)])

class SimpleTester(Role):
    """Role that writes tests"""
    name: str = "Bob"
    profile: str = "SimpleTester"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteTest(context=self.context)])
        self._watch([SimpleWriteCode])

class SimpleReviewer(Role):
    """Role that reviews code and tests"""
    name: str = "Charlie"
    profile: str = "SimpleReviewer"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteReview(is_human=self.is_human, context=self.context)])
        self._watch([SimpleWriteTest])

class Environment:
    """Environment for multi-agent collaboration"""
    def __init__(self, tracer: Optional[ExecutionTracer] = None):
        self.roles: List[Role] = []
        self.history: List[Message] = []
        self.tracer = tracer
    
    def add_role(self, role: Role):
        """Add a role to the environment"""
        self.roles.append(role)
        if self.tracer:
            self.tracer.log("ENV_ADD_ROLE", "Environment", f"Added role: {role.name} ({role.profile})")
    
    def get_roles(self, profile: Optional[str] = None) -> List[Role]:
        """Get roles by profile"""
        if profile:
            return [r for r in self.roles if r.profile == profile]
        return self.roles
    
    def publish_message(self, message: Message):
        """Publish a message to the environment"""
        self.history.append(message)
        if self.tracer:
            self.tracer.log("ENV_MESSAGE", "Environment", 
                          f"Message from {message.sent_from}: {message.content[:100]}")
    
    def get_messages_for_role(self, role: Role) -> List[Message]:
        """Get messages that a role should respond to"""
        relevant_messages = []
        for msg in self.history:
            # Check if this message is from an action the role watches
            for watched_action in role.watch_list:
                if msg.cause_by == watched_action.name:
                    relevant_messages.append(msg)
                    break
        return relevant_messages

class Team:
    """Team of agents working together"""
    def __init__(self, context: Optional[Context] = None, log_file: Optional[str] = None):
        self.context = context or Context()
        self.tracer = ExecutionTracer(log_file)
        self.context.tracer = self.tracer
        self.env = Environment(self.tracer)
        self.investment: float = 10.0
        self.idea: str = ""
        self.log_file = log_file
    
    def hire(self, roles: List[Role]):
        """Hire roles into the team"""
        for role in roles:
            role.context = self.context
            self.env.add_role(role)
    
    def invest(self, investment: float):
        """Set investment/budget"""
        self.investment = investment
    
    def run_project(self, idea: str):
        """Set the project idea"""
        self.idea = idea
        self.tracer.log("TEAM_START", "Team", f"Starting project: {idea}")
    
    async def run(self, n_round: int = 3):
        """Run the team collaboration for n rounds"""
        self.tracer.log("TEAM_RUN", "Team", f"Running {n_round} rounds")
        
        # Initial message with the idea
        initial_msg = Message(
            content=f"Let's work on this project: {self.idea}",
            instruct_content=self.idea,
            role="Human",
            sent_from="User",
            cause_by="UserInput"
        )
        self.env.publish_message(initial_msg)
        
        for round_num in range(n_round):
            self.tracer.log("ROUND_START", "Team", f"Round {round_num + 1}/{n_round}")
            
            # Each role acts in sequence
            for role in self.env.roles:
                # Determine what messages this role should respond to
                if round_num == 0 and isinstance(role, SimpleCoder):
                    # Coder responds to initial message
                    response = await role.act(initial_msg)
                else:
                    # Other roles respond to relevant messages
                    relevant_msgs = self.env.get_messages_for_role(role)
                    if relevant_msgs:
                        response = await role.act(relevant_msgs[-1])  # Act on most recent relevant message
                    else:
                        continue
                
                if response:
                    self.env.publish_message(response)
            
            self.tracer.log("ROUND_END", "Team", f"Round {round_num + 1} completed")
        
        self.tracer.log("TEAM_END", "Team", "Project completed")
        
        # Final summary
        summary = f"Project '{self.idea}' completed after {n_round} rounds with {len(self.env.history)} messages exchanged."
        self.tracer.log("SUMMARY", "Team", summary)

# EVOLVE-BLOCK-END

# Fixed main execution function (not evolved)
async def run_multi_agent_task(idea: str, n_rounds: int = 3, log_file: str = None):
    """Run a multi-agent task and return the trace"""
    # Create context
    context = Context()
    context.config.llm.api_key = os.getenv("OPENAI_API_KEY", "fake-key")
    
    # Create team
    team = Team(context=context, log_file=log_file)
    team.hire([
        SimpleCoder(context=context),
        SimpleTester(context=context),
        SimpleReviewer(context=context)
    ])
    
    team.invest(investment=3.0)
    team.run_project(idea)
    await team.run(n_round=n_rounds)
    
    # Return the trace content
    if log_file and os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return f.read()
    return ""