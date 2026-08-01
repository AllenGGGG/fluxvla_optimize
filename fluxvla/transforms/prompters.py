# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module defines an abstract base class `PromptBuilder` and
a concrete implementation `PurePrompter` for building multi-turn
prompts for chat-based language models (LLMs). The goal is to
ensure consistent formatting across turns and models, with special
handling for different roles (e.g., human vs. LLM).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils import build_tokenizer_from_cfg


class PromptBuilder(ABC):
    """
    Abstract base class for multi-turn prompt builders.

    Subclasses are expected to implement methods for adding
    new dialogue turns, retrieving the current prompt, and
    generating a potential prompt with a new user message.
    This design allows different LLM prompt formats (e.g.,
    OpenAI, LLaMA, ChatGLM) to be encapsulated under a unified
    interface.

    Args:
        model_family (str): Identifier of the target LLM
            family (e.g., "llama", "chatglm").
        system_prompt (Optional[str]): Optional system-level
            instruction to include (if applicable).
    """

    def __init__(self,
                 model_family: str,
                 system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str:
        """
        Add a turn to the prompt.

        Args:
            role (str): Either "human" or "gpt" to indicate the
                speaker of the turn.
            message (str): The actual message content.

        Returns:
            str: The formatted string that was added to the prompt.
        """
        ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> str:
        """
        Return the prompt that would result if the given user message
        were added next.

        Args:
            user_msg (str): The new user message to hypothetically
                append.

        Returns:
            str: The resulting prompt string with the hypothetical
                message.
        """
        ...

    @abstractmethod
    def get_prompt(self) -> str:
        """
        Return the current full prompt.

        Returns:
            str: The constructed prompt up to the current point in
                the conversation.
        """
        ...


@TRANSFORMS.register_module()
class PurePrompter(PromptBuilder):
    """
    A simple prompt builder that formats chat turns using basic
    prefix/suffix logic.

    This implementation assumes a strict alternation between "human"
    and "gpt" turns, and uses newline-based formatting to separate
    input/output messages. Suitable for models like LLaMA where prompt
    formatting is minimal or customizable.

    Args:
        model_family (str): Identifier for the target model family.
        system_prompt (Optional[str]): Optional system-level
            instruction to include.
    """

    def __init__(self,
                 model_family: str,
                 system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        self.bos, self.eos = '<s>', '</s>'

        self.wrap_human = lambda msg: f'In: {msg}\nOut: '
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        self.prompt, self.turn_count = '', 0

    def add_turn(self, role: str, message: str) -> str:
        """
        Add a new message to the prompt from the given role.

        Args:
            role (str): Either "human" or "gpt", based on turn alternation.
            message (str): The message to add (image tokens will be
                stripped).

        Returns:
            str: The formatted message that was appended to the prompt.
        """
        assert (role == 'human') if (self.turn_count %
                                     2 == 0) else (role == 'gpt')
        message = message.replace('<image>', '').strip()

        if (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)

        self.prompt += wrapped_message
        self.turn_count += 1

        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        """
        Return a copy of the current prompt with an additional
        (hypothetical) user message.

        Args:
            message (str): New user message to hypothetically append.

        Returns:
            str: The full prompt with the hypothetical turn included.
        """
        prompt_copy = str(self.prompt)
        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        """
        Return the actual constructed prompt so far.

        Returns:
            str: The current prompt string without the BOS token.
        """
        return self.prompt.removeprefix(self.bos).rstrip()

    def clear(self) -> None:
        """
        Clear the current prompt and reset the turn count.
        """
        self.prompt = ''
        self.turn_count = 0


@TRANSFORMS.register_module()
class ParquetPrompter:
    """Private Prompter for Libero dataset.
    This prompter generates prompts for the Libero dataset
    based on the provided data. It formats the prompt
    to include the task description and any additional
    information from the data.
    """

    def __init__(self,
                 action_tokenizer: Optional[dict] = None,
                 use_conversation: bool = True,
                 add_new_line: bool = False,
                 lowercase_task_description: bool = False,
                 *args,
                 **kwargs):
        self.prompt = ''
        self.turn_count = 0
        self.wrap_human = lambda msg: f'In: {msg}\nOut: '
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        self.bos, self.eos = '<s>', '</s>'
        if action_tokenizer is not None:
            self.action_tokenizer = build_tokenizer_from_cfg(action_tokenizer)
        else:
            self.action_tokenizer = None
        self.use_conversation = use_conversation
        self.add_new_line = add_new_line
        self.lowercase_task_description = lowercase_task_description

    def __call__(self, inputs):
        assert 'task_description' in inputs, \
            "Data must contain 'task_description' key."
        task_description = inputs['task_description']
        if self.lowercase_task_description:
            task_description = str(task_description).lower()
        if not self.use_conversation:
            prompt = task_description
        else:
            conversation = [
                {
                    'from':
                    'human',
                    'value':
                    f'What action should the robot take to {task_description}?'  # noqa: E501
                },
            ]
            actions = inputs['actions']
            if self.action_tokenizer is not None:
                conversation.append({
                    'from': 'gpt',
                    'value': self.action_tokenizer(actions[0])
                })
            for turn in conversation:
                self.add_turn(turn['from'], turn['value'])
            # Generate a prompt based on the task description and data
            prompt = self.prompt.removeprefix(self.bos).rstrip()
            self.prompt = ''
            self.turn_count = 0
        if self.add_new_line:
            prompt += '\n'
        inputs['prompt'] = prompt
        return inputs

    def add_turn(self, role: str, message: str) -> str:
        """
        Add a new message to the prompt from the given role.

        Args:
            role (str): Either "human" or "gpt", based on turn alternation.
            message (str): The message to add (image tokens will be
                stripped).

        Returns:
            str: The formatted message that was appended to the prompt.
        """
        assert (role == 'human') if (self.turn_count %
                                     2 == 0) else (role == 'gpt')
        message = message.replace('<image>', '').strip()

        if (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)

        self.prompt += wrapped_message
        self.turn_count += 1

        return wrapped_message


@TRANSFORMS.register_module()
class EpisodeMetadataPrompter:
    """Appends pi0.7-style episode metadata to `task_description`.

    Must run BEFORE any transform that wraps `task_description` into the
    final `prompt` string (e.g. `ParquetPrompter`, `PreparePromptWithState`),
    mirroring the pi0.7 prompt layout where metadata sits alongside the task
    text inside the context, not after an "Out:"/"Action:" completion
    marker:

        Task: peel vegetables. Subtask: pick up the peeler.
        Speed: 1.25. Advantage: true. Control Mode: joint.

    This is a general-purpose transform (not task/embodiment-specific): the
    same class is used for both training (with dropout) and eval (fixed
    desired values, no dropout) by toggling `training`.

    Speed text:
        Read verbatim from `inputs['tempo_speed']` -- the speed multiplier
        `VSTADataset`/`VSTAPackedParquetDatasetV3` sampled for this window's
        VSTA augmentation (tempo_speed > 1 -> faster execution). Omitted
        entirely when `tempo_speed` is absent (no VSTA augmentation applied
        to this sample).

    Advantage text:
        Read verbatim from the per-frame `inputs['advantage']` column (1/0),
        written directly into the main parquet by the data-collection pipeline
        (ros2_data_collection's recorder_node / Phase 2's VLM boundary
        correction) -- not from `episode_metadata`, which only ever held a
        coarser episode-level judgement and no longer carries advantage at
        all. No merge step: HF `datasets.load_dataset('parquet', ...)` exposes
        every parquet column verbatim in the row dict, so this value is
        already present on `inputs` by the time this transform runs.

    Args:
        training (bool): If True, apply block/field dropout using sampled
            values from `inputs`. If False (eval), use `desired_speed` /
            `desired_advantage` verbatim and never drop anything.
        block_dropout_prob (float): Probability of omitting the entire
            metadata block (training only).
        field_dropout_prob (float): Probability of omitting Speed/Advantage
            individually (training only; Control Mode is never dropped).
        control_mode (str | None): Static text identifier (e.g. 'joint',
            'ee'). If None, the Control Mode field is omitted entirely.
        desired_speed (float | None): Eval-only fixed Speed value (a
            tempo_speed multiplier, e.g. 1.0 for normal speed).
        desired_advantage (bool | None): Eval-only fixed Advantage value.
        use_speed (bool): If False, never read/emit the Speed field,
            regardless of what `inputs` contains -- for datasets/runs that
            don't carry a meaningful `tempo_speed` signal.
        use_advantage (bool): If False, never read/emit the Advantage
            field, regardless of what `inputs` contains -- for
            datasets/runs that don't carry a meaningful `advantage` signal.
        seed (int | None): RNG seed for dropout sampling.
    """

    def __init__(self,
                 training: bool = True,
                 block_dropout_prob: float = 0.15,
                 field_dropout_prob: float = 0.05,
                 control_mode: Optional[str] = 'joint',
                 desired_speed: Optional[float] = None,
                 desired_advantage: Optional[bool] = None,
                 use_speed: bool = True,
                 use_advantage: bool = True,
                 seed: Optional[int] = None,
                 *args,
                 **kwargs) -> None:
        self.training = training
        self.block_dropout_prob = block_dropout_prob
        self.field_dropout_prob = field_dropout_prob
        self.control_mode = control_mode
        self.desired_speed = desired_speed
        self.desired_advantage = desired_advantage
        self.use_speed = use_speed
        self.use_advantage = use_advantage
        self._rng = np.random.default_rng(seed)

    def _speed_value(self, inputs: Dict[str, Any]) -> Optional[float]:
        if not self.use_speed:
            return None
        if not self.training:
            return self.desired_speed

        tempo_speed = inputs.get('tempo_speed')
        return float(tempo_speed) if tempo_speed is not None else None

    def _advantage(self, inputs: Dict[str, Any]) -> Optional[bool]:
        if not self.use_advantage:
            return None
        if not self.training:
            return self.desired_advantage

        advantage = inputs.get('advantage')
        return bool(advantage) if advantage is not None else None

    def _keep_field(self) -> bool:
        if not self.training:
            return True
        return self._rng.random() >= self.field_dropout_prob

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        assert 'task_description' in inputs, \
            "Data must contain 'task_description' key."
        assert 'prompt' not in inputs, (
            'EpisodeMetadataPrompter must run BEFORE the transform that '
            "builds inputs['prompt'] (e.g. ParquetPrompter, "
            'PreparePromptWithState) — it appends to task_description, '
            'which is read too late otherwise and the metadata silently '
            'never reaches the final prompt.')

        if self.training and self._rng.random() < self.block_dropout_prob:
            return inputs

        parts = []

        speed_value = self._speed_value(inputs)
        if speed_value is not None and self._keep_field():
            parts.append(f'Speed: {speed_value}.')

        advantage = self._advantage(inputs)
        if advantage is not None and self._keep_field():
            parts.append(f"Advantage: {'true' if advantage else 'false'}.")

        if self.control_mode is not None:
            parts.append(f'Control Mode: {self.control_mode}.')

        if parts:
            task_description = str(inputs['task_description']).rstrip()
            inputs['task_description'] = (
                f"{task_description} {' '.join(parts)}")

        return inputs


@TRANSFORMS.register_module()
class PreparePromptWithState():
    """Prepare prompt with state for PI05.
    This transform prepares the prompt with state for PI05.
    It formats the prompt to include the task description and state.

    Args:
        max_state_dim (int): The maximum dimension of the state.
        task_key (str): The key of the task description in the input data.
    """

    def __init__(self,
                 max_state_dim: int = 32,
                 task_key: str = 'task_description',
                 *args,
                 **kwargs):
        self.max_state_dim = max_state_dim
        self.task_key = task_key

    def __call__(self, inputs: Dict) -> Dict:
        state = inputs['states']
        if state is None:
            raise ValueError('State is required for PI05')
        assert self.task_key in inputs, \
            f"Data must contain '{self.task_key}' key."
        task_description = inputs[self.task_key]

        # Prepare state (pad to max_state_dim)
        state_padded = np.zeros(self.max_state_dim)
        state_padded[:state.shape[0]] = state

        # State should already be normalized to [-1, 1]
        # by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins
        # (see openpi `PaligemmaTokenizer.tokenize()`)
        discretized_states = np.digitize(
            state_padded, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        cleaned_text = task_description.strip().replace('_', ' ').replace(
            '\n', ' ')
        state_str = ' '.join(map(str, discretized_states))
        action_prefix = ';\nAction: '
        full_prompt = (
            f'Task: {cleaned_text}, State: {state_str}{action_prefix}')

        inputs['prompt'] = full_prompt
        # Normalize state to [-1, 1] range if needed
        # (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return inputs
